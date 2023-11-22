#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;
#define bool _Bool
typedef struct
{
  int size;
  int **matrix;
} Graph;

__global__ void bfsKernel(int *dev_matrix, int graphSize, int *currentQueue, int currentQueueSize, int *nextQueue, int *nextQueueSize, int *distance, int level)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < currentQueueSize)
  {
    int vertex = currentQueue[tid];
    for (int i = 0; i < graphSize; i++)
    {
      if (dev_matrix[vertex * graphSize + i] && distance[i] == INT_MAX)
      {
        distance[i] = 1;
        int position = atomicAdd(nextQueueSize, 1);
        nextQueue[position] = i;
      }
    }
  }
}

void bfsGPU(Graph *g, int startVertex, int *distance)
{
  const int N_THREADS_PER_BLOCK = 1024;
  int graphSize = g->size;

  // Flatten 2D matrix into a 1D array for CUDA
  int *flatMatrix = (int *)malloc(graphSize * graphSize * sizeof(int));
  for (int i = 0; i < graphSize; i++)
  {
    for (int j = 0; j < graphSize; j++)
    {
      flatMatrix[i * graphSize + j] = g->matrix[i][j];
    }
  }

  // GPU variables
  int *dev_matrix, *dev_currentQueue, *dev_nextQueue, *dev_nextQueueSize, *dev_distance;

  // Allocate memory on device
  cudaMalloc((void **)&dev_matrix, graphSize * graphSize * sizeof(int));
  cudaMalloc((void **)&dev_currentQueue, graphSize * sizeof(int));
  cudaMalloc((void **)&dev_nextQueue, graphSize * sizeof(int));
  cudaMalloc((void **)&dev_nextQueueSize, sizeof(int));
  cudaMalloc((void **)&dev_distance, graphSize * sizeof(int));

  // Copy graph to device
  cudaMemcpy(dev_matrix, flatMatrix, graphSize * graphSize * sizeof(int), cudaMemcpyHostToDevice);

  // Initialize variables
  int currentQueueSize = 1;
  int nextQueueSize = 0;
  int level = 0;

  // Initialize distances and copy to device
  for (int i = 0; i < graphSize; i++)
  {
    distance[i] = INT_MAX;
  }
  distance[startVertex] = 0;
  cudaMemcpy(dev_distance, distance, graphSize * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_currentQueue, &startVertex, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_nextQueueSize, &nextQueueSize, sizeof(int), cudaMemcpyHostToDevice);

  // Main BFS loop
  while (currentQueueSize > 0)
  {
    int n_blocks = (currentQueueSize + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
    bfsKernel<<<n_blocks, N_THREADS_PER_BLOCK>>>(dev_matrix, graphSize, dev_currentQueue, currentQueueSize, dev_nextQueue, dev_nextQueueSize, dev_distance, level);
    cudaDeviceSynchronize();

    // Swap queues
    int *temp = dev_currentQueue;
    dev_currentQueue = dev_nextQueue;
    dev_nextQueue = temp;

    // Update queue sizes and level
    cudaMemcpy(&currentQueueSize, dev_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
    nextQueueSize = 0;
    cudaMemcpy(dev_nextQueueSize, &nextQueueSize, sizeof(int), cudaMemcpyHostToDevice);
    ++level;
  }

  cudaMemcpy(distance, dev_distance, graphSize * sizeof(int), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(dev_matrix);
  cudaFree(dev_currentQueue);
  cudaFree(dev_nextQueue);
  cudaFree(dev_nextQueueSize);
  cudaFree(dev_distance);
  free(flatMatrix);
}
Graph generateGraph(int size)
{
  Graph g;
  g.size = size;
  g.matrix = (int **)malloc(size * sizeof(int *));
  for (int i = 0; i < size; i++)
  {
    g.matrix[i] = (int *)malloc(size * sizeof(int));
    for (int j = 0; j < size; j++)
    {
      g.matrix[i][j] = rand() % 2;
    }
  }
  return g;
}

#define QUEUE_SIZE 50000

typedef struct
{
  int items[QUEUE_SIZE];
  int front;
  int rear;
} Queue;

Queue *createQueue()
{
  Queue *q = (Queue *)malloc(sizeof(Queue));
  q->front = -1;
  q->rear = -1;
  return q;
}

bool isQueueEmpty(Queue *q)
{
  return q->rear == -1;
}

void enqueue(Queue *q, int value)
{
  if (q->rear == QUEUE_SIZE - 1)
    return;
  else
  {
    if (q->front == -1)
      q->front = 0;
    q->rear++;
    q->items[q->rear] = value;
  }
}

int dequeue(Queue *q)
{
  int item;
  if (isQueueEmpty(q))
  {
    item = -1;
  }
  else
  {
    item = q->items[q->front];
    q->front++;
    if (q->front > q->rear)
    {
      q->front = q->rear = -1;
    }
  }
  return item;
}

void printGraph(Graph *g)
{
  printf("Adjacency Matrix:\n");
  for (int i = 0; i < g->size; i++)
  {
    for (int j = 0; j < g->size; j++)
    {
      printf("%d ", g->matrix[i][j]);
    }
    printf("\n");
  }
}

void printBFS(int *visited, int size)
{
  printf("Node traversed:\n");
  for (int i = 0; i < size; i++)
  {
    printf("%d ", visited[i]);
  }
  printf("\n");
}

void serialBFS(Graph *g, int startVertex, int *visited);
void parallelBFS(Graph *g, int startVertex, int *visited);
void cudaParallelBFS(Graph *g, int startVertex, int *visited);

int main()
{
  int graphSize = 500;
  Graph g = generateGraph(graphSize);
  int *visited = (int *)malloc(graphSize * sizeof(int));

  for (int i = 0; i < graphSize; i++)
  {
    visited[i] = 0;
  }

  printf("Graph size: %d\n", graphSize);

  // printGraph(&g);

  clock_t startSerial = clock();
  serialBFS(&g, 0, visited);
  clock_t endSerial = clock();
  double timeSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
  printf("Serial BFS Version took %f seconds\n", timeSerial);

  printBFS(visited, graphSize);

  for (int i = 0; i < graphSize; i++)
  {
    visited[i] = 0;
  }

  clock_t startParallel = clock();
  parallelBFS(&g, 0, visited);
  clock_t endParallel = clock();
  double timeParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
  printf("parallelBFS Version took %f seconds\n", timeParallel);

  printBFS(visited, graphSize);

  for (int i = 0; i < graphSize; i++)
  {
    visited[i] = 0;
  }

  clock_t startCUDA = clock();
  bfsGPU(&g, 0, visited);
  clock_t endCUDA = clock();
  double timeCUDA = (double)(endCUDA - startCUDA) / CLOCKS_PER_SEC;

  printf("CUDA Parallel BFS Version took %f seconds\n", timeCUDA);

  printBFS(visited, graphSize);
  free(g.matrix);
  free(visited);

  return 0;
}

void serialBFS(Graph *g, int startVertex, int *visited)
{
  Queue *q = createQueue();
  visited[startVertex] = 1;
  enqueue(q, startVertex);

  while (!isQueueEmpty(q))
  {
    int currentVertex = dequeue(q);
    for (int i = 0; i < g->size; i++)
    {
      if (g->matrix[currentVertex][i] && !visited[i])
      {
        visited[i] = 1;
        enqueue(q, i);
      }
    }
  }
  free(q);
}

void parallelBFS(Graph *g, int startVertex, int *visited)
{
  Queue *q = createQueue();
  visited[startVertex] = 1;
  enqueue(q, startVertex);

  while (!isQueueEmpty(q))
  {
    int currentVertex = dequeue(q);

#pragma omp parallel for
    for (int i = 0; i < g->size; i++)
    {
      if (g->matrix[currentVertex][i] && !visited[i])
      {
#pragma omp critical
        {
          if (!visited[i])
          {
            visited[i] = 1;
            enqueue(q, i);
          }
        }
      }
    }
  }
  free(q);
}
