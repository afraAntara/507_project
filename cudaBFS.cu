#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <time.h>

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5)

Graph *initGraph(int numVertices, GraphDensity density)
{
  Graph *graph = (Graph *)malloc(sizeof(Graph));
  graph->numVertices = numVertices;
  graph->numEdges = 0; // This will be calculated in generateEdges

  // Allocate memory for arrays
  graph->adjacencyList = (int *)malloc(numVertices * numVertices * sizeof(int));
  graph->edgesOffset = (int *)malloc(numVertices * sizeof(int));
  graph->edgesSize = (int *)malloc(numVertices * sizeof(int));

  // Initialize arrays
  for (int i = 0; i < numVertices; i++)
  {
    graph->edgesOffset[i] = i * numVertices;
    graph->edgesSize[i] = 0;
  }

  // Generate edges based on density
  generateEdges(graph, density);

  return graph;
}
void generateEdges(Graph *graph, GraphDensity density)
{
  srand(time(NULL)); // Seed the random number generator
  int maxAdditionalEdges;

  // First, create a chain to ensure connectivity
  for (int i = 0; i < graph->numVertices - 1; i++)
  {
    // Connect vertex i to i + 1
    graph->adjacencyList[graph->edgesOffset[i] + graph->edgesSize[i]] = i + 1;
    graph->edgesSize[i]++;
    graph->numEdges++;
  }

  // Determine the maximum number of additional edges based on density
  switch (density)
  {
  case Sparse:
    maxAdditionalEdges = graph->numVertices;
    break;
  case Medium:
    maxAdditionalEdges = graph->numVertices * 2;
    break;
  case Dense:
    maxAdditionalEdges = graph->numVertices * (graph->numVertices - 1) / 2;
    break;
  }

  // Add additional edges
  for (int i = 0; i < maxAdditionalEdges; i++)
  {
    int v = rand() % graph->numVertices;
    int w = rand() % graph->numVertices;

    // Avoid self-loops and duplicate edges
    if (v != w && graph->edgesSize[v] < graph->numVertices - 1)
    {
      // Add edge from v to w
      graph->adjacencyList[graph->edgesOffset[v] + graph->edgesSize[v]] = w;
      graph->edgesSize[v]++;

      // Increment the total edge count
      graph->numEdges++;
    }
  }
}

// Function to free the graph from memory
void freeGraph(Graph *graph)
{
  free(graph->adjacencyList);
  free(graph->edgesOffset);
  free(graph->edgesSize);
  free(graph);
}

__global__ void initializeDeviceArray(int n, int *d_arr, int value, int start_index)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == start_index)
  {
    d_arr[start_index] = 0;
  }
  else if (tid < n)
  {
    d_arr[tid] = value;
  }
}

__global__ void printDeviceArray(int *d_arr, int n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    printf("d_arr[%i] = %i \n", tid, d_arr[tid]);
  }
}

__global__ void computeNextQueue(int *adjacencyList, int *edgesOffset, int *edgesSize, int *distance,
                                 int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < queueSize)
  {
    int current = currentQueue[tid];
    for (int i = edgesOffset[current]; i < edgesOffset[current] + edgesSize[current]; ++i)
    {
      int v = adjacencyList[i];
      if (distance[v] == INT_MAX)
      {
        distance[v] = level + 1;
        int position = atomicAdd(nextQueueSize, 1);
        nextQueue[position] = v;
      }
    }
  }
}
void bfsGPU(int start, Graph *G, int *distance)
{
  const int n_blocks = (G->numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

  // Initialization of GPU variables
  int *d_adjacencyList, *d_edgesOffset, *d_edgesSize, *d_firstQueue, *d_secondQueue, *d_nextQueueSize, *d_distance;

  // Allocation on device
  const int size = G->numVertices * sizeof(int);
  const int adjacencySize = G->numVertices * G->numVertices * sizeof(int); // Assuming the adjacency list is a full matrix
  cudaMalloc((void **)&d_adjacencyList, adjacencySize);
  cudaMalloc((void **)&d_edgesOffset, size);
  cudaMalloc((void **)&d_edgesSize, size);
  cudaMalloc((void **)&d_firstQueue, size);
  cudaMalloc((void **)&d_secondQueue, size);
  cudaMalloc((void **)&d_distance, size);
  cudaMalloc((void **)&d_nextQueueSize, sizeof(int));

  // Copy inputs to device
  cudaMemcpy(d_adjacencyList, G->adjacencyList, adjacencySize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgesOffset, G->edgesOffset, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_edgesSize, G->edgesSize, size, cudaMemcpyHostToDevice);

  // Initialize distance array on host and copy to device
  for (int i = 0; i < G->numVertices; ++i)
  {
    distance[i] = INT_MAX;
  }
  distance[start] = 0;
  cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice);

  int currentQueueSize = 1, level = 0;

  while (currentQueueSize > 0)
  {
    int *d_currentQueue, *d_nextQueue;
    if (level % 2 == 0)
    {
      d_currentQueue = d_firstQueue;
      d_nextQueue = d_secondQueue;
    }
    else
    {
      d_currentQueue = d_secondQueue;
      d_nextQueue = d_firstQueue;
    }

    computeNextQueue<<<n_blocks, N_THREADS_PER_BLOCK>>>(d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                                                        currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);
    cudaDeviceSynchronize();
    ++level;
    cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
    int resetQueueSize = 0;
    cudaMemcpy(d_nextQueueSize, &resetQueueSize, sizeof(int), cudaMemcpyHostToDevice);
  }

  cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_adjacencyList);
  cudaFree(d_edgesOffset);
  cudaFree(d_edgesSize);
  cudaFree(d_firstQueue);
  cudaFree(d_secondQueue);
  cudaFree(d_distance);
  cudaFree(d_nextQueueSize);
}
int main()
{
  int numVertices = 500; // Example number of vertices
  Graph *myGraph = initGraph(numVertices, Sparse);

  // printf("Graph's Adjacency List:\n");
  // printGraph(myGraph);

  // Allocate memory for BFS distance and visited arrays
  int *distance = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));
  clock_t startSerial, endSerial, startParallel, endParallel;
  startParallel = clock();
  bfsGPU(0, myGraph, distance);
  endParallel = clock();
  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
  printf("Cuda BFS took %f seconds.\n", timeTakenParallel);
}
