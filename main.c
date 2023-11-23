#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
// Function to print the graph
void printGraph(Graph *graph)
{
  printf("Graph with %d vertices and %d edges\n", graph->numVertices, graph->numEdges);

  for (int i = 0; i < graph->numVertices; i++)
  {
    printf("Vertex %d: ", i);
    for (int j = 0; j < graph->edgesSize[i]; j++)
    {
      printf("%d ", graph->adjacencyList[graph->edgesOffset[i] + j]);
    }
    printf("\n");
  }
}

void bfsCPU(int start, Graph *G, int *distance, int *visited)
{
  for (int i = 0; i < G->numVertices; i++)
  {
    distance[i] = INT_MAX;
    visited[i] = 0;
  }
  distance[start] = 0;

  int *queue = (int *)malloc(G->numVertices * sizeof(int)); // Queue
  int front = 0, rear = 0;                                  // Queue front and rear

  // Enqueue start vertex
  queue[rear++] = start;

  while (front < rear)
  {
    int current = queue[front++];
    for (int i = G->edgesOffset[current]; i < G->edgesOffset[current] + G->edgesSize[current]; ++i)
    {
      int v = G->adjacencyList[i];
      if (distance[v] == INT_MAX)
      {
        distance[v] = distance[current] + 1;
        queue[rear++] = v;
      }
    }
  }

  free(queue);
}
void ompBFS(int start, Graph *G, int *distance); // Make sure this is declared

int main()
{
  int numVertices = 30000; // Example number of vertices
  Graph *myGraph = initGraph(numVertices, Dense);

  printf("Graph with %d vertices and %d edges\n", myGraph->numVertices, myGraph->numEdges);
  // printf("Graph's Adjacency List:\n");
  // printGraph(myGraph);

  // Allocate memory for BFS distance and visited arrays
  int *distance = (int *)malloc(numVertices * sizeof(int));
  int *visited = (int *)malloc(numVertices * sizeof(int));

  // Timing variables
  clock_t startSerial, endSerial, startParallel, endParallel;

  // Perform standard BFS
  startSerial = clock();
  bfsCPU(0, myGraph, distance, visited);
  endSerial = clock();
  double timeTakenSerial = (double)(endSerial - startSerial) / CLOCKS_PER_SEC;
  printf("Standard BFS took %f seconds.\n", timeTakenSerial);
  // printf("Distances from vertex 0:\n");
  // for (int i = 0; i < numVertices; i++)
  // {
  //   printf("Vertex %d: %d\n", i, distance[i]);
  // }

  // Reset distance and visited for ompBFS
  for (int i = 0; i < numVertices; i++)
  {
    distance[i] = INT_MAX;
    visited[i] = 0;
  }

  // Perform OpenMP BFS
  startParallel = clock();
  ompBFS(0, myGraph, distance);
  endParallel = clock();
  double timeTakenParallel = (double)(endParallel - startParallel) / CLOCKS_PER_SEC;
  printf("OpenMP BFS took %f seconds.\n", timeTakenParallel);

  // Print distances from vertex 0 (from ompBFS)
  // printf("Distances from vertex 0:\n");
  // for (int i = 0; i < numVertices; i++)
  // {
  //   printf("Vertex %d: %d\n", i, distance[i]);
  // }

  // Free allocated resources
  free(distance);
  free(visited);
  freeGraph(myGraph);

  return 0;
}
