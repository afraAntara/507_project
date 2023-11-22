#include "graph.h"
#include <time.h>

// Function to initialize a graph
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
