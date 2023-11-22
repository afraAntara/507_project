#ifndef GRAPH_H
#define GRAPH_H

#include <stdlib.h>
#include <stdio.h>

// Graph Density Types
typedef enum
{
  Sparse,
  Medium,
  Dense
} GraphDensity;

// Graph Struct
typedef struct
{
  int numVertices;
  int numEdges;
  int *adjacencyList; // Dynamically allocated array for adjacency list
  int *edgesOffset;   // Dynamically allocated array for edges offset
  int *edgesSize;     // Dynamically allocated array for edges size
} Graph;

// Function Prototypes
Graph *initGraph(int numVertices, GraphDensity density);
void generateEdges(Graph *graph, GraphDensity density);
void freeGraph(Graph *graph);

#endif // GRAPH_H
