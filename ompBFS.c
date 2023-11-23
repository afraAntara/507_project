#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "graph.h"

void ompBFS(int start, Graph *G, int *distance)
{
  int numThreads;
#pragma omp parallel
  {
#pragma omp single
    numThreads = omp_get_num_threads();
  }

  int **localNextFrontiers = (int **)malloc(numThreads * sizeof(int *));
  int *localNextFrontierSizes = (int *)calloc(numThreads, sizeof(int));
  for (int i = 0; i < numThreads; i++)
  {
    localNextFrontiers[i] = (int *)malloc(G->numVertices * sizeof(int));
  }

  int *currentFrontier = (int *)malloc(G->numVertices * sizeof(int));
  int currentFrontierSize = 1;
  currentFrontier[0] = start;
  distance[start] = 0;

  while (currentFrontierSize > 0)
  {
#pragma omp parallel
    {
      // Each thread processes a chunk of the current frontier
      int threadNum = omp_get_thread_num();
      int chunkSize = currentFrontierSize / numThreads;
      int start = threadNum * chunkSize;
      int end = (threadNum == numThreads - 1) ? currentFrontierSize : start + chunkSize; // last thread takes any remainder

      int *localNextFrontier = localNextFrontiers[threadNum];
      int localNextFrontierSize = 0;

      for (int i = start; i < end; i++)
      {
        int current = currentFrontier[i];
        for (int j = G->edgesOffset[current]; j < G->edgesOffset[current] + G->edgesSize[current]; ++j)
        {
          int v = G->adjacencyList[j];
          {
            if (distance[v] == INT_MAX)
            {
              distance[v] = distance[current] + 1;
              // #pragma omp critical
              localNextFrontier[localNextFrontierSize++] = v;
            }
          }
        }
      }

      localNextFrontierSizes[threadNum] = localNextFrontierSize;
    }

    // Merging local buffers into the global next frontier
    int totalSize = 0;
    for (int i = 0; i < numThreads; i++)
    {
      totalSize += localNextFrontierSizes[i];
    }

    int *nextFrontier = (int *)malloc(totalSize * sizeof(int));
    int nextFrontierSize = 0;

    for (int i = 0; i < numThreads; i++)
    {
      for (int j = 0; j < localNextFrontierSizes[i]; j++)
      {
        nextFrontier[nextFrontierSize++] = localNextFrontiers[i][j];
      }
      localNextFrontierSizes[i] = 0; // Reset for next iteration
    }

    free(currentFrontier);
    currentFrontier = nextFrontier;
    currentFrontierSize = nextFrontierSize;
  }

  free(currentFrontier);
  for (int i = 0; i < numThreads; i++)
  {
    free(localNextFrontiers[i]);
  }
  free(localNextFrontiers);
  free(localNextFrontierSizes);
}
