// A C++ program to implement greedy algorithm for graph coloring
#include <iostream>
#include <fstream>
#include <list>
using namespace std;

// A class that represents an undirected graph
class Graph
{
   int V;    // No. of vertices
   list<int> *adj;    // A dynamic array of adjacency lists
public:
   // Constructor and destructor
   Graph(int V)   { this->V = V; adj = new list<int>[V]; }
   ~Graph()       { delete [] adj; }

   // function to add an edge to graph
   void addEdge(int v, int w);

   // Prints greedy coloring of the vertices
   void greedyColoring();
};
                                         
void Graph::addEdge(int v, int w)
{
   adj[v].push_back(w);
   adj[w].push_back(v);  // Note: the graph is undirected
}
                                                  
// Assigns colors (starting from 0) to all vertices and prints
// the assignment of colors
void Graph::greedyColoring()
{
   int result[V];

   // Assign the first color to first vertex
   result[0]  = 0;
  
   // Initialize remaining V-1 vertices as unassigned
   for (int u = 1; u < V; u++)
      result[u] = -1;  // no color is assigned to u
   
   // A temporary array to store the available colors. True
   // value of available[cr] would mean that the color cr is
   // assigned to one of its adjacent vertices
   bool available[V];
   for (int cr = 0; cr < V; cr++)
      available[cr] = false;
   
   // Assign colors to remaining V-1 vertices
   for (int u = 1; u < V; u++)
   {
   // Process all adjacent vertices and flag their colors
   // as unavailable
   list<int>::iterator i;
   for (i = adj[u].begin(); i != adj[u].end(); ++i)
      if (result[*i] != -1)
         available[result[*i]] = true;
   
   // Find the first available color
   int cr;
   for (cr = 0; cr < V; cr++)
      if (available[cr] == false)
      break;
   
   result[u] = cr; // Assign the found color
   
   // Reset the values back to false for the next iteration
   for (i = adj[u].begin(); i != adj[u].end(); ++i)
      if (result[*i] != -1)
         available[result[*i]] = false;
}

// print the result
for (int u = 0; u < V; u++)
   cout <<  u << " " << result[u] << endl;
}

// Driver program to test above function
int main(int argc, char *argv[])
{
   ifstream f;
   int u,v;
   int count;

   f.open(argv[1]);
   if (!f) {
      cout << "ERROR: Could not open '" << argv[1] << "'." << endl;
      return 1;
   }
   // cout << "Input network: " << argv[1] << endl;

   // read number of nodes
   f >> u;
   Graph G(u);
   // cout << "Number of nodes: " << u << endl;

   // read number of edges
   count=0;
   while (f>>u>>v) {
      G.addEdge(u,v);
      count++;
   }
   // cout << "Number of edges:" << count << endl;
   f.close();
   G.greedyColoring();
   
   return 0;
}
