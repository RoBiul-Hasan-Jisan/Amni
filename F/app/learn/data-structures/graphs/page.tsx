import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { GraphVisualizer } from "@/components/visualizations/graph-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Network } from "lucide-react";

export default function GraphsPage() {
  const result = getSubtopicBySlug("data-structures", "graphs");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python",
      label: "Python",
      code: `# Adjacency List Implementation
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, directed=False):
        self.graph[u].append(v)
        if not directed:
            self.graph[v].append(u)
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            print(vertex, end=" ")
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def dfs(self, start):
        visited = set()
        stack = [start]
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                print(vertex, end=" ")
                visited.add(vertex)
                stack.extend(self.graph[vertex])

# Usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)

print("BFS starting from vertex 0:")
g.bfs(0)  # Output: 0 1 2 3`,
    },
    {
      language: "javascript",
      label: "JavaScript",
      code: `// Adjacency List Implementation
class Graph {
  constructor() {
    this.adjacencyList = new Map();
  }

  addVertex(vertex) {
    if (!this.adjacencyList.has(vertex)) {
      this.adjacencyList.set(vertex, []);
    }
  }

  addEdge(vertex1, vertex2, directed = false) {
    if (!this.adjacencyList.has(vertex1)) this.addVertex(vertex1);
    if (!this.adjacencyList.has(vertex2)) this.addVertex(vertex2);
    
    this.adjacencyList.get(vertex1).push(vertex2);
    if (!directed) {
      this.adjacencyList.get(vertex2).push(vertex1);
    }
  }

  bfs(start) {
    const visited = new Set();
    const queue = [start];
    visited.add(start);
    const result = [];

    while (queue.length) {
      const vertex = queue.shift();
      result.push(vertex);

      for (const neighbor of this.adjacencyList.get(vertex)) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
    return result;
  }

  dfs(start) {
    const visited = new Set();
    const result = [];

    const dfsHelper = (vertex) => {
      visited.add(vertex);
      result.push(vertex);

      for (const neighbor of this.adjacencyList.get(vertex)) {
        if (!visited.has(neighbor)) {
          dfsHelper(neighbor);
        }
      }
    };

    dfsHelper(start);
    return result;
  }
}

// Usage
const graph = new Graph();
graph.addEdge('A', 'B');
graph.addEdge('A', 'C');
graph.addEdge('B', 'D');
graph.addEdge('C', 'E');

console.log("BFS:", graph.bfs('A')); // ['A', 'B', 'C', 'D', 'E']`,
    },
    {
      language: "java",
      label: "Java",
      code: `import java.util.*;

// Adjacency List Implementation
class Graph {
    private Map<Integer, List<Integer>> adjacencyList;
    private boolean directed;

    public Graph(boolean directed) {
        this.adjacencyList = new HashMap<>();
        this.directed = directed;
    }

    public void addVertex(int vertex) {
        adjacencyList.putIfAbsent(vertex, new ArrayList<>());
    }

    public void addEdge(int source, int destination) {
        addVertex(source);
        addVertex(destination);
        
        adjacencyList.get(source).add(destination);
        if (!directed) {
            adjacencyList.get(destination).add(source);
        }
    }

    public List<Integer> bfs(int start) {
        List<Integer> result = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();

        visited.add(start);
        queue.offer(start);

        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            result.add(vertex);

            for (int neighbor : adjacencyList.get(vertex)) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        return result;
    }

    public List<Integer> dfs(int start) {
        List<Integer> result = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        Stack<Integer> stack = new Stack<>();

        stack.push(start);

        while (!stack.isEmpty()) {
            int vertex = stack.pop();
            if (!visited.contains(vertex)) {
                visited.add(vertex);
                result.add(vertex);
                
                for (int neighbor : adjacencyList.get(vertex)) {
                    if (!visited.contains(neighbor)) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        return result;
    }
}

// Usage
Graph graph = new Graph(false);
graph.addEdge(0, 1);
graph.addEdge(0, 2);
graph.addEdge(1, 3);
graph.addEdge(2, 4);

System.out.println("BFS: " + graph.bfs(0));`,
    },
    {
      language: "cpp",
      label: "C++",
      code: `#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
using namespace std;

// Adjacency List Implementation
class Graph {
private:
    unordered_map<int, vector<int>> adjacencyList;
    bool directed;

public:
    Graph(bool isDirected = false) : directed(isDirected) {}

    void addEdge(int u, int v) {
        adjacencyList[u].push_back(v);
        if (!directed) {
            adjacencyList[v].push_back(u);
        }
    }

    vector<int> bfs(int start) {
        vector<int> result;
        unordered_set<int> visited;
        queue<int> q;

        visited.insert(start);
        q.push(start);

        while (!q.empty()) {
            int vertex = q.front();
            q.pop();
            result.push_back(vertex);

            for (int neighbor : adjacencyList[vertex]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        return result;
    }

    vector<int> dfs(int start) {
        vector<int> result;
        unordered_set<int> visited;
        stack<int> s;

        s.push(start);

        while (!s.empty()) {
            int vertex = s.top();
            s.pop();

            if (visited.find(vertex) == visited.end()) {
                visited.insert(vertex);
                result.push_back(vertex);

                for (int neighbor : adjacencyList[vertex]) {
                    if (visited.find(neighbor) == visited.end()) {
                        s.push(neighbor);
                    }
                }
            }
        }
        return result;
    }
};

// Usage
int main() {
    Graph graph(false);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 4);

    vector<int> bfsResult = graph.bfs(0);
    cout << "BFS: ";
    for (int node : bfsResult) cout << node << " ";
    
    return 0;
}`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the main difference between BFS and DFS?",
      options: [
        "BFS uses stack, DFS uses queue",
        "BFS explores level by level, DFS goes deep first",
        "BFS is for directed graphs, DFS for undirected",
        "BFS has better space complexity"
      ],
      correctAnswer: 1,
      explanation: "BFS explores all neighbors at the present depth before moving to the next level, while DFS explores as far as possible along each branch before backtracking.",
    },
    {
      id: 2,
      question: "What is the time complexity of BFS/DFS on an adjacency list representation?",
      options: ["O(V)", "O(V+E)", "O(V*E)", "O(E)"],
      correctAnswer: 1,
      explanation: "BFS/DFS visits each vertex once (O(V)) and explores all edges (O(E)), resulting in O(V+E) time complexity.",
    },
    {
      id: 3,
      question: "Which graph representation is most space-efficient for sparse graphs?",
      options: ["Adjacency Matrix", "Adjacency List", "Incidence Matrix", "Edge List"],
      correctAnswer: 1,
      explanation: "Adjacency List uses O(V+E) space, making it efficient for sparse graphs (graphs with few edges compared to vertices).",
    },
    {
      id: 4,
      question: "What is a cycle in a graph?",
      options: [
        "A path where start and end vertices are the same",
        "A graph with no edges",
        "A linear arrangement of vertices",
        "A vertex with degree 0"
      ],
      correctAnswer: 0,
      explanation: "A cycle is a path where the first and last vertices are the same, with no repeated vertices in between.",
    },
    {
      id: 5,
      question: "What is the degree of a vertex?",
      options: [
        "Number of edges incident to it",
        "Height of the vertex in the tree",
        "Distance from the root",
        "Number of cycles it participates in"
      ],
      correctAnswer: 0,
      explanation: "Degree of a vertex is the number of edges connected to it. In directed graphs, we have in-degree and out-degree.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is a Graph?</h2>
          <p className="text-muted-foreground mb-4">
            A <strong className="text-foreground">graph</strong> is a non-linear data structure consisting of nodes (vertices) 
            and edges connecting these nodes. Think of it as a network of interconnected points, 
            like social networks, road maps, or web pages linked together.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Imagine a social network like Facebook. Each person is a vertex, and friendships 
                  are edges connecting them. You can navigate from one friend to another through 
                  mutual connections - that's graph traversal in action!
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Characteristics */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Key Characteristics</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Directed vs Undirected</h4>
              <p className="text-sm text-muted-foreground">
                Directed graphs have edges with direction (like one-way streets), while 
                undirected graphs have bidirectional edges (like two-way roads).
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Weighted vs Unweighted</h4>
              <p className="text-sm text-muted-foreground">
                Edges can have weights representing cost, distance, or capacity. 
                Unweighted graphs treat all edges equally.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Cyclic vs Acyclic</h4>
              <p className="text-sm text-muted-foreground">
                Cyclic graphs contain cycles (paths that start and end at same vertex), 
                while acyclic graphs have no cycles (like trees).
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Connected vs Disconnected</h4>
              <p className="text-sm text-muted-foreground">
                Connected graphs have paths between all vertex pairs, while disconnected 
                graphs have isolated components.
              </p>
            </div>
          </div>
        </section>

        {/* Graph Representations */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Graph Representations</h2>
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Adjacency Matrix</h4>
              <p className="text-sm text-muted-foreground">
                2D array where matrix[i][j] = 1 if edge exists between i and j.
                <br />
                <strong>Space: O(V²)</strong>
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Adjacency List</h4>
              <p className="text-sm text-muted-foreground">
                Array of lists where each vertex has list of its neighbors.
                <br />
                <strong>Space: O(V+E)</strong>
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Edge List</h4>
              <p className="text-sm text-muted-foreground">
                Simple list of edges (u, v, weight).
                <br />
                <strong>Space: O(E)</strong>
              </p>
            </div>
          </div>
        </section>

        {/* Interactive Visualization */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interactive Graph Visualization</h2>
          <p className="text-muted-foreground mb-4">
            Create vertices and edges, then visualize BFS, DFS, and other algorithms.
            Experiment with directed/undirected and weighted/unweighted graphs.
          </p>
          <GraphVisualizer />
        </section>

        {/* Graph Algorithms */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Graph Algorithms</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Algorithm</th>
                  <th className="border border-border p-3 text-left text-foreground">Time Complexity</th>
                  <th className="border border-border p-3 text-left text-foreground">Purpose</th>
                  <th className="border border-border p-3 text-left text-foreground">Use Case</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-border p-3 text-foreground font-medium">BFS</td>
                  <td className="border border-border p-3 font-mono text-primary">O(V+E)</td>
                  <td className="border border-border p-3 text-muted-foreground">Level-order traversal, shortest path (unweighted)</td>
                  <td className="border border-border p-3 text-muted-foreground">Social network degrees, web crawling</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground font-medium">DFS</td>
                  <td className="border border-border p-3 font-mono text-primary">O(V+E)</td>
                  <td className="border border-border p-3 text-muted-foreground">Path finding, cycle detection, topological sort</td>
                  <td className="border border-border p-3 text-muted-foreground">Maze solving, dependency resolution</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground font-medium">Dijkstra</td>
                  <td className="border border-border p-3 font-mono text-warning">O(E log V)</td>
                  <td className="border border-border p-3 text-muted-foreground">Shortest path (weighted, no negative edges)</td>
                  <td className="border border-border p-3 text-muted-foreground">GPS navigation, network routing</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground font-medium">Kruskal</td>
                  <td className="border border-border p-3 font-mono text-warning">O(E log V)</td>
                  <td className="border border-border p-3 text-muted-foreground">Minimum Spanning Tree (MST)</td>
                  <td className="border border-border p-3 text-muted-foreground">Network design, clustering</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground font-medium">Prim</td>
                  <td className="border border-border p-3 font-mono text-warning">O(E log V)</td>
                  <td className="border border-border p-3 text-muted-foreground">Minimum Spanning Tree (MST)</td>
                  <td className="border border-border p-3 text-muted-foreground">Network cabling, approximation algorithms</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground font-medium">Topological Sort</td>
                  <td className="border border-border p-3 font-mono text-primary">O(V+E)</td>
                  <td className="border border-border p-3 text-muted-foreground">Linear ordering of vertices in DAG</td>
                  <td className="border border-border p-3 text-muted-foreground">Task scheduling, build systems</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            <Clock className="inline h-4 w-4 mr-1" />
            Space Complexity: Typically O(V) for most graph algorithms
          </p>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Code Examples</h2>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Mistakes to Avoid</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Forgetting to Track Visited Nodes</h4>
                <p className="text-sm text-muted-foreground">
                  In graph traversal (BFS/DFS), always maintain a visited set to avoid infinite loops 
                  and reprocessing nodes.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Choosing Wrong Data Structure</h4>
                <p className="text-sm text-muted-foreground">
                  Using adjacency matrix for sparse graphs wastes space. Use adjacency lists for 
                  graphs where E &lt;&lt; V².
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Assuming Graph is Connected</h4>
                <p className="text-sm text-muted-foreground">
                  Always consider disconnected graphs. You may need to run BFS/DFS from multiple 
                  starting points to cover all components.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Using DFS for Shortest Path</h4>
                <p className="text-sm text-muted-foreground">
                  DFS doesn't guarantee shortest path in unweighted graphs. Use BFS for unweighted 
                  graphs and Dijkstra for weighted graphs.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interview Patterns</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <Network className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Graph Traversal</h4>
                <p className="text-sm text-muted-foreground">
                  BFS for level-order problems, shortest path (unweighted). 
                  DFS for path existence, cycle detection, backtracking problems.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <Network className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Union-Find (Disjoint Set)</h4>
                <p className="text-sm text-muted-foreground">
                  Detect cycles in undirected graphs, connected components, Kruskal's MST algorithm.
                  Great for dynamic connectivity problems.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <Network className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Topological Sort</h4>
                <p className="text-sm text-muted-foreground">
                  Course scheduling, build system dependencies, task ordering. 
                  Works only on Directed Acyclic Graphs (DAGs).
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <Network className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Shortest Path Algorithms</h4>
                <p className="text-sm text-muted-foreground">
                  BFS for unweighted, Dijkstra for non-negative weights, 
                  Bellman-Ford for negative weights, A* for heuristic search.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Real-world Applications */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Real-world Applications</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Social Networks</h4>
              <p className="text-sm text-muted-foreground">
                Friend suggestions (BFS degrees), community detection, 
                influence propagation, shortest connection path.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Navigation Systems</h4>
              <p className="text-sm text-muted-foreground">
                GPS routing (Dijkstra, A*), traffic optimization, 
                public transportation planning, delivery logistics.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Web Crawling</h4>
              <p className="text-sm text-muted-foreground">
                BFS for systematic crawling, PageRank algorithm, 
                link analysis, sitemap generation.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Network Design</h4>
              <p className="text-sm text-muted-foreground">
                Minimum spanning trees (Kruskal, Prim) for network cables, 
                internet backbone design, circuit layout.
              </p>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Graphs Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}