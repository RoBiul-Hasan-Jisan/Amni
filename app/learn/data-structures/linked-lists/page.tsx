"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { LinkedListVisualizer } from "@/components/visualizations/linkedlist-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle2, Lightbulb, ArrowRight } from "lucide-react";

const pythonCode = `class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_head(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_tail(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def delete_node(self, key):
        current = self.head
        if current and current.data == key:
            self.head = current.next
            return
        prev = None
        while current and current.data != key:
            prev = current
            current = current.next
        if current:
            prev.next = current.next
    
    def search(self, key):
        current = self.head
        position = 0
        while current:
            if current.data == key:
                return position
            current = current.next
            position += 1
        return -1
    
    def reverse(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        print(" -> ".join(map(str, elements)) + " -> NULL")

# Usage
ll = LinkedList()
ll.insert_at_head(10)
ll.insert_at_tail(20)
ll.insert_at_tail(30)
ll.display()  # 10 -> 20 -> 30 -> NULL`;

const javascriptCode = `class Node {
  constructor(data) {
    this.data = data;
    this.next = null;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
    this.size = 0;
  }

  insertAtHead(data) {
    const newNode = new Node(data);
    newNode.next = this.head;
    this.head = newNode;
    this.size++;
  }

  insertAtTail(data) {
    const newNode = new Node(data);
    if (!this.head) {
      this.head = newNode;
    } else {
      let current = this.head;
      while (current.next) {
        current = current.next;
      }
      current.next = newNode;
    }
    this.size++;
  }

  insertAt(data, index) {
    if (index < 0 || index > this.size) return false;
    if (index === 0) {
      this.insertAtHead(data);
      return true;
    }
    const newNode = new Node(data);
    let current = this.head;
    for (let i = 0; i < index - 1; i++) {
      current = current.next;
    }
    newNode.next = current.next;
    current.next = newNode;
    this.size++;
    return true;
  }

  deleteNode(key) {
    let current = this.head;
    if (current && current.data === key) {
      this.head = current.next;
      this.size--;
      return true;
    }
    let prev = null;
    while (current && current.data !== key) {
      prev = current;
      current = current.next;
    }
    if (current) {
      prev.next = current.next;
      this.size--;
      return true;
    }
    return false;
  }

  search(key) {
    let current = this.head;
    let index = 0;
    while (current) {
      if (current.data === key) return index;
      current = current.next;
      index++;
    }
    return -1;
  }

  reverse() {
    let prev = null;
    let current = this.head;
    while (current) {
      const nextNode = current.next;
      current.next = prev;
      prev = current;
      current = nextNode;
    }
    this.head = prev;
  }

  display() {
    const elements = [];
    let current = this.head;
    while (current) {
      elements.push(current.data);
      current = current.next;
    }
    console.log(elements.join(' -> ') + ' -> NULL');
  }
}

// Usage
const ll = new LinkedList();
ll.insertAtHead(10);
ll.insertAtTail(20);
ll.insertAtTail(30);
ll.display(); // 10 -> 20 -> 30 -> NULL`;

const javaCode = `public class LinkedList {
    private Node head;
    private int size;

    private class Node {
        int data;
        Node next;
        
        Node(int data) {
            this.data = data;
            this.next = null;
        }
    }

    public void insertAtHead(int data) {
        Node newNode = new Node(data);
        newNode.next = head;
        head = newNode;
        size++;
    }

    public void insertAtTail(int data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
        } else {
            Node current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
        size++;
    }

    public boolean delete(int key) {
        Node current = head;
        if (current != null && current.data == key) {
            head = current.next;
            size--;
            return true;
        }
        Node prev = null;
        while (current != null && current.data != key) {
            prev = current;
            current = current.next;
        }
        if (current != null) {
            prev.next = current.next;
            size--;
            return true;
        }
        return false;
    }

    public int search(int key) {
        Node current = head;
        int index = 0;
        while (current != null) {
            if (current.data == key) return index;
            current = current.next;
            index++;
        }
        return -1;
    }

    public void reverse() {
        Node prev = null;
        Node current = head;
        while (current != null) {
            Node nextNode = current.next;
            current.next = prev;
            prev = current;
            current = nextNode;
        }
        head = prev;
    }

    public void display() {
        Node current = head;
        while (current != null) {
            System.out.print(current.data + " -> ");
            current = current.next;
        }
        System.out.println("NULL");
    }
}`;

const cppCode = `#include <iostream>
using namespace std;

class Node {
public:
    int data;
    Node* next;
    
    Node(int data) {
        this->data = data;
        this->next = nullptr;
    }
};

class LinkedList {
private:
    Node* head;
    int size;

public:
    LinkedList() : head(nullptr), size(0) {}

    void insertAtHead(int data) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head = newNode;
        size++;
    }

    void insertAtTail(int data) {
        Node* newNode = new Node(data);
        if (head == nullptr) {
            head = newNode;
        } else {
            Node* current = head;
            while (current->next != nullptr) {
                current = current->next;
            }
            current->next = newNode;
        }
        size++;
    }

    bool deleteNode(int key) {
        Node* current = head;
        if (current != nullptr && current->data == key) {
            head = current->next;
            delete current;
            size--;
            return true;
        }
        Node* prev = nullptr;
        while (current != nullptr && current->data != key) {
            prev = current;
            current = current->next;
        }
        if (current != nullptr) {
            prev->next = current->next;
            delete current;
            size--;
            return true;
        }
        return false;
    }

    int search(int key) {
        Node* current = head;
        int index = 0;
        while (current != nullptr) {
            if (current->data == key) return index;
            current = current->next;
            index++;
        }
        return -1;
    }

    void reverse() {
        Node* prev = nullptr;
        Node* current = head;
        while (current != nullptr) {
            Node* nextNode = current->next;
            current->next = prev;
            prev = current;
            current = nextNode;
        }
        head = prev;
    }

    void display() {
        Node* current = head;
        while (current != nullptr) {
            cout << current->data << " -> ";
            current = current->next;
        }
        cout << "NULL" << endl;
    }

    ~LinkedList() {
        while (head != nullptr) {
            Node* temp = head;
            head = head->next;
            delete temp;
        }
    }
};`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is the time complexity of inserting a node at the head of a singly linked list?",
    options: ["O(1)", "O(n)", "O(log n)", "O(n²)"],
    correctAnswer: 0,
    explanation:
      "Inserting at head is O(1) because we only need to create a new node, point it to the current head, and update the head pointer. No traversal is needed.",
  },
  {
    id: 2,
    question: "What is the main advantage of linked lists over arrays?",
    options: [
      "Faster random access",
      "Less memory usage",
      "Dynamic size and efficient insertions/deletions",
      "Better cache locality",
    ],
    correctAnswer: 2,
    explanation:
      "Linked lists excel at dynamic sizing and O(1) insertions/deletions (when you have the reference). Arrays have better cache locality and random access.",
  },
  {
    id: 3,
    question: "In a singly linked list, how do you access the previous node?",
    options: [
      "Using the prev pointer",
      "You cannot directly access it",
      "Using the head pointer",
      "Using array indexing",
    ],
    correctAnswer: 1,
    explanation:
      "In a singly linked list, each node only has a 'next' pointer. To access the previous node, you must traverse from the head, which is why doubly linked lists were invented.",
  },
  {
    id: 4,
    question: "What happens to the old head when you insert a new node at the head?",
    options: [
      "It gets deleted",
      "It becomes the second node",
      "It stays as the head",
      "It moves to the tail",
    ],
    correctAnswer: 1,
    explanation:
      "When inserting at head, the new node's 'next' pointer is set to the current head, making the old head the second node in the list.",
  },
  {
    id: 5,
    question: "What is the time complexity of searching for an element in an unsorted linked list?",
    options: ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
    correctAnswer: 2,
    explanation:
      "In a linked list, you must traverse from the head node by node until you find the element or reach the end. This gives O(n) time complexity in the worst case.",
  },
  {
    id: 6,
    question: "Which data structure is most suitable for implementing a browser's back button functionality?",
    options: ["Array", "Queue", "Singly Linked List", "Stack (can use linked list)"],
    correctAnswer: 3,
    explanation:
      "A stack follows LIFO (Last In, First Out) which matches the back button behavior - the last page visited should be the first one returned to.",
  },
];

export default function LinkedListPage() {
  const result = getSubtopicBySlug("data-structures", "linked-lists");
  
  if (!result) {
    return <div className="p-8 text-center text-muted-foreground">Topic not found</div>;
  }
  
  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      {/* Introduction */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">What is a Linked List?</h2>
        <p className="text-muted-foreground mb-4 leading-relaxed">
          A <strong className="text-foreground">Linked List</strong> is a linear data structure where elements are stored in nodes, 
          and each node points to the next node in the sequence. Unlike arrays, linked list elements are not stored in 
          contiguous memory locations - they are connected through pointers.
        </p>
        <div className="grid md:grid-cols-2 gap-4 mt-6">
          <Card className="border-primary/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Advantages
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>Dynamic size - grows and shrinks as needed</li>
                <li>Efficient insertions/deletions at known positions</li>
                <li>No memory wastage (allocate as needed)</li>
                <li>Easy implementation of stacks and queues</li>
              </ul>
            </CardContent>
          </Card>
          <Card className="border-destructive/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-destructive" />
                Disadvantages
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>No random access - must traverse from head</li>
                <li>Extra memory for storing pointers</li>
                <li>Poor cache locality</li>
                <li>Reverse traversal difficult in singly linked list</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Node Structure */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Node Structure</h2>
        <p className="text-muted-foreground mb-4">
          Each node in a linked list contains two parts:
        </p>
        <div className="flex items-center gap-4 p-6 bg-muted/50 rounded-lg mb-4">
          <div className="flex border-2 border-primary rounded-lg overflow-hidden">
            <div className="px-6 py-4 border-r border-primary bg-primary/10">
              <div className="text-xs text-muted-foreground mb-1">Data</div>
              <div className="font-mono font-bold text-foreground">value</div>
            </div>
            <div className="px-6 py-4 bg-muted/30">
              <div className="text-xs text-muted-foreground mb-1">Next</div>
              <div className="font-mono text-sm text-muted-foreground">pointer</div>
            </div>
          </div>
          <ArrowRight className="h-5 w-5 text-muted-foreground" />
          <div className="text-muted-foreground">Points to next node or NULL</div>
        </div>
      </section>

      {/* Types of Linked Lists */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Types of Linked Lists</h2>
        <div className="grid gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">1. Singly Linked List</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Each node has data and a pointer to the next node. Traversal is only possible in one direction (forward).
              </p>
              <div className="mt-2 font-mono text-xs text-muted-foreground">
                [A|→] → [B|→] → [C|→] → NULL
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">2. Doubly Linked List</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Each node has data, a pointer to the next node, AND a pointer to the previous node. Allows bidirectional traversal.
              </p>
              <div className="mt-2 font-mono text-xs text-muted-foreground">
                NULL ← [←|A|→] ↔ [←|B|→] ↔ [←|C|→] → NULL
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">3. Circular Linked List</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                The last node points back to the first node instead of NULL, forming a circle. Useful for round-robin scheduling.
              </p>
              <div className="mt-2 font-mono text-xs text-muted-foreground">
                [A|→] → [B|→] → [C|→] → (back to A)
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Interactive Visualization */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Interactive Visualization</h2>
        <p className="text-muted-foreground mb-4">
          Experiment with linked list operations. Watch how nodes are connected and how traversal works.
        </p>
        <Card>
          <CardContent className="pt-6">
            <LinkedListVisualizer />
          </CardContent>
        </Card>
      </section>

      {/* Time Complexity */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Time Complexity</h2>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Operation</TableHead>
                <TableHead>Singly Linked List</TableHead>
                <TableHead>Doubly Linked List</TableHead>
                <TableHead>Notes</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">Insert at Head</TableCell>
                <TableCell><Badge variant="secondary">O(1)</Badge></TableCell>
                <TableCell><Badge variant="secondary">O(1)</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">Direct pointer update</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Insert at Tail</TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell><Badge variant="secondary">O(1)*</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">*With tail pointer</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Insert at Position</TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">Need to traverse</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Delete Head</TableCell>
                <TableCell><Badge variant="secondary">O(1)</Badge></TableCell>
                <TableCell><Badge variant="secondary">O(1)</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">Direct pointer update</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Delete Tail</TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell><Badge variant="secondary">O(1)*</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">*With tail pointer</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Search</TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">Linear search required</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Access by Index</TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell><Badge variant="outline">O(n)</Badge></TableCell>
                <TableCell className="text-muted-foreground text-sm">No random access</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </section>

      {/* Code Examples */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Implementation</h2>
        <p className="text-muted-foreground mb-4">
          Complete linked list implementation with all basic operations:
        </p>
        <MultiLanguageCode
          codes={[
            { language: "python", label: "Python", code: pythonCode },
            { language: "javascript", label: "JavaScript", code: javascriptCode },
            { language: "java", label: "Java", code: javaCode },
            { language: "cpp", label: "C++", code: cppCode },
          ]}
        />
      </section>

      {/* Common Operations Explained */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Key Operations Explained</h2>
        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Reversing a Linked List</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-3">
                One of the most common interview questions. Use three pointers: prev, current, and next.
              </p>
              <ol className="text-sm text-muted-foreground space-y-1 list-decimal list-inside">
                <li>Initialize prev = NULL, current = head</li>
                <li>Store next = current.next</li>
                <li>Reverse the link: current.next = prev</li>
                <li>Move prev and current one step forward</li>
                <li>Repeat until current is NULL</li>
                <li>Set head = prev</li>
              </ol>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Detecting a Cycle (Floyd's Algorithm)</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-3">
                Use two pointers: slow (moves 1 step) and fast (moves 2 steps). If they meet, there's a cycle.
              </p>
              <CodeBlock
                language="python"
                code={`def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`}
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Finding Middle Element</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-3">
                Use slow and fast pointers. When fast reaches end, slow is at middle.
              </p>
              <CodeBlock
                language="python"
                code={`def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # Middle node`}
              />
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Arrays vs Linked Lists */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Arrays vs Linked Lists</h2>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Aspect</TableHead>
                <TableHead>Array</TableHead>
                <TableHead>Linked List</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="font-medium">Memory</TableCell>
                <TableCell className="text-muted-foreground">Contiguous</TableCell>
                <TableCell className="text-muted-foreground">Non-contiguous</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Size</TableCell>
                <TableCell className="text-muted-foreground">Fixed (static)</TableCell>
                <TableCell className="text-muted-foreground">Dynamic</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Access</TableCell>
                <TableCell className="text-muted-foreground">O(1) random access</TableCell>
                <TableCell className="text-muted-foreground">O(n) sequential</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Insert/Delete</TableCell>
                <TableCell className="text-muted-foreground">O(n) - shifting needed</TableCell>
                <TableCell className="text-muted-foreground">O(1) if position known</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Cache Performance</TableCell>
                <TableCell className="text-muted-foreground">Excellent (locality)</TableCell>
                <TableCell className="text-muted-foreground">Poor (scattered)</TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="font-medium">Extra Space</TableCell>
                <TableCell className="text-muted-foreground">None</TableCell>
                <TableCell className="text-muted-foreground">Pointer overhead</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </section>

      {/* Interview Tips */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Interview Patterns</h2>
        <div className="grid gap-4">
          <Card className="border-primary/20 bg-primary/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-primary" />
                Two Pointer Technique
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Many linked list problems use slow/fast pointers: cycle detection, finding middle, 
                nth node from end, palindrome check.
              </p>
            </CardContent>
          </Card>
          <Card className="border-primary/20 bg-primary/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-primary" />
                Dummy Node Pattern
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Create a dummy node before head to simplify edge cases. Useful when the head might change 
                (merge lists, remove elements, partition).
              </p>
            </CardContent>
          </Card>
          <Card className="border-primary/20 bg-primary/5">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-primary" />
                Common Interview Questions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Reverse a linked list (iterative and recursive)</li>
                <li>• Detect and remove cycle</li>
                <li>• Merge two sorted lists</li>
                <li>• Find intersection of two lists</li>
                <li>• Remove nth node from end</li>
                <li>• Check if palindrome</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Common Mistakes */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Common Mistakes</h2>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-4 bg-destructive/10 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
            <div>
              <p className="font-medium text-foreground">Losing reference to head</p>
              <p className="text-sm text-muted-foreground">
                Always save the head before traversing. Use a temporary pointer for iteration.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-destructive/10 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
            <div>
              <p className="font-medium text-foreground">Not handling NULL/empty list</p>
              <p className="text-sm text-muted-foreground">
                Always check if head is NULL before accessing head.next or head.data.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-destructive/10 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
            <div>
              <p className="font-medium text-foreground">Memory leaks (C/C++)</p>
              <p className="text-sm text-muted-foreground">
                Remember to free/delete nodes when removing them. Use smart pointers in modern C++.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-destructive/10 rounded-lg">
            <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
            <div>
              <p className="font-medium text-foreground">Off-by-one errors in traversal</p>
              <p className="text-sm text-muted-foreground">
                {"Be careful with while(current) vs while(current.next) - know when to stop."}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Quiz */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
        <Quiz questions={quizQuestions} title="Linked List Quiz" />
      </section>
    </TopicContent>
  );
}
