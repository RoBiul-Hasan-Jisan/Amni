import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { ArrayVisualizer } from "@/components/visualizations/array-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, ChevronRight } from "lucide-react";
import Link from "next/link"; 
export default function ArraysPage() {
  const result = getSubtopicBySlug("data-structures", "arrays");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python",
      label: "Python",
      code: `# Creating an array (list in Python)
numbers = [10, 20, 30, 40, 50]

# Accessing elements by index
first = numbers[0]    # 10
last = numbers[-1]    # 50

# Inserting an element
numbers.append(60)    # Add to end
numbers.insert(2, 25) # Insert at index 2

# Removing elements
numbers.pop()         # Remove last element
numbers.remove(30)    # Remove first occurrence of 30

# Iterating through array
for num in numbers:
    print(num)`,
    },
    {
      language: "javascript",
      label: "JavaScript",
      code: `// Creating an array
const numbers = [10, 20, 30, 40, 50];

// Accessing elements by index
const first = numbers[0];    // 10
const last = numbers[numbers.length - 1]; // 50

// Inserting an element
numbers.push(60);           // Add to end
numbers.splice(2, 0, 25);   // Insert at index 2

// Removing elements
numbers.pop();              // Remove last element
const index = numbers.indexOf(30);
numbers.splice(index, 1);   // Remove 30

// Iterating through array
numbers.forEach(num => console.log(num));`,
    },
    {
      language: "java",
      label: "Java",
      code: `import java.util.ArrayList;

// Using ArrayList (dynamic array)
ArrayList<Integer> numbers = new ArrayList<>();
numbers.add(10);
numbers.add(20);
numbers.add(30);

// Accessing elements
int first = numbers.get(0);    // 10
int last = numbers.get(numbers.size() - 1);

// Inserting at specific index
numbers.add(2, 25);

// Removing elements
numbers.remove(numbers.size() - 1); // Remove last
numbers.remove(Integer.valueOf(30)); // Remove 30

// Iterating
for (int num : numbers) {
    System.out.println(num);
}`,
    },
    {
      language: "cpp",
      label: "C++",
      code: `#include <vector>
#include <iostream>

// Using vector (dynamic array)
std::vector<int> numbers = {10, 20, 30, 40, 50};

// Accessing elements
int first = numbers[0];      // 10
int last = numbers.back();   // 50

// Inserting elements
numbers.push_back(60);       // Add to end
numbers.insert(numbers.begin() + 2, 25);

// Removing elements
numbers.pop_back();          // Remove last
numbers.erase(numbers.begin() + 2);

// Iterating
for (int num : numbers) {
    std::cout << num << std::endl;
}`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the time complexity of accessing an element by index in an array?",
      options: ["O(1)", "O(n)", "O(log n)", "O(n^2)"],
      correctAnswer: 0,
      explanation: "Arrays provide constant time O(1) access because elements are stored in contiguous memory locations, allowing direct calculation of any element's address.",
    },
    {
      id: 2,
      question: "What happens when you insert an element at the beginning of an array?",
      options: [
        "Only the first element is affected",
        "All elements need to shift right",
        "The array size remains the same",
        "It takes O(1) time"
      ],
      correctAnswer: 1,
      explanation: "Inserting at the beginning requires shifting all existing elements one position to the right to make room, resulting in O(n) time complexity.",
    },
    {
      id: 3,
      question: "Which data structure would be more efficient for frequent insertions at random positions?",
      options: ["Array", "Linked List", "Both are equal", "Neither"],
      correctAnswer: 1,
      explanation: "Linked Lists allow O(1) insertion once you have a reference to the position, while arrays require O(n) time to shift elements.",
    },
    {
      id: 4,
      question: "What is the index of the first element in most programming languages?",
      options: ["1", "0", "-1", "Depends on the language"],
      correctAnswer: 1,
      explanation: "Most modern programming languages use 0-based indexing, meaning the first element is at index 0.",
    },
  ];

const arrayProblems = [
  {
    id: 1,
    slug: "two-sum",
    title: "Two Sum",
    description: "Find two numbers in an array that add up to a target sum.",
    difficulty: "Easy",
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 2,
    slug: "maximum-subarray",
    title: "Maximum Subarray",
    description: "Find the contiguous subarray with the largest sum.",
    difficulty: "Medium",
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 3,
    slug: "contains-duplicate",
    title: "Contains Duplicate",
    description: "Check if any value appears at least twice in the array.",
    difficulty: "Easy",
    timeComplexity: "O(n)",
    spaceComplexity: "O(n)",
  },
  {
    id: 4,
    slug: "product-of-array-except-self",
    title: "Product of Array Except Self",
    description: "Return an array where each element is the product of all other elements.",
    difficulty: "Medium",
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 5,
    slug: "best-time-to-buy-sell-stock",
    title: "Best Time to Buy and Sell Stock",
    description: "Find the maximum profit from buying and selling stock once.",
    difficulty: "Easy",
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 6,
    slug: "3sum",
    title: "3Sum",
    description: "Find all triplets that sum to zero.",
    difficulty: "Medium",
    timeComplexity: "O(n²)",
    spaceComplexity: "O(n)",
  },
  {
    id: 7,
    slug: "maximum-product-subarray",
    title: "Maximum Product Subarray",
    description: "Find the contiguous subarray with the largest product.",
    difficulty: "Medium",
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 8,
    slug: "find-minimum-in-rotated-sorted-array",
    title: "Find Minimum in Rotated Sorted Array",
    description: "Find the minimum element in a rotated sorted array.",
    difficulty: "Medium",
    timeComplexity: "O(log n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 9,
    slug: "container-with-most-water",
    title: "Container With Most Water",
    description: "Find two lines that together with the x-axis form a container with the most water.",
    difficulty: "Medium",
    timeComplexity: "O(n)",
    spaceComplexity: "O(1)",
  },
  {
    id: 10,
    slug: "search-in-rotated-sorted-array",
    title: "Search in Rotated Sorted Array",
    description: "Search for a target value in a rotated sorted array.",
    difficulty: "Medium",
    timeComplexity: "O(log n)",
    spaceComplexity: "O(1)",
  },
];
  return (
    <TopicContent topic={topic} subtopic={subtopic}>
     <div className="space-y-8" suppressHydrationWarning>
     {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is an Array?</h2>
          <p className="text-muted-foreground mb-4">
            An <strong className="text-foreground">array</strong> is a fundamental data structure that stores elements 
            in contiguous memory locations. Think of it like a row of boxes, where each box 
            can hold one item and has a number (index) so you can find it quickly.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Imagine a parking lot where each spot has a number. You can go directly to spot #5 
                  without checking spots 1-4. That is how arrays work - you can jump to any position instantly!
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
              <h4 className="font-semibold mb-2 text-foreground">Fixed vs Dynamic Size</h4>
              <p className="text-sm text-muted-foreground">
                Traditional arrays have fixed size (C, Java), while dynamic arrays 
                (Python lists, JavaScript arrays) can grow automatically.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Same Data Type</h4>
              <p className="text-sm text-muted-foreground">
                In typed languages, arrays typically hold elements of the same type. 
                Dynamic languages like Python allow mixed types.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Zero-Indexed</h4>
              <p className="text-sm text-muted-foreground">
                Most languages start counting from 0, so the first element is at index 0, 
                the second at index 1, and so on.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Contiguous Memory</h4>
              <p className="text-sm text-muted-foreground">
                Elements are stored next to each other in memory, enabling fast access 
                and efficient iteration.
              </p>
            </div>
          </div>
        </section>

        {/* Interactive Visualization */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interactive Array Visualization</h2>
          <p className="text-muted-foreground mb-4">
            Experiment with array operations below. Insert, delete, and sort elements 
            to see how the array changes.
          </p>
          <ArrayVisualizer />
        </section>

        {/* Time Complexity */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Time & Space Complexity</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Operation</th>
                  <th className="border border-border p-3 text-left text-foreground">Time Complexity</th>
                  <th className="border border-border p-3 text-left text-foreground">Explanation</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-border p-3 text-foreground">Access by Index</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 text-muted-foreground">Direct calculation of memory address</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">Search (unsorted)</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 text-muted-foreground">May need to check every element</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">Insert at End</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)*</td>
                  <td className="border border-border p-3 text-muted-foreground">Amortized for dynamic arrays</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">Insert at Beginning</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 text-muted-foreground">All elements must shift right</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">Delete from End</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 text-muted-foreground">Just decrease the size</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">Delete from Beginning</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 text-muted-foreground">All elements must shift left</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            <Clock className="inline h-4 w-4 mr-1" />
            Space Complexity: O(n) where n is the number of elements
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
                <h4 className="font-semibold text-foreground mb-1">Off-by-One Errors</h4>
                <p className="text-sm text-muted-foreground">
                  Remember that arrays are 0-indexed. An array of length 5 has indices 0 to 4, not 1 to 5.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Array Index Out of Bounds</h4>
                <p className="text-sm text-muted-foreground">
                  Always check if an index is valid before accessing. Accessing index -1 or length will cause errors in most languages.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Modifying While Iterating</h4>
                <p className="text-sm text-muted-foreground">
                  Avoid adding or removing elements while looping through an array. This can skip elements or cause infinite loops.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Tips */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interview Patterns</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Two Pointer Technique</h4>
                <p className="text-sm text-muted-foreground">
                  Use two pointers (start/end or slow/fast) to solve problems like finding pairs, 
                  reversing arrays, or detecting cycles.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Sliding Window</h4>
                <p className="text-sm text-muted-foreground">
                  Maintain a window of elements for problems involving subarrays - 
                  maximum sum subarray, longest substring, etc.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Prefix Sum</h4>
                <p className="text-sm text-muted-foreground">
                  Pre-calculate cumulative sums for quick range sum queries. 
                  Turns O(n) queries into O(1).
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Arrays Quiz" />
        </section>


{/* Top 10 Array Problems */}
<section>
  <h2 className="text-2xl font-bold mb-4 text-foreground">Top 10 Array Problems</h2>
  <p className="text-muted-foreground mb-4">
    Practice these essential array problems frequently asked in technical interviews.
    Each problem has a dedicated page with detailed explanations and solutions.
  </p>
  
  <div className="grid md:grid-cols-2 gap-4">
    {arrayProblems.map((problem) => (
      <Link 
        key={problem.id}
        href={`/learn/data-structures/arrays/problems/${problem.slug}`}
    className="group block p-4 bg-card border border-border rounded-lg hover:border-primary/50 transition-all duration-200 hover:shadow-md"
    >
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">
                #{problem.id}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded ${
                problem.difficulty === 'Easy' ? 'bg-green-500/10 text-green-500' :
                problem.difficulty === 'Medium' ? 'bg-yellow-500/10 text-yellow-500' :
                'bg-red-500/10 text-red-500'
              }`}>
                {problem.difficulty}
              </span>
            </div>
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors mb-1">
              {problem.title}
            </h3>
            <p className="text-sm text-muted-foreground">
              {problem.description}
            </p>
            <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
              <span> {problem.timeComplexity}</span>
              <span> {problem.spaceComplexity}</span>
            </div>
          </div>
          <ChevronRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
        </div>
      </Link>
    ))}
       </div>
     </section>
        
    </div>
    </TopicContent>
  );
}
