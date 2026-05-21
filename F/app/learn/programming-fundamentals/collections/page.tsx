import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb } from "lucide-react";

export default function CollectionsPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "collections");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python",
      label: "Python",
      code: `# Lists - Ordered, mutable
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
fruits[0] = "strawberry"

# Tuples - Ordered, immutable
coordinates = (10, 20)
x, y = coordinates  # Unpacking

# Dictionaries - Key-value pairs
person = {"name": "Alice", "age": 30}
person["email"] = "alice@example.com"

# Sets - Unordered, unique elements
unique_numbers = {1, 2, 3, 3, 2}  # {1, 2, 3}
unique_numbers.add(4)

# List comprehension
squares = [x**2 for x in range(10)]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}`,
    },
    {
      language: "javascript",
      label: "JavaScript",
      code: `// Arrays - Ordered, mutable
const fruits = ["apple", "banana", "cherry"];
fruits.push("orange");
fruits[0] = "strawberry";

// Objects (like dictionaries)
const person = { name: "Alice", age: 30 };
person.email = "alice@example.com";

// Sets - Unique values
const uniqueNumbers = new Set([1, 2, 3, 3, 2]); // {1, 2, 3}
uniqueNumbers.add(4);

// Maps - Key-value with any key type
const map = new Map();
map.set("name", "Alice");
map.set(42, "answer");

// Array methods
const squares = Array.from({length: 10}, (_, i) => i ** 2);`,
    },
    {
      language: "java",
      label: "Java",
      code: `import java.util.*;

// ArrayList - Dynamic array
List<String> fruits = new ArrayList<>();
fruits.add("apple");
fruits.add("banana");
fruits.set(0, "strawberry");

// HashMap - Key-value pairs
Map<String, Integer> person = new HashMap<>();
person.put("age", 30);
person.put("name", "Alice");

// HashSet - Unique elements
Set<Integer> uniqueNumbers = new HashSet<>();
uniqueNumbers.add(1);
uniqueNumbers.add(2);
uniqueNumbers.add(2); // Duplicate ignored

// Array and List conversion
int[] arr = {1, 2, 3, 4, 5};
List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);`,
    },
    {
      language: "cpp",
      label: "C++",
      code: `#include <vector>
#include <map>
#include <unordered_set>
#include <string>

// Vector - Dynamic array
std::vector<std::string> fruits = {"apple", "banana", "cherry"};
fruits.push_back("orange");
fruits[0] = "strawberry";

// Map - Key-value pairs
std::map<std::string, int> person;
person["name"] = 1; // Using int for simplicity
person["age"] = 30;

// Unordered set - Unique elements
std::unordered_set<int> uniqueNumbers = {1, 2, 3, 3, 2}; // {1, 2, 3}
uniqueNumbers.insert(4);

// Array (fixed size)
int arr[] = {1, 2, 3, 4, 5};`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which collection type is immutable (cannot be changed after creation)?",
      options: ["List", "Dictionary", "Tuple", "Set"],
      correctAnswer: 2,
      explanation: "Tuples are immutable sequences that cannot be modified after creation. Lists, dictionaries, and sets are mutable.",
    },
    {
      id: 2,
      question: "What is the time complexity of checking if an item exists in a set?",
      options: ["O(1)", "O(n)", "O(log n)", "O(n²)"],
      correctAnswer: 0,
      explanation: "Sets use hash tables for O(1) average time complexity for membership testing (the 'in' operator).",
    },
    {
      id: 3,
      question: "Which collection type would you use to store key-value pairs?",
      options: ["List", "Tuple", "Set", "Dictionary"],
      correctAnswer: 3,
      explanation: "Dictionaries store key-value pairs, allowing fast lookup by key. Lists store ordered sequences, tuples are immutable sequences, and sets store unique values.",
    },
    {
      id: 4,
      question: "What happens when you create a set with duplicate values?",
      options: [
        "It raises an error",
        "It keeps all duplicates",
        "It keeps only unique values",
        "It sorts the values"
      ],
      correctAnswer: 2,
      explanation: "Sets automatically remove duplicates, keeping only unique values. This is one of the key features of sets.",
    },
    {
      id: 5,
      question: "Which collection type maintains the order of elements?",
      options: ["Set only", "List and Tuple only", "Dictionary only", "List, Tuple, and Dictionary (Python 3.7+)"],
      correctAnswer: 3,
      explanation: "Lists and tuples maintain order by definition. Since Python 3.7, dictionaries also preserve insertion order. Sets are unordered.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What are Collections?</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Collections</strong> are data structures that store multiple items in a single object. 
            Python provides four main collection types: <strong>lists</strong>, <strong>tuples</strong>, <strong>dictionaries</strong>, 
            and <strong>sets</strong> — each with unique properties and use cases.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Think of collections like different types of containers: a <strong>list</strong> is like a shopping list (ordered, can change), 
                  a <strong>tuple</strong> is like coordinates (fixed, can't change), a <strong>dictionary</strong> is like a phone book 
                  (find values by keys), and a <strong>set</strong> is like a bag of unique items (no duplicates, order doesn't matter).
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Quick Reference */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Quick Reference</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Collection</th>
                  <th className="border border-border p-3 text-left text-foreground">Syntax</th>
                  <th className="border border-border p-3 text-left text-foreground">Ordered</th>
                  <th className="border border-border p-3 text-left text-foreground">Mutable</th>
                  <th className="border border-border p-3 text-left text-foreground">Duplicates</th>
                </tr>
              </thead>
              <tbody>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 font-semibold text-foreground">List</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">[1, 2, 3]</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 font-semibold text-foreground">Tuple</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">(1, 2, 3)</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                  <td className="border border-border p-3 text-muted-foreground">✗</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 font-semibold text-foreground">Dictionary</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">{"{key: value}"}</td>
                  <td className="border border-border p-3 text-muted-foreground">✓*</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                  <td className="border border-border p-3 text-muted-foreground">Keys: ✗, Values: ✓</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 font-semibold text-foreground">Set</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">{"{1, 2, 3}"}</td>
                  <td className="border border-border p-3 text-muted-foreground">✗</td>
                  <td className="border border-border p-3 text-muted-foreground">✓</td>
                  <td className="border border-border p-3 text-muted-foreground">✗</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            *Dictionaries preserve insertion order in Python 3.7+
          </p>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Code Examples Across Languages</h2>
          <p className="text-muted-foreground mb-4">
            See how different collection types are implemented across programming languages:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Lists Deep Dive */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Lists</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Lists</strong> are ordered, mutable sequences that can contain mixed types.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Creating Lists</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Accessing Elements</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`fruits[0]     # First: 'apple'
fruits[-1]    # Last: 'cherry'
fruits[1:3]   # Slicing: ['banana', 'cherry']`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Adding Elements</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`fruits.append("orange")     # Add to end
fruits.insert(1, "berry")   # Insert at index
fruits.extend(["grape"])     # Add multiple`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Removing Elements</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`fruits.remove("banana")    # Remove by value
last = fruits.pop()         # Remove last
item = fruits.pop(1)        # Remove at index
fruits.clear()              # Empty list`}
              </pre>
            </div>
          </div>

          <div className="bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-foreground">List Comprehensions</h4>
            <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
              {`# Basic comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# With if-else
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]`}
            </pre>
          </div>
        </section>

        {/* Tuples Deep Dive */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Tuples</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Tuples</strong> are ordered, immutable sequences. They're faster than lists and can be used as dictionary keys.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Creating Tuples</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`coordinates = (10, 20)
person = ("Alice", 30, "NYC")
single = (5,)  # Comma required!
empty = ()`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Tuple Unpacking</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`x, y = coordinates  # x=10, y=20
name, age, city = person

# Swap variables
a, b = b, a`}
              </pre>
            </div>
          </div>

          <div className="mt-4 bg-primary/5 border border-primary/20 rounded-lg p-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Tuple Use Cases</h4>
                <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                  <li>Returning multiple values from functions</li>
                  <li>Dictionary keys (lists cannot be used as keys)</li>
                  <li>Fixed data that shouldn't change (e.g., coordinates, RGB values)</li>
                  <li>More memory-efficient than lists for static data</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Dictionaries Deep Dive */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Dictionaries</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Dictionaries</strong> store key-value pairs for fast lookup. Keys must be immutable (strings, numbers, tuples).
          </p>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Creating Dictionaries</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`person = {"name": "Alice", "age": 30}
dict1 = dict(name="Bob", age=25)
dict2 = dict([("a", 1), ("b", 2)])`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Accessing Values</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`person["name"]              # 'Alice'
person.get("age")           # 30
person.get("country", "USA") # Default value`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Modifying Dictionaries</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`person["email"] = "alice@example.com"  # Add
person["age"] = 31                     # Update
del person["city"]                     # Delete`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Iterating</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`for key in person:
    print(key)
    
for key, value in person.items():
    print(f"{key}: {value}")`}
              </pre>
            </div>
          </div>

          <div className="bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-foreground">Dictionary Comprehension</h4>
            <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
              {`# Create dictionary of squares
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filter items
evens = {x: x**2 for x in range(10) if x % 2 == 0}`}
            </pre>
          </div>
        </section>

        {/* Sets Deep Dive */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Sets</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Sets</strong> are unordered collections of unique elements. They offer O(1) membership testing.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Creating Sets</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`fruits = {"apple", "banana", "cherry"}
numbers = set([1, 2, 3, 3, 2])  # {1, 2, 3}
empty = set()  # {} creates dict!`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Set Operations</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

A | B  # Union: {1,2,3,4,5,6}
A & B  # Intersection: {3,4}
A - B  # Difference: {1,2}
A ^ B  # Symmetric diff: {1,2,5,6}`}
              </pre>
            </div>
          </div>

          <div className="bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-foreground">Set Operations & Methods</h4>
            <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
              {`# Adding and removing
fruits.add("orange")
fruits.remove("banana")     # Raises KeyError if not found
fruits.discard("grape")     # No error if not found

# Membership (fast O(1))
if "apple" in fruits:
    print("Found!")

# Remove duplicates from list
unique = list(set([1, 2, 2, 3, 3, 3]))  # [1, 2, 3]`}
            </pre>
          </div>
        </section>

        {/* Performance Comparison */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Performance Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Operation</th>
                  <th className="border border-border p-3 text-left text-foreground">List</th>
                  <th className="border border-border p-3 text-left text-foreground">Set</th>
                  <th className="border border-border p-3 text-left text-foreground">Dict</th>
                </tr>
              </thead>
              <tbody>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 text-foreground">Index/Access</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 text-muted-foreground">-</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 text-foreground">Search (in)</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 text-foreground">Insert</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)*</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 text-foreground">Delete</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                </tr>
                <tr className="hover:bg-muted/50">
                  <td className="border border-border p-3 text-foreground">Memory/Item</td>
                  <td className="border border-border p-3 text-muted-foreground">Low</td>
                  <td className="border border-border p-3 text-muted-foreground">High</td>
                  <td className="border border-border p-3 text-muted-foreground">High</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            *Append is O(1), insert at arbitrary position is O(n)
          </p>
        </section>

        {/* Choosing the Right Collection */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Choosing the Right Collection</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">✓ Use a List when:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>Order matters</li>
                <li>You need indexing</li>
                <li>Duplicates are allowed</li>
                <li>You frequently modify the collection</li>
              </ul>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">✓ Use a Tuple when:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>Order matters and shouldn't change</li>
                <li>The data is fixed (e.g., coordinates)</li>
                <li>You need a dictionary key</li>
                <li>Performance is critical (tuples are faster)</li>
              </ul>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">✓ Use a Dict when:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>You need fast key-based lookup</li>
                <li>Keys have meaningful names</li>
                <li>You need to associate values</li>
                <li>Order matters (Python 3.7+ preserves order)</li>
              </ul>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">✓ Use a Set when:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>You need unique elements</li>
                <li>Order doesn't matter</li>
                <li>Fast membership testing is needed</li>
                <li>You perform set operations (union, intersection)</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Mistakes to Avoid</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Modifying List While Iterating</h4>
                <p className="text-sm text-muted-foreground">
                  Modifying a list during iteration can cause skipped elements. Iterate over a copy instead.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong
for item in my_list:
    if condition:
        my_list.remove(item)

# Correct
for item in my_list[:]:  # Iterate over copy
    if condition:
        my_list.remove(item)`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Using List as Dictionary Key</h4>
                <p className="text-sm text-muted-foreground">
                  Lists are mutable and cannot be used as dictionary keys. Use tuples instead.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong
my_dict = {[1, 2]: "value"}  # TypeError!

# Correct
my_dict = {(1, 2): "value"}  # Works!`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Mutable Default Arguments</h4>
                <p className="text-sm text-muted-foreground">
                  Using mutable default arguments persists across function calls.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong
def add_item(item, my_list=[]):
    my_list.append(item)
    return my_list

# Correct
def add_item(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Shallow vs Deep Copies</h4>
                <p className="text-sm text-muted-foreground">
                  <code className="bg-muted px-1 rounded">.copy()</code> creates shallow copies. Nested objects are still referenced.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`import copy

# Shallow copy
new_list = original_list.copy()

# Deep copy (for nested structures)
new_list = copy.deepcopy(original_list)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Best Practices */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Best Practices</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use List Comprehensions</h4>
                <p className="text-sm text-muted-foreground">
                  List comprehensions are more readable and faster than manual loops for creating lists.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Tuples for Fixed Data</h4>
                <p className="text-sm text-muted-foreground">
                  If data shouldn't change, use a tuple. It's more memory-efficient and signals intent to other developers.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Sets for Membership Testing</h4>
                <p className="text-sm text-muted-foreground">
                  When you need to check if an item exists in a collection, sets provide O(1) lookup vs O(n) for lists.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use get() for Dictionaries</h4>
                <p className="text-sm text-muted-foreground">
                  Use <code className="bg-muted px-1 rounded">dict.get()</code> to safely access values without risking KeyError.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Collections Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}