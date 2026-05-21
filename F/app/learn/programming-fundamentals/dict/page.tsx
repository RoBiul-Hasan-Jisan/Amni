// app/learn/programming-fundamentals/dict/page.tsx

import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb } from "lucide-react";

export default function DictPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "dict");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python",
      label: "Python",
      code: `# Creating dictionaries
data = {"a": 2, "b": 4, "c": 6}
built = dict(a=2, b=4, c=6)  # keyword args
built2 = dict([("a", 2), ("b", 4), ("c", 6)])  # from list of pairs
empty = {}

# Accessing elements
value = data["a"]        # 2 (raises KeyError if missing)
value = data.get("a")    # 2 (returns None if missing)
value = data.get("x", 0) # 0 (custom default)

# Adding/updating elements
data["d"] = 8            # add new key-value pair
data["a"] = 10           # update existing key

# Removing elements
data.pop("b")            # remove and return value
data.popitem()           # remove and return last inserted (key, value)
del data["c"]            # delete key-value pair

# Iterating through dictionary
for key in data:
    print(key, data[key])

for key, value in data.items():
    print(key, value)

# Dictionary comprehension
squares = {n: n * n for n in [2, 4, 6, 8]}

# Check if key exists
if "a" in data:
    print("Key exists")`,
    },
    {
      language: "javascript",
      label: "JavaScript",
      code: `// Creating objects (dictionaries)
const data = { a: 2, b: 4, c: 6 };
const empty = {};

// Accessing elements
const value = data["a"];     // 2
const value2 = data.a;       // 2 (dot notation)
const value3 = data["x"];    // undefined (no error)

// Adding/updating elements
data["d"] = 8;               // add new key-value pair
data.a = 10;                 // update existing key

// Removing elements
delete data.b;               // removes key-value pair

// Checking if key exists
const hasA = "a" in data;    // true
const hasX = data.hasOwnProperty("x"); // false

// Iterating
for (let key in data) {
    if (data.hasOwnProperty(key)) {
        console.log(key, data[key]);
    }
}

Object.keys(data).forEach(key => {
    console.log(key, data[key]);
});

Object.entries(data).forEach(([key, value]) => {
    console.log(key, value);
});`,
    },
    {
      language: "java",
      label: "Java",
      code: `import java.util.HashMap;
import java.util.Map;

// Creating HashMap
Map<String, Integer> data = new HashMap<>();
data.put("a", 2);
data.put("b", 4);
data.put("c", 6);

// Accessing elements
int value = data.get("a");      // 2 (returns null if missing)
int value2 = data.getOrDefault("x", 0); // 0

// Adding/updating elements
data.put("d", 8);               // add new key-value pair
data.put("a", 10);              // update existing key

// Removing elements
data.remove("b");               // remove and return value
data.remove("c", 6);            // remove only if value matches

// Checking if key exists
boolean hasA = data.containsKey("a");  // true

// Iterating
for (Map.Entry<String, Integer> entry : data.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

data.forEach((key, value) -> {
    System.out.println(key + ": " + value);
});`,
    },
    {
      language: "cpp",
      label: "C++",
      code: `#include <unordered_map>
#include <iostream>

// Creating unordered_map
std::unordered_map<std::string, int> data;
data["a"] = 2;
data["b"] = 4;
data["c"] = 6;

// Accessing elements
int value = data["a"];           // 2 (creates if missing)
int value2 = data.at("a");       // 2 (throws if missing)

// Adding/updating elements
data["d"] = 8;                   // add new key-value pair
data["a"] = 10;                  // update existing key

// Removing elements
data.erase("b");                 // remove key-value pair

// Checking if key exists
bool hasA = data.find("a") != data.end();  // true

// Iterating
for (const auto& pair : data) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the time complexity of accessing a value by key in a hash table/dictionary?",
      options: ["O(1)", "O(n)", "O(log n)", "O(n^2)"],
      correctAnswer: 0,
      explanation: "Dictionaries use hash tables providing average O(1) lookup time, making them extremely efficient for key-based access.",
    },
    {
      id: 2,
      question: "Which of the following can be used as dictionary keys?",
      options: ["Lists", "Dictionaries", "Strings", "All of the above"],
      correctAnswer: 2,
      explanation: "Keys must be hashable (immutable). Strings, numbers, and tuples are hashable. Lists and dictionaries are mutable and cannot be keys.",
    },
    {
      id: 3,
      question: "What happens when you insert a duplicate key into a dictionary?",
      options: [
        "Error is thrown",
        "Both values are stored",
        "The old value is overwritten",
        "The new value is ignored"
      ],
      correctAnswer: 2,
      explanation: "Dictionaries have unique keys. Inserting a duplicate key overwrites the existing value with the new one.",
    },
    {
      id: 4,
      question: "What does the `get()` method return when a key is not found (without a default argument)?",
      options: ["KeyError", "None", "0", "Empty string"],
      correctAnswer: 1,
      explanation: "The `get()` method returns `None` (or a specified default) when a key is missing, unlike bracket notation which raises KeyError.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8 border-b border-border pb-4">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Dictionaries
          </h1>
          <p className="text-muted-foreground text-lg">
            Dictionaries store key–value pairs. Keys must be hashable; values can be any type. 
            As of Python 3.7, dictionaries preserve insertion order.
          </p>
        </div>

        {/* Creating Dictionaries */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Creating Dictionaries
          </h2>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono text-foreground">Literal {`{}`}</td>
                  <td className="p-3 font-mono">data = {"{"}"a": 2, "b": 4{"}"}</td>
                  <td className="p-3">You know the key-value pairs</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono text-foreground">Empty dict</td>
                  <td className="p-3 font-mono">data = {`{}`}</td>
                  <td className="p-3">Start empty, add pairs later</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono text-foreground">dict()</td>
                  <td className="p-3 font-mono">dict() → {`{}`}</td>
                  <td className="p-3">Empty dict (same as {`{}`})</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono text-foreground">dict(kwargs)</td>
                  <td className="p-3 font-mono">dict(a=2, b=4)</td>
                  <td className="p-3">Keys are valid identifiers</td>
                </tr>
                <tr>
                  <td className="p-3 font-mono text-foreground">dict(zip())</td>
                  <td className="p-3 font-mono">dict(zip(keys, values))</td>
                  <td className="p-3">Build from two sequences</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
empty = {}
from_range = dict.fromkeys(["a", "b", "c"], 0)  # {"a": 0, "b": 0, "c": 0}
from_zip = dict(zip(["x", "y", "z"], [2, 4, 6]))  # {"x": 2, "y": 4, "z": 6}

len(data)    # 3
len(empty)   # 0
if data:     # True (non-empty)
if not empty:  # True (empty)`}
            </pre>
          </div>
        </section>

        {/* Dictionary Properties */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Dictionary Properties
          </h2>
          <ul className="list-disc list-inside text-muted-foreground space-y-2 mb-4">
            <li><span className="font-semibold text-foreground">Ordered</span> (Python 3.7+) - insertion order is preserved; iteration order is stable.</li>
            <li><span className="font-semibold text-foreground">Changeable</span> - add, change, or remove items after creation.</li>
            <li><span className="font-semibold text-foreground">Unique keys</span> - duplicate keys overwrite earlier values; only the last wins.</li>
            <li><span className="font-semibold text-foreground">Hashable keys</span> - keys must be immutable (strings, numbers, tuples of immutables).</li>
          </ul>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6, "b": 8}  # "b": 4 overwritten
print(data)  # {"a": 2, "b": 8, "c": 6}`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Mixed value types:</span> Values can be strings, numbers, lists, dictionaries, or any type. 
              Keys must be hashable (strings, numbers, tuples of hashables).
            </p>
          </div>
        </section>

        {/* Dictionary Methods Reference */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Dictionary Methods
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">get(key, default)</td><td className="p-3">Return value for key; default if missing (no KeyError)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">keys()</td><td className="p-3">View of keys</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">values()</td><td className="p-3">View of values</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">items()</td><td className="p-3">View of (key, value) pairs</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">fromkeys(seq, value)</td><td className="p-3">New dict with keys from seq, values value (default None)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">setdefault(key, default)</td><td className="p-3">Return value; if key missing, insert default and return it</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">update(other)</td><td className="p-3">Update dict with key–value pairs from other</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">pop(key, default)</td><td className="p-3">Remove key, return value; KeyError if missing (or default)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">popitem()</td><td className="p-3">Remove and return last inserted (key, value)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">clear()</td><td className="p-3">Remove all elements from the dictionary</td></tr>
                <tr><td className="p-3 font-mono text-foreground">copy()</td><td className="p-3">Return a shallow copy of the dictionary</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Access Dictionary Items */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Access Dictionary Items
          </h2>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Behavior</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono text-foreground">d[key]</td>
                  <td className="p-3">Raises KeyError if missing</td>
                  <td className="p-3 font-mono">data["a"] → 2</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono text-foreground">d.get(key)</td>
                  <td className="p-3">Returns None if missing</td>
                  <td className="p-3 font-mono">data.get("x") → None</td>
                </tr>
                <tr>
                  <td className="p-3 font-mono text-foreground">d.get(key, default)</td>
                  <td className="p-3">Returns default if missing</td>
                  <td className="p-3 font-mono">data.get("x", 0) → 0</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
data["b"]       # 4
data["x"]       # KeyError
data.get("b")   # 4
data.get("x")   # None
data.get("x", 0)  # 0 - custom default`}
            </pre>
          </div>
        </section>

        {/* keys(), values(), items() */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            keys(), values(), items()
          </h2>
          <p className="text-muted-foreground mb-4">
            These return <span className="font-semibold text-foreground">view objects</span> - they reflect the current state of the dictionary. 
            Changes to the dictionary are visible in the view; the view is not a snapshot.
          </p>

          <div className="bg-muted/30 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
k = data.keys()
print(list(k))  # ["a", "b", "c"]
data["d"] = 8
print(list(k))  # ["a", "b", "c", "d"] - view updated`}
            </pre>
          </div>

          <ul className="list-disc list-inside text-muted-foreground space-y-1 mb-4">
            <li><code className="bg-muted px-1.5 py-0.5 rounded font-mono text-foreground">keys()</code> - view of keys</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded font-mono text-foreground">values()</code> - view of values</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded font-mono text-foreground">items()</code> - view of (key, value) tuples</li>
          </ul>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              To get a snapshot (list), wrap in <code className="bg-muted px-1.5 py-0.5 rounded">list()</code>: <code className="bg-muted px-1.5 py-0.5 rounded">list(d.items())</code>.
            </p>
          </div>
        </section>

        {/* setdefault() and fromkeys() */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            setdefault() & fromkeys()
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">setdefault(key, default):</span> if key exists, return its value. 
                If not, set <code className="bg-muted px-1.5 py-0.5 rounded">d[key] = default</code> and return default. Useful for "get or create".
              </p>
              <div className="bg-muted/30 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`data = {"a": 2, "b": 4}
data.setdefault("c", 6)   # 6; data["c"] = 6
data.setdefault("a", 0)   # 2 - key exists, unchanged`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">fromkeys(seq, value):</span> create a dictionary with keys from seq; 
                all values set to value (default None).
              </p>
              <div className="bg-muted/30 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`dict.fromkeys(["a", "b", "c"])       # {"a": None, "b": None, "c": None}
dict.fromkeys(["a", "b", "c"], 0)    # {"a": 0, "b": 0, "c": 0}`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Check if Key Exists */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Check if Key Exists
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <code className="bg-muted px-1.5 py-0.5 rounded font-mono">in</code> and <code className="bg-muted px-1.5 py-0.5 rounded font-mono">not in</code> on the dictionary. 
            <span className="font-semibold text-foreground"> Membership tests keys only, not values</span> - so a value like <code className="bg-muted px-1.5 py-0.5 rounded">2</code> is not "in" the dictionary unless it is a key.
          </p>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
"b" in data     # True  - "b" is a key
"x" not in data # True
2 in data       # False - 2 is a value, not a key`}
            </pre>
          </div>
        </section>

        {/* Add & Update Dictionary Items */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Add & Update Items
          </h2>
          
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">d[key] = value</td><td className="p-3">Add new key-value pair or update existing key</td></tr>
                <tr><td className="p-3 font-mono text-foreground">d.update(other)</td><td className="p-3">Merge another dict or iterable of pairs</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4}
data["c"] = 6           # add - {"a": 2, "b": 4, "c": 6}
data["b"] = 8           # update - {"a": 2, "b": 8, "c": 6}

data.update({"d": 10, "e": 12})  # add multiple
data.update(f=14, g=16)          # keyword args
print(data)  # {"a": 2, "b": 8, "c": 6, "d": 10, "e": 12, "f": 14, "g": 16}`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">update() return value:</span> update() mutates the dictionary and returns None; 
              do not assign the result and expect a new dictionary.
            </p>
          </div>
        </section>

        {/* Remove Dictionary Items */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Remove Dictionary Items
          </h2>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Method / Keyword</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">pop(key)</td><td className="p-3">Remove key, return value; KeyError if missing</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">pop(key, default)</td><td className="p-3">Remove key, return value; return default if missing</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">popitem()</td><td className="p-3">Remove and return last inserted (key, value); Python 3.7+ LIFO</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">del d[key]</td><td className="p-3">Remove item; KeyError if missing</td></tr>
                <tr><td className="p-3 font-mono text-foreground">clear()</td><td className="p-3">Remove all items; dict becomes {`{}`}</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6, "d": 8}
data.pop("b")      # 4; data = {"a": 2, "c": 6, "d": 8}
data.pop("x", 0)   # 0 - key missing, default returned
data.popitem()     # ("d", 8) - last inserted in Python 3.7+
del data["a"]      # remove "a"
data.clear()       # {}`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">popitem() and version:</span> In Python 3.7+, removes the last inserted item (LIFO). 
              In Python 3.6 and earlier, removes an arbitrary item.
            </p>
          </div>
        </section>

        {/* Unpacking Dictionaries */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Unpacking Dictionaries
          </h2>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Pattern</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Result / note</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Spread into dict</td><td className="p-3 font-mono">{`{**base, **override}`}</td><td className="p-3">New dict; override keys take precedence</td></tr>
                <tr><td className="p-3">Function arguments</td><td className="p-3 font-mono">func(**kwargs)</td><td className="p-3">Unpack dict as keyword arguments</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`base = {"a": 2, "b": 4, "c": 6}
extended = {**base, "c": 8, "d": 10}   # c overwritten, d added
# {"a": 2, "b": 4, "c": 8, "d": 10}

def display(name, age):
    print(f"{name} is {age} years old")

person = {"name": "Alice", "age": 30}
display(**person)  # Alice is 30 years old`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Unpacking creates a new dictionary:</span> Using {`{**other}`} in a dictionary literal builds a new dictionary; 
              does not mutate other. Later keys overwrite earlier ones.
            </p>
          </div>
        </section>

        {/* Loop Dictionaries */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Loop Dictionaries
          </h2>
          <ul className="list-disc list-inside text-muted-foreground space-y-2 mb-4">
            <li><span className="font-semibold text-foreground">By keys (default):</span> <code className="bg-muted px-1.5 py-0.5 rounded">for k in d</code> iterates over keys.</li>
            <li><span className="font-semibold text-foreground">By values:</span> <code className="bg-muted px-1.5 py-0.5 rounded">for v in d.values()</code>.</li>
            <li><span className="font-semibold text-foreground">By keys and values:</span> <code className="bg-muted px-1.5 py-0.5 rounded">for k, v in d.items()</code>.</li>
          </ul>

          <div className="bg-muted/30 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
for k in data:
    print(k, data[k])  # a 2, b 4, c 6

for k in data.keys():
    print(k)  # a, b, c - explicit keys

for v in data.values():
    print(v)  # 2, 4, 6

for k, v in data.items():
    print(k, v)  # a 2, b 4, c 6`}
            </pre>
          </div>

          <div>
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Loop in reverse:</span> To iterate in reverse insertion order, 
              use <code className="bg-muted px-1.5 py-0.5 rounded">reversed()</code> on the dictionary or on <code className="bg-muted px-1.5 py-0.5 rounded">d.items()</code> (Python 3.8+).
            </p>
            <div className="bg-muted/30 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`data = {"a": 2, "b": 4, "c": 6}
for k in reversed(data):
    print(k, data[k])  # c 6, b 4, a 2

for k, v in reversed(data.items()):
    print(k, v)  # c 6, b 4, a 2`}
              </pre>
            </div>
          </div>
        </section>

        {/* Dictionary Comprehension */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Dictionary Comprehension
          </h2>
          <p className="text-muted-foreground mb-4">
            Build a dictionary from an iterable with <code className="bg-muted px-1.5 py-0.5 rounded">{'{key_expr: value_expr for ... in ...}'}</code>. 
            You can add an <code className="bg-muted px-1.5 py-0.5 rounded">if</code> to filter.
          </p>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Syntax: {key_expr: value_expr for item in iterable if condition}

squares = {n: n * n for n in [2, 4, 6, 8]}   # {2: 4, 4: 16, 6: 36, 8: 64}
upper = {k.upper(): v for k, v in [("a", 2), ("b", 4)]}  # {"A": 2, "B": 4}
evens_only = {x: x * 2 for x in range(6) if x % 2 == 0}  # {0: 0, 2: 4, 4: 8}

# Reverse dictionary (values must be unique)
original = {"uk": "United Kingdom", "de": "Germany"}
reversed_dict = {v: k for k, v in original.items()}`}
            </pre>
          </div>
        </section>

        {/* Copy Dictionaries */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Copy Dictionaries
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Shallow Copy</h3>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">copy()</td><td className="p-3 font-mono">new = original.copy()</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono text-foreground">dict()</td><td className="p-3 font-mono">new = dict(original)</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mb-4 rounded">
            <p className="font-semibold text-foreground">When to use shallow copy:</p>
            <p className="text-muted-foreground">The dict is flat or contains only immutable values (ints, strings). Shallow copy is faster and sufficient for most cases.</p>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Deep Copy</h3>
          <div className="bg-muted/30 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`import copy

# Shallow copy - fine for flat dicts
a = {"x": 2, "y": 4}
b = a.copy()
a["z"] = 6  # a = {"x": 2, "y": 4, "z": 6}; b unchanged

# Shallow copy - nested dicts are shared
original = {"a": {"x": 2, "y": 4}, "b": {"x": 6, "y": 8}}
shallow = original.copy()
shallow["a"]["x"] = 0   # original["a"]["x"] is also 0 - shared

# Deep copy - fully independent
deep = copy.deepcopy(original)
deep["a"]["x"] = 0      # original unchanged - independent`}
            </pre>
          </div>
        </section>

        {/* Sort Dictionaries */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Sort Dictionaries
          </h2>
          <p className="text-muted-foreground mb-4">
            Dictionaries have no <code className="bg-muted px-1.5 py-0.5 rounded">sort()</code> method. To iterate in sorted order, 
            use <code className="bg-muted px-1.5 py-0.5 rounded">sorted()</code> on keys or <code className="bg-muted px-1.5 py-0.5 rounded">items()</code>.
          </p>

          <div className="bg-muted/30 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"c": 6, "a": 2, "b": 4}
for k in sorted(data):
    print(k, data[k])  # a 2, b 4, c 6

for k, v in sorted(data.items()):
    print(k, v)  # a 2, b 4, c 6

# Sort by value
for k, v in sorted(data.items(), key=lambda p: p[1]):
    print(k, v)  # a 2, b 4, c 6`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">New dictionary from sorted items:</span> 
              <code className="bg-muted px-1.5 py-0.5 rounded">dict(sorted(data.items()))</code> - creates a dictionary with keys in sorted order.
            </p>
          </div>
        </section>

        {/* Nested Dictionaries */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Nested Dictionaries
          </h2>
          <p className="text-muted-foreground mb-4">
            Values can be dictionaries. Access with chained brackets.
          </p>

          <div className="bg-muted/30 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nested = {
    "p1": {"x": 2, "y": 4},
    "p2": {"x": 6, "y": 8},
}
nested["p1"]["x"]   # 2
nested["p2"]["y"]   # 8

# Loop nested dictionaries
for key, obj in nested.items():
    print(key)
    for k, v in obj.items():
        print(f"  {k}: {v}")`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 rounded">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Safe access in nested dicts:</span> 
              <code className="bg-muted px-1.5 py-0.5 rounded">d.get("a", {}).get("b", default)</code> avoids KeyError.
            </p>
          </div>
        </section>

        {/* Dictionary vs Other Collections */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Dictionary vs Other Collections
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted">
                  <th className="text-left p-3 font-semibold text-foreground">Collection</th>
                  <th className="text-left p-3 font-semibold text-foreground">Ordered?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Mutable?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Duplicates allowed?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">dict</td><td className="p-3">Yes (insertion)</td><td className="p-3">Yes (values)</td><td className="p-3">No (keys)</td><td className="p-3">Key-value mapping; fast lookup by key</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">list</td><td className="p-3">Yes</td><td className="p-3">Yes</td><td className="p-3">Yes</td><td className="p-3">Ordered sequence; position matters</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">tuple</td><td className="p-3">Yes</td><td className="p-3">No</td><td className="p-3">Yes</td><td className="p-3">Immutable sequence; can be dict key</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">set</td><td className="p-3">No</td><td className="p-3">Yes</td><td className="p-3">No</td><td className="p-3">Unique items; set operations</td></tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Note: Dictionary order is guaranteed from Python 3.7 onward (insertion order). In Python 3.6 and earlier, iteration order was not guaranteed.
          </p>
        </section>

        {/* Common Use Cases */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Common Use Cases
          </h2>
          <div className="space-y-4">
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Configuration / Settings</h4>
              <pre className="text-xs bg-muted/30 p-2 rounded font-mono text-foreground">
                {`config = {
    "theme": "dark",
    "language": "en",
    "notifications": True,
    "timeout": 30
}`}
              </pre>
            </div>
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Counting Frequencies</h4>
              <pre className="text-xs bg-muted/30 p-2 rounded font-mono text-foreground">
                {`counts = {}
for item in items:
    counts[item] = counts.get(item, 0) + 1`}
              </pre>
            </div>
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Grouping Items</h4>
              <pre className="text-xs bg-muted/30 p-2 rounded font-mono text-foreground">
                {`groups = {}
for category, item in data:
    groups.setdefault(category, []).append(item)`}
              </pre>
            </div>
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Memoization / Caching</h4>
              <pre className="text-xs bg-muted/30 p-2 rounded font-mono text-foreground">
                {`cache = {}
def expensive_function(n):
    if n in cache:
        return cache[n]
    result = compute(n)
    cache[n] = result
    return result`}
              </pre>
            </div>
          </div>
        </section>

        {/* Tricky Points */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">Creating - Empty dictionary</p>
              <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">{`{}`}</code> is falsy; use <code className="bg-muted px-1.5 py-0.5 rounded">if not d:</code> for "no keys".</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">dict() keyword args</p>
              <p className="text-sm text-muted-foreground">Keys must be valid identifiers; use {`{"2": 4}`} for numeric-like keys.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">Access - d[key] vs d.get(key)</p>
              <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">[]</code> raises KeyError if missing; <code className="bg-muted px-1.5 py-0.5 rounded">get()</code> returns None or default.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">keys(), values(), items()</p>
              <p className="text-sm text-muted-foreground">Return views, not lists; reflect live changes; modifying dict size while iterating raises RuntimeError.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">Check if Key Exists - in</p>
              <p className="text-sm text-muted-foreground">Membership tests <span className="font-semibold">keys</span> only; values not tested.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">Copy - Shallow vs Deep</p>
              <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">copy()</code> copies top level only. Nested structures are shared. Use <code className="bg-muted px-1.5 py-0.5 rounded">deepcopy()</code> for full independence.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded">
              <p className="font-semibold text-foreground mb-1">Sorting</p>
              <p className="text-sm text-muted-foreground">Dictionaries have no <code className="bg-muted px-1.5 py-0.5 rounded">sort()</code>; use <code className="bg-muted px-1.5 py-0.5 rounded">sorted(d)</code> for ordered iteration.</p>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Interview Questions
          </h2>
          <div className="space-y-5">
            <div>
              <p className="font-semibold text-foreground mb-1">What is the time complexity of dictionary operations?</p>
              <p className="text-muted-foreground">Average O(1) for get, set, and delete. Worst case O(n) due to hash collisions.</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-1">Why must dictionary keys be hashable?</p>
              <p className="text-muted-foreground">Dictionaries use hash tables; keys are hashed for O(1) lookup. Mutable types (lists, dicts) are unhashable.</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-1">Are dictionaries ordered?</p>
              <p className="text-muted-foreground">Yes, since Python 3.7 (insertion order). In Python 3.6 it was an implementation detail.</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-1">When to use d[key] vs d.get(key)?</p>
              <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">get()</code> when key may be missing and you want a default. Use <code className="bg-muted px-1.5 py-0.5 rounded">[]</code> when key must exist.</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-1">How do you merge two dictionaries?</p>
              <p className="text-muted-foreground">Use {`{**d1, **d2}`} (Python 3.5+) or <code className="bg-muted px-1.5 py-0.5 rounded">d1 | d2</code> (Python 3.9+). For in-place merge, use <code className="bg-muted px-1.5 py-0.5 rounded">d1.update(d2)</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-1">How do you create a dictionary from two lists?</p>
              <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">dict(zip(keys_list, values_list))</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-1">What's the difference between pop() and popitem()?</p>
              <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">pop(key)</code> removes a specific key. <code className="bg-muted px-1.5 py-0.5 rounded">popitem()</code> removes the last inserted item (LIFO in Python 3.7+).</p>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Dictionaries Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}