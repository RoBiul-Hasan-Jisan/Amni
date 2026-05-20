// app/learn/programming-fundamentals/collections/page.tsx

import Link from "next/link";

export default function CollectionsPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation Bar */}
      <nav className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/learn" className="text-blue-600 dark:text-blue-400 hover:underline">
                ← Back to Learning Path
              </Link>
            </div>
            <div className="text-gray-700 dark:text-gray-300 font-semibold">
              Programming Fundamentals
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Collections
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Collections store multiple items in a single data structure. Python provides four main collection types: lists, tuples, dictionaries, and sets — each with unique properties and use cases.
          </p>
        </div>

        {/* Lists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Lists
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Ordered, mutable sequences that can contain mixed types. Lists are defined with square brackets <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">[]</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Creating and accessing lists</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Creating lists
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Accessing elements (0-indexed)
print(fruits[0])    # apple
print(fruits[-1])   # cherry (last element)
print(fruits[1:3])  # ['banana', 'cherry'] (slicing)`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Modifying lists:</span> Add, remove, and change elements.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">List operations</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Adding elements
fruits.append("orange")        # Add to end
fruits.insert(1, "blueberry")  # Insert at index
fruits.extend(["grape", "kiwi"]) # Add multiple

# Removing elements
fruits.remove("banana")        # Remove by value
last = fruits.pop()            # Remove and return last
item = fruits.pop(1)           # Remove and return at index
del fruits[0]                  # Delete by index
fruits.clear()                 # Empty the list

# Modifying elements
fruits[0] = "strawberry"       # Change element

# Checking existence
if "apple" in fruits:
    print("Found!")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">List methods and operations:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`numbers = [3, 1, 4, 1, 5, 9, 2]

# Length
len(numbers)                    # 7

# Sorting
numbers.sort()                  # Sorts in-place: [1, 1, 2, 3, 4, 5, 9]
numbers.sort(reverse=True)      # Descending
sorted_numbers = sorted(numbers) # Returns new sorted list

# Reversing
numbers.reverse()               # Reverse in-place
reversed_list = list(reversed(numbers))

# Counting
count = numbers.count(1)        # 2

# Finding index
index = numbers.index(5)        # First occurrence

# Copying (shallow copy)
copy1 = numbers.copy()
copy2 = numbers[:]              # Slicing creates copy`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">List comprehensions:</span> Concise way to create lists.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Basic comprehension
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# With if-else
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]

# Nested comprehension
matrix = [[j for j in range(3)] for i in range(3)]`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Ordered, immutable sequences. Tuples are defined with parentheses <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">()</code> and are faster than lists.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Creating and using tuples</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Creating tuples
coordinates = (10, 20)
person = ("Alice", 30, "New York")
single_item = (5,)              # Comma required for single-item tuple
empty_tuple = ()
not_a_tuple = (5)               # This is just an integer!

# Accessing elements (same as lists)
print(coordinates[0])   # 10
print(coordinates[-1])  # 20

# Tuple unpacking
x, y = coordinates      # x=10, y=20
name, age, city = person

# Immutability (cannot change)
# coordinates[0] = 15   # TypeError: 'tuple' object does not support item assignment

# Methods (limited)
print(coordinates.count(10))  # 1
print(coordinates.index(20))  # 1`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Tuple use cases:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# 1. Returning multiple values from functions
def get_min_max(numbers):
    return min(numbers), max(numbers)

min_val, max_val = get_min_max([1, 5, 3, 9, 2])

# 2. Dictionary keys (lists cannot be keys)
locations = { (40.7128, 74.0060): "NYC", (34.0522, 118.2437): "LA" }

# 3. Using as namedtuple for structured data
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Dictionaries
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Unordered collections of key-value pairs. Keys must be immutable (strings, numbers, tuples), values can be any type. Defined with curly braces <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{}</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Creating and accessing dictionaries</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Alternative constructors
dict1 = dict(name="Bob", age=25)
dict2 = dict([("a", 1), ("b", 2)])

# Accessing values
print(person["name"])       # Alice
print(person.get("age"))    # 30 (returns None if key missing)
print(person.get("country", "USA"))  # Default value

# Safe access (avoids KeyError)
if "name" in person:
    print(person["name"])`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Modifying dictionaries:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Adding/updating
person["email"] = "alice@example.com"  # Add new key
person["age"] = 31                     # Update existing

# Removing
del person["city"]                     # Delete by key
email = person.pop("email")            # Remove and return value
last_item = person.popitem()           # Remove and return last inserted item

# Merging dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
dict1.update(dict2)                    # dict1 becomes {"a": 1, "b": 3, "c": 4}

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Iterating through dictionaries:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`person = {"name": "Alice", "age": 30, "city": "NYC"}

# Iterate over keys
for key in person:
    print(key)

# Iterate over values
for value in person.values():
    print(value)

# Iterate over key-value pairs
for key, value in person.items():
    print(f"{key}: {value}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Dictionary methods:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Getting keys, values, items
keys = person.keys()       # dict_keys view
values = person.values()   # dict_values view
items = person.items()     # dict_items view

# Set defaults
person.setdefault("country", "USA")  # Sets only if key missing

# Clearing
person.clear()              # Empty the dictionary

# Copying
copy = person.copy()        # Shallow copy`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Sets */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Sets
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Unordered collections of unique elements. Sets are defined with curly braces <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{}</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">set()</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Creating and using sets</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Creating sets
fruits = {"apple", "banana", "cherry"}
numbers = set([1, 2, 3, 3, 2, 1])  # {1, 2, 3} (duplicates removed)
empty_set = set()                    # {} creates empty dict!

# Adding and removing
fruits.add("orange")                # Add element
fruits.remove("banana")              # Raises KeyError if not found
fruits.discard("grape")              # No error if not found
popped = fruits.pop()                # Remove and return arbitrary element
fruits.clear()                       # Empty the set

# Checking membership (very fast O(1))
if "apple" in fruits:
    print("Found!")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Set operations:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

# Union (all elements from both)
print(A | B)           # {1, 2, 3, 4, 5, 6}
print(A.union(B))      # Same

# Intersection (common elements)
print(A & B)           # {3, 4}
print(A.intersection(B))

# Difference (elements in A but not in B)
print(A - B)           # {1, 2}
print(A.difference(B))

# Symmetric difference (elements in either but not both)
print(A ^ B)           # {1, 2, 5, 6}
print(A.symmetric_difference(B))

# Subset and superset
print(A.issubset(B))   # False
print(A.issuperset({1, 2}))  # True
print(A.isdisjoint(B)) # False (they share elements)`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Set comprehensions:</span>
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Set comprehension
squares = {x**2 for x in range(10)}  # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# With condition
evens = {x for x in range(20) if x % 2 == 0}

# Removing duplicates from list
numbers = [1, 2, 2, 3, 3, 3, 4]
unique = set(numbers)  # {1, 2, 3, 4}`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Frozenset */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Frozenset
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Immutable version of a set. Can be used as dictionary keys or stored in other sets.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Creating frozensets
fs = frozenset([1, 2, 3, 3, 2])  # frozenset({1, 2, 3})

# Immutable (no add, remove, etc.)
# fs.add(4)  # AttributeError

# Can be used as dictionary keys
dict_with_frozenset = {frozenset([1, 2]): "value"}

# Set operations work (return new frozensets)
result = fs.union({4, 5})        # frozenset({1, 2, 3, 4, 5})
intersection = fs.intersection({2, 3, 4})  # frozenset({2, 3})`}
            </pre>
          </div>
        </section>

        {/* Collection Performance */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Performance Comparison
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Operation</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">List</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Set</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Dict</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">Index/Access</td>
                  <td className="p-3">O(1)</td>
                  <td className="p-3">-</td>
                  <td className="p-3">O(1)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">Search (in)</td>
                  <td className="p-3">O(n)</td>
                  <td className="p-3">O(1)</td>
                  <td className="p-3">O(1)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">Insert</td>
                  <td className="p-3">O(1)*</td>
                  <td className="p-3">O(1)</td>
                  <td className="p-3">O(1)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">Delete</td>
                  <td className="p-3">O(n)</td>
                  <td className="p-3">O(1)</td>
                  <td className="p-3">O(1)</td>
                </tr>
                <tr>
                  <td className="p-3">Memory per item</td>
                  <td className="p-3">Low</td>
                  <td className="p-3">High</td>
                  <td className="p-3">High</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">*Append is O(1), insert at arbitrary position is O(n)</p>
        </section>

        {/* Choosing the Right Collection */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Choosing the Right Collection
          </h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Use a List when:</p>
              <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1 list-disc list-inside">
                <li>Order matters</li>
                <li>You need indexing</li>
                <li>Duplicates are allowed</li>
                <li>You frequently modify the collection</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Use a Tuple when:</p>
              <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1 list-disc list-inside">
                <li>Order matters and shouldn't change</li>
                <li>The data is fixed (e.g., coordinates)</li>
                <li>You need a dictionary key</li>
                <li>Performance is critical (tuples are faster)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Use a Dict when:</p>
              <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1 list-disc list-inside">
                <li>You need fast key-based lookup</li>
                <li>Keys have meaningful names</li>
                <li>You need to associate values</li>
                <li>Order doesn't matter (Python 3.7+ preserves insertion order)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Use a Set when:</p>
              <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1 list-disc list-inside">
                <li>You need unique elements</li>
                <li>Order doesn't matter</li>
                <li>Fast membership testing is needed</li>
                <li>You perform set operations (union, intersection)</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Tricky Points */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Lists are mutable but tuple contents may be mutable</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                A tuple containing a list allows modifying the list: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t = (1, [2, 3]); t[1].append(4)</code> works.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Don't modify a list while iterating</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Modifying a list during iteration causes skipped elements. Iterate over a copy: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">for item in list[:]:</code>
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Default arguments and mutable objects</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Using mutable default arguments (lists, dicts) persists across function calls. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code> instead.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Shallow vs Deep copies</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">.copy()</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">[:]</code> create shallow copies (nested objects are referenced). Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">copy.deepcopy()</code> for nested structures.
              </p>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Interview Questions
          </h2>
          <div className="space-y-5">
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is the difference between a list and a tuple?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                Lists are mutable (can be changed), tuples are immutable. Tuples are faster and can be used as dictionary keys. Lists use more memory.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                How do you remove duplicates from a list?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list(set(my_list))</code> — convert to set then back to list. Note: order is not preserved. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict.fromkeys(my_list).keys()</code> to preserve order in Python 3.7+.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is the time complexity of checking if an item exists in a list vs a set?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                List: O(n) - needs to check each element. Set: O(1) average - uses hash table. Sets are much faster for membership testing.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                Can a list be a dictionary key? Why or why not?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                No, list is mutable and not hashable. Dictionary keys must be immutable (strings, numbers, tuples, frozensets).
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is the difference between <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list.remove()</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list.pop()</code>?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">remove()</code> removes the first occurrence of a value and returns nothing. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">pop()</code> removes an element at an index (default last) and returns it.
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/match-pattern"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Match Pattern
          </Link>
          <Link
            href="/learn/programming-fundamentals/string"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: → String
          </Link>
        </div>
      </div>
    </div>
  );
}