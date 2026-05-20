// app/learn/programming-fundamentals/set/page.tsx

import Link from "next/link";

export default function SetPage() {
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
            Sets
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            A set is an unordered, mutable collection of unique elements. Sets are optimized for membership testing and mathematical operations like union, intersection, and difference.
          </p>
        </div>

        {/* Creating Sets */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Creating Sets
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Approach</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Use when</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Literal {`{}`}</td>
                  <td className="p-3 font-mono">s = {`{1, 2, 3}`}</td>
                  <td className="p-3">You know the elements</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">set()</td>
                  <td className="p-3 font-mono">s = set()</td>
                  <td className="p-3">Empty set (can't use {`{}`} - that's a dict)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">set(iterable)</td>
                  <td className="p-3 font-mono">set([1, 2, 2, 3]) → {`{1, 2, 3}`}</td>
                  <td className="p-3">Remove duplicates from another collection</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Empty set (note: {} creates empty dict)
empty_set = set()

# Set with values
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 4, 5}

# From other iterables (automatically removes duplicates)
from_list = set([1, 2, 2, 3, 3, 3])   # {1, 2, 3}
from_string = set("hello")             # {'h', 'e', 'l', 'o'}
from_range = set(range(5))             # {0, 1, 2, 3, 4}

# Set comprehension
squares = {x**2 for x in range(10)}    # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
evens = {x for x in range(10) if x % 2 == 0}  # {0, 2, 4, 6, 8}`}
            </pre>
          </div>
        </section>

        {/* Set Properties */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Set Properties
          </h2>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mb-4">
            <p className="font-semibold text-gray-900 dark:text-white">Key Characteristics:</p>
            <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1 ml-4 list-disc mt-2">
              <li><strong>Unordered:</strong> Elements have no index; you cannot access by position</li>
              <li><strong>Unique:</strong> Duplicate elements are automatically removed</li>
              <li><strong>Mutable:</strong> You can add and remove elements</li>
              <li><strong>Hashable elements only:</strong> Elements must be immutable (no lists or dicts in sets)</li>
              <li><strong>Fast membership testing:</strong> O(1) average case for "in" operator</li>
            </ul>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Sets are unordered - order may vary
s = {3, 1, 4, 1, 5, 9}
print(s)  # {1, 3, 4, 5, 9} - duplicates removed, order not guaranteed

# Elements must be hashable (immutable)
s = {1, "hello", (1, 2)}     # OK - int, str, tuple
# s = {1, [1, 2]}             # TypeError: unhashable type: 'list'

# Length
len(s)                        # Number of unique elements

# Membership (O(1) average)
if 3 in s:
    print("Found")

# Truthiness: empty set is falsy
if s:                         # True if set has elements
    print("Set has elements")`}
            </pre>
          </div>
        </section>

        {/* Set Methods */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Set Methods
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">add(x)</td><td className="p-3">Adds element x to the set</td><td className="p-3 font-mono">s.add(5)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">remove(x)</td><td className="p-3">Removes x; raises KeyError if missing</td><td className="p-3 font-mono">s.remove(5)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">discard(x)</td><td className="p-3">Removes x if present; no error if missing</td><td className="p-3 font-mono">s.discard(5)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">pop()</td><td className="p-3">Removes and returns an arbitrary element</td><td className="p-3 font-mono">s.pop()</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">clear()</td><td className="p-3">Removes all elements</td><td className="p-3 font-mono">s.clear()</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">copy()</td><td className="p-3">Returns a shallow copy</td><td className="p-3 font-mono">s.copy()</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">union() / |</td><td className="p-3">All elements from both sets</td><td className="p-3 font-mono">s1 | s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">intersection() / &amp;</td><td className="p-3">Elements common to both sets</td><td className="p-3 font-mono">s1 &amp; s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">difference() / -</td><td className="p-3">Elements in s1 but not in s2</td><td className="p-3 font-mono">s1 - s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">symmetric_difference() / ^</td><td className="p-3">Elements in either but not both</td><td className="p-3 font-mono">s1 ^ s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">issubset() / &lt;=</td><td className="p-3">All elements of s1 are in s2</td><td className="p-3 font-mono">s1 &lt;= s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">issuperset() / &gt;=</td><td className="p-3">s1 contains all elements of s2</td><td className="p-3 font-mono">s1 &gt;= s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">isdisjoint()</td><td className="p-3">No common elements</td><td className="p-3 font-mono">s1.isdisjoint(s2)</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Adding and Removing Elements */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Adding and Removing Elements
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`s = {1, 2, 3}

# Add single element
s.add(4)                    # {1, 2, 3, 4}
s.add(2)                    # {1, 2, 3, 4} - no duplicate

# Add multiple elements (update)
s.update([5, 6, 7])         # {1, 2, 3, 4, 5, 6, 7}
s.update({8, 9}, [10, 11])  # Any number of iterables

# Remove with error if missing
s.remove(10)                # Removes 10
# s.remove(99)              # KeyError: 99

# Remove without error if missing
s.discard(99)               # No error, set unchanged

# Remove and return arbitrary element
popped = s.pop()            # Removes and returns some element
print(popped)               # Could be any element

# Clear all elements
s.clear()                   # set()`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mt-4">
            <p className="font-semibold text-gray-900 dark:text-white">remove() vs discard() vs pop()</p>
            <table className="min-w-full mt-2">
              <thead><tr className="text-gray-700 dark:text-gray-300"><th className="text-left p-2">Method</th><th className="text-left p-2">When to use</th></tr></thead>
              <tbody className="text-sm">
                <tr><td className="p-2 font-mono">remove(x)</td><td className="p-2">You know x exists; want error if it doesn't</td></tr>
                <tr><td className="p-2 font-mono">discard(x)</td><td className="p-2">You're not sure if x exists; don't want error</td></tr>
                <tr><td className="p-2 font-mono">pop()</td><td className="p-2">Need to remove and process an arbitrary element (like a stack/queue alternative)</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Mathematical Set Operations */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Mathematical Set Operations
          </h2>
          
          <div className="grid gap-6 md:grid-cols-2 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">Union (|)</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`{1,2,3} | {3,4,5}
# {1,2,3,4,5}`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">Intersection (&amp;)</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`{1,2,3} & {3,4,5}
# {3}`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">Difference (-)</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`{1,2,3} - {3,4,5}
# {1,2}`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">Symmetric Difference (^)</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`{1,2,3} ^ {3,4,5}
# {1,2,4,5}`}
              </pre>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# Union - all elements from both
print(a | b)           # {1, 2, 3, 4, 5, 6}
print(a.union(b))      # Same

# Intersection - common elements
print(a & b)           # {3, 4}
print(a.intersection(b))  # Same

# Difference - in a but not in b
print(a - b)           # {1, 2}
print(a.difference(b)) # Same

# Symmetric difference - in either but not both
print(a ^ b)           # {1, 2, 5, 6}
print(a.symmetric_difference(b))  # Same

# Subset and superset
c = {1, 2}
print(c <= a)          # True - c is subset of a
print(c.issubset(a))   # True
print(a >= c)          # True - a is superset of c
print(a.issuperset(c)) # True

# Disjoint - no common elements
d = {7, 8}
print(a.isdisjoint(d)) # True
print(a.isdisjoint(b)) # False`}
            </pre>
          </div>
        </section>

        {/* In-place vs New Set Operations */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            In-place vs New Set Operations
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Operation (new set)</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">In-place (mutates)</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Effect</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s1 | s2</td><td className="p-3 font-mono">s1 |= s2</td><td className="p-3">s1.update(s2) - adds all from s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s1 &amp; s2</td><td className="p-3 font-mono">s1 &amp;= s2</td><td className="p-3">s1.intersection_update(s2) - keeps only common</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s1 - s2</td><td className="p-3 font-mono">s1 -= s2</td><td className="p-3">s1.difference_update(s2) - removes elements in s2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s1 ^ s2</td><td className="p-3 font-mono">s1 ^= s2</td><td className="p-3">s1.symmetric_difference_update(s2)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`a = {1, 2, 3}
b = {2, 3, 4}

# New set (original a unchanged)
c = a | b          # c = {1, 2, 3, 4}; a = {1, 2, 3}

# In-place (a is mutated)
a |= b             # a becomes {1, 2, 3, 4}; b unchanged

# Similar for other operations
a = {1, 2, 3, 4}
a &= {3, 4, 5}     # a becomes {3, 4}`}
            </pre>
          </div>
        </section>

        {/* Set Comprehensions */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Set Comprehensions
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Basic: {expression for item in iterable}
squares = {x**2 for x in range(10)}
# {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# With condition
evens = {x for x in range(20) if x % 2 == 0}
# {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}

# From another collection (remove duplicates)
nums = [1, 2, 2, 3, 3, 3, 4]
unique = {x for x in nums}    # {1, 2, 3, 4}

# Character set from string
chars = {c for c in "hello world" if c != ' '}
# {'h', 'e', 'l', 'o', 'w', 'r', 'd'}

# Nested loops
pairs = {(x, y) for x in range(3) for y in range(3)}
# {(0, 1), (1, 2), (0, 0), (2, 0), (1, 0), (2, 2), (1, 1), (0, 2), (2, 1)}`}
            </pre>
          </div>
        </section>

        {/* Common Use Cases */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Common Use Cases
          </h2>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">1. Removing duplicates from a list</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`numbers = [1, 2, 2, 3, 3, 3, 4, 4, 5]
unique_numbers = list(set(numbers))  # [1, 2, 3, 4, 5]
# Note: order is not preserved`}
              </pre>
              <p className="text-gray-600 dark:text-gray-400 text-sm mt-1">To preserve order: use dict.fromkeys()</p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">2. Fast membership testing</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`valid_ids = {101, 102, 103, 104, 105}
if user_id in valid_ids:    # O(1) average vs O(n) for list
    print("Valid user")`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">3. Finding common elements between collections</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
common = set(list1) & set(list2)  # {4, 5}`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">4. Tracking seen items while iterating</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`seen = set()
for item in items:
    if item in seen:
        print(f"Duplicate: {item}")
    else:
        seen.add(item)`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">5. Finding unique characters in a string</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`text = "abracadabra"
unique_chars = set(text)  # {'a', 'b', 'c', 'd', 'r'}`}
              </pre>
            </div>
          </div>
        </section>

        {/* Loop Sets */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Looping Through Sets
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`fruits = {"apple", "banana", "cherry"}

# Basic iteration (order not guaranteed)
for fruit in fruits:
    print(fruit)

# With index (convert to list first - order not guaranteed)
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Sorted iteration
for fruit in sorted(fruits):
    print(fruit)  # apple, banana, cherry

# Check if all elements satisfy condition
all(x > 0 for x in numbers)

# Check if any element satisfies condition
any(x > 10 for x in numbers)`}
            </pre>
          </div>
        </section>

        {/* Frozenset */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Frozenset (Immutable Set)
          </h2>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mb-4">
            <p className="font-semibold text-gray-900 dark:text-white">What is a frozenset?</p>
            <p className="text-gray-700 dark:text-gray-300">A frozenset is an immutable version of a set. Once created, you cannot add or remove elements. Frozensets are hashable and can be used as dictionary keys or elements of another set.</p>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Creating frozensets
fs = frozenset([1, 2, 3])
fs2 = frozenset({3, 4, 5})

# Immutable - cannot modify
# fs.add(4)           # AttributeError
# fs.remove(1)        # AttributeError

# But set operations work (return new frozensets)
print(fs | fs2)       # frozenset({1, 2, 3, 4, 5})
print(fs & fs2)       # frozenset({3})

# Can be used as dictionary keys
dict_with_frozenset = {frozenset([1, 2]): "value"}

# Can be elements in a set (unlike regular sets)
set_of_sets = {frozenset([1, 2]), frozenset([3, 4])}
# set_of_sets = {{1, 2}, {3, 4}}  # TypeError - regular sets are unhashable`}
            </pre>
          </div>
        </section>

        {/* Performance Comparison */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Performance Comparison: Set vs List
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Operation</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Set</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">List</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Membership (in)</td><td className="p-3 text-green-600 font-semibold">O(1) average</td><td className="p-3 text-red-600">O(n)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Insertion</td><td className="p-3 text-green-600 font-semibold">O(1) average</td><td className="p-3 text-green-600 font-semibold">O(1) amortized</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Deletion</td><td className="p-3 text-green-600 font-semibold">O(1) average</td><td className="p-3 text-red-600">O(n)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Ordered</td><td className="p-3 text-red-600">No</td><td className="p-3 text-green-600 font-semibold">Yes</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Duplicates</td><td className="p-3 text-green-600 font-semibold">No</td><td className="p-3 text-red-600">Yes</td></tr>
                <tr><td className="p-3">Index access</td><td className="p-3 text-red-600">Not supported</td><td className="p-3 text-green-600 font-semibold">O(1)</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Tricky Points */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Empty set vs empty dict</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`s = {}        # This is an EMPTY DICT, not a set!
s = set()     # This is an empty set

# To create a set with one element, don't use {}
s = {1}       # Set with one element (OK)
# s = {}      # Empty dict, not set`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Modifying set while iterating</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# WRONG - RuntimeError
s = {1, 2, 3, 4}
for x in s:
    if x % 2 == 0:
        s.remove(x)    # RuntimeError: Set changed size during iteration

# CORRECT - iterate over copy
for x in list(s):      # or s.copy()
    if x % 2 == 0:
        s.remove(x)    # s now {1, 3}`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Unhashable types cannot be in sets</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# Hashable (immutable): int, float, str, tuple, frozenset
s = {1, 2.5, "hello", (1, 2), frozenset([3, 4])}  # OK

# Unhashable (mutable): list, dict, set
# s = {[1, 2]}           # TypeError: unhashable type: 'list'
# s = {{1, 2}}           # TypeError: unhashable type: 'set'`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Sets are unordered</p>
              <p className="text-gray-700 dark:text-gray-300">Don't rely on any specific order. Even though CPython 3.7+ preserves insertion order as an implementation detail, this is not guaranteed by the language spec. Use sorted() if you need ordered output.</p>
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
              <p className="font-semibold text-gray-900 dark:text-white">How do you remove duplicates from a list while preserving order?</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# Using dict (preserves insertion order in Python 3.7+)
unique = list(dict.fromkeys([1, 2, 2, 3, 3, 4]))
# [1, 2, 3, 4]

# Using set with manual order tracking
seen = set()
unique = []
for x in original:
    if x not in seen:
        seen.add(x)
        unique.append(x)`}
              </pre>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">What is the time complexity of 'in' for sets vs lists?</p>
              <p className="text-gray-700 dark:text-gray-300">Sets: O(1) average (hash table). Lists: O(n) (linear search). This is why sets are preferred for membership testing with large collections.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">Why can't you put a list in a set?</p>
              <p className="text-gray-700 dark:text-gray-300">Sets require elements to be hashable (immutable). Lists are mutable and therefore unhashable. Use a tuple instead: {JSON.stringify([1, 2, 3])}.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">What is the difference between set.remove() and set.discard()?</p>
              <p className="text-gray-700 dark:text-gray-300">remove() raises KeyError if the element doesn't exist. discard() does nothing and doesn't raise an error. Use remove() when you expect the element to be present; use discard() when you're not sure.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">How do you find common elements between two lists efficiently?</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# Convert to sets for O(n) instead of O(n²)
common = list(set(list1) & set(list2))

# Or using intersection method
common = list(set(list1).intersection(list2))`}
              </pre>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">What is a frozenset and when would you use it?</p>
              <p className="text-gray-700 dark:text-gray-300">A frozenset is an immutable set. Use it when you need a set that can be hashed - as a dictionary key or as an element in another set. Also use it when you want to ensure the set contents don't change.</p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/dict"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Dictionary
          </Link>
          <Link
            href="/learn/programming-fundamentals/functions"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Function →
          </Link>
        </div>
      </div>
    </div>
  );
}