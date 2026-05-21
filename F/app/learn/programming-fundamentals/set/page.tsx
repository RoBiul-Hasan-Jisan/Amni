import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function SetPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "set");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Sets
          </h1>
          <p className="text-muted-foreground text-lg">
            A set is an unordered, mutable collection of unique elements. Sets are optimized for membership testing and mathematical operations like union, intersection, and difference.
          </p>
        </div>

        {/* Creating Sets */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Creating Sets
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Literal {`{}`}</td>
                  <td className="p-3 font-mono">s = {`{1, 2, 3}`}</td>
                  <td className="p-3">You know the elements</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">set()</td>
                  <td className="p-3 font-mono">s = set()</td>
                  <td className="p-3">Empty set (can't use {`{}`} - that's a dict)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">set(iterable)</td>
                  <td className="p-3 font-mono">set([1, 2, 2, 3]) → {`{1, 2, 3}`}</td>
                  <td className="p-3">Remove duplicates from another collection</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Set Properties
          </h2>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground mb-1">Key Characteristics:</p>
                <ul className="text-muted-foreground text-sm space-y-1 ml-4 list-disc">
                  <li><strong className="text-foreground">Unordered:</strong> Elements have no index; you cannot access by position</li>
                  <li><strong className="text-foreground">Unique:</strong> Duplicate elements are automatically removed</li>
                  <li><strong className="text-foreground">Mutable:</strong> You can add and remove elements</li>
                  <li><strong className="text-foreground">Hashable elements only:</strong> Elements must be immutable (no lists or dicts in sets)</li>
                  <li><strong className="text-foreground">Fast membership testing:</strong> O(1) average case for "in" operator</li>
                </ul>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Set Methods
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">add(x)</td><td className="p-3">Adds element x to the set</td><td className="p-3 font-mono">s.add(5)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">remove(x)</td><td className="p-3">Removes x; raises KeyError if missing</td><td className="p-3 font-mono">s.remove(5)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">discard(x)</td><td className="p-3">Removes x if present; no error if missing</td><td className="p-3 font-mono">s.discard(5)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">pop()</td><td className="p-3">Removes and returns an arbitrary element</td><td className="p-3 font-mono">s.pop()</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">clear()</td><td className="p-3">Removes all elements</td><td className="p-3 font-mono">s.clear()</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">copy()</td><td className="p-3">Returns a shallow copy</td><td className="p-3 font-mono">s.copy()</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">union() / |</td><td className="p-3">All elements from both sets</td><td className="p-3 font-mono">s1 | s2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">intersection() / &amp;</td><td className="p-3">Elements common to both sets</td><td className="p-3 font-mono">s1 &amp; s2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">difference() / -</td><td className="p-3">Elements in s1 but not in s2</td><td className="p-3 font-mono">s1 - s2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">symmetric_difference() / ^</td><td className="p-3">Elements in either but not both</td><td className="p-3 font-mono">s1 ^ s2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">issubset() / &lt;=</td><td className="p-3">All elements of s1 are in s2</td><td className="p-3 font-mono">s1 &lt;= s2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">issuperset() / &gt;=</td><td className="p-3">s1 contains all elements of s2</td><td className="p-3 font-mono">s1 &gt;= s2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">isdisjoint()</td><td className="p-3">No common elements</td><td className="p-3 font-mono">s1.isdisjoint(s2)</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Adding and Removing Elements */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Adding and Removing Elements
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground mb-1">remove() vs discard() vs pop()</p>
                <table className="min-w-full mt-2">
                  <thead>
                    <tr className="text-muted-foreground">
                      <th className="text-left p-2">Method</th>
                      <th className="text-left p-2">When to use</th>
                    </tr>
                  </thead>
                  <tbody className="text-sm text-muted-foreground">
                    <tr><td className="p-2 font-mono">remove(x)</td><td className="p-2">You know x exists; want error if it doesn't</td></tr>
                    <tr><td className="p-2 font-mono">discard(x)</td><td className="p-2">You're not sure if x exists; don't want error</td></tr>
                    <tr><td className="p-2 font-mono">pop()</td><td className="p-2">Need to remove and process an arbitrary element (like a stack/queue alternative)</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* Mathematical Set Operations */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Mathematical Set Operations
          </h2>
          
          <div className="grid gap-6 md:grid-cols-2 mb-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Union (|)</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`{1,2,3} | {3,4,5}
# {1,2,3,4,5}`}
              </pre>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Intersection (&amp;)</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`{1,2,3} & {3,4,5}
# {3}`}
              </pre>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Difference (-)</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`{1,2,3} - {3,4,5}
# {1,2}`}
              </pre>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Symmetric Difference (^)</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`{1,2,3} ^ {3,4,5}
# {1,2,4,5}`}
              </pre>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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

        {/* Set Comprehensions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Set Comprehensions
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Use Cases
          </h2>
          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">1. Removing duplicates from a list</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`numbers = [1, 2, 2, 3, 3, 3, 4, 4, 5]
unique_numbers = list(set(numbers))  # [1, 2, 3, 4, 5]
# Note: order is not preserved`}
              </pre>
              <p className="text-muted-foreground text-sm mt-1">To preserve order: use dict.fromkeys()</p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">2. Fast membership testing</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`valid_ids = {101, 102, 103, 104, 105}
if user_id in valid_ids:    # O(1) average vs O(n) for list
    print("Valid user")`}
              </pre>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">3. Finding common elements between collections</p>
              <pre className="bg-muted/50 p-2 rounded text-primary text-sm overflow-x-auto">
                {`list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
common = set(list1) & set(list2)  # {4, 5}`}
              </pre>
            </div>
          </div>
        </section>

        {/* Frozenset */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Frozenset (Immutable Set)
          </h2>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground mb-1">What is a frozenset?</p>
                <p className="text-muted-foreground">A frozenset is an immutable version of a set. Once created, you cannot add or remove elements. Frozensets are hashable and can be used as dictionary keys or elements of another set.</p>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Performance Comparison: Set vs List
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Operation</th>
                  <th className="text-left p-3 font-semibold text-foreground">Set</th>
                  <th className="text-left p-3 font-semibold text-foreground">List</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Membership (in)</td><td className="p-3 text-primary font-semibold">O(1) average</td><td className="p-3 text-destructive">O(n)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Insertion</td><td className="p-3 text-primary font-semibold">O(1) average</td><td className="p-3 text-primary font-semibold">O(1) amortized</td></tr>
                <tr className="border-b border-border"><td className="p-3">Deletion</td><td className="p-3 text-primary font-semibold">O(1) average</td><td className="p-3 text-destructive">O(n)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Ordered</td><td className="p-3 text-destructive">No</td><td className="p-3 text-primary font-semibold">Yes</td></tr>
                <tr className="border-b border-border"><td className="p-3">Duplicates</td><td className="p-3 text-primary font-semibold">No</td><td className="p-3 text-destructive">Yes</td></tr>
                <tr><td className="p-3">Index access</td><td className="p-3 text-destructive">Not supported</td><td className="p-3 text-primary font-semibold">O(1)</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Tricky Points */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Empty set vs empty dict</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`s = {}        # This is an EMPTY DICT, not a set!
s = set()     # This is an empty set

# To create a set with one element, don't use {}
s = {1}       # Set with one element (OK)
# s = {}      # Empty dict, not set`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Modifying set while iterating</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
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
              </div>
            </div>

            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Unhashable types cannot be in sets</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# Hashable (immutable): int, float, str, tuple, frozenset
s = {1, 2.5, "hello", (1, 2), frozenset([3, 4])}  # OK

# Unhashable (mutable): list, dict, set
# s = {[1, 2]}           # TypeError: unhashable type: 'list'
# s = {{1, 2}}           # TypeError: unhashable type: 'set'`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Sets are unordered</p>
                  <p className="text-muted-foreground">Don't rely on any specific order. Even though CPython 3.7+ preserves insertion order as an implementation detail, this is not guaranteed by the language spec. Use sorted() if you need ordered output.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Interview Questions
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">How do you remove duplicates from a list while preserving order?</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
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
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What is the time complexity of 'in' for sets vs lists?</p>
                  <p className="text-muted-foreground">Sets: O(1) average (hash table). Lists: O(n) (linear search). This is why sets are preferred for membership testing with large collections.</p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What is the difference between set.remove() and set.discard()?</p>
                  <p className="text-muted-foreground">remove() raises KeyError if the element doesn't exist. discard() does nothing and doesn't raise an error. Use remove() when you expect the element to be present; use discard() when you're not sure.</p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What is a frozenset and when would you use it?</p>
                  <p className="text-muted-foreground">A frozenset is an immutable set. Use it when you need a set that can be hashed - as a dictionary key or as an element in another set. Also use it when you want to ensure the set contents don't change.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

      
      </div>
    </TopicContent>
  );
}