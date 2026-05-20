// app/learn/programming-fundamentals/list/page.tsx

import Link from "next/link";

export default function ListPage() {
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
            Lists
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Lists are ordered, mutable sequences that can hold mixed types. Lists show up everywhere in Python - and in interviews.
          </p>
        </div>

        {/* Creating Lists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Creating Lists
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
                  <td className="p-3 font-mono">Literal []</td>
                  <td className="p-3 font-mono">row = [2, 4, 6]</td>
                  <td className="p-3">You know the elements</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Empty list</td>
                  <td className="p-3 font-mono">seq = []</td>
                  <td className="p-3">Start empty, append or extend later</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">list()</td>
                  <td className="p-3 font-mono">list() → []</td>
                  <td className="p-3">Empty list (same as [])</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">list(iterable)</td>
                  <td className="p-3 font-mono">list(range(4)) → [0, 1, 2, 3]</td>
                  <td className="p-3">Convert from another iterable</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`row = [2, 4, 6, 8]
empty = []
from_range = list(range(0, 6, 2))   # [0, 2, 4]
from_string = list("ab")             # ['a', 'b']
 
len(row)    # 4
len(empty)  # 0
if row:     # True
if not empty:  # True`}
            </pre>
          </div>
        </section>

        {/* List Methods Reference */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            List Methods
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">count()</td><td className="p-3">Count occurrences of a value</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">index()</td><td className="p-3">Return index of first occurrence; raises if missing</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">append()</td><td className="p-3">Add element at the end</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">insert()</td><td className="p-3">Insert element at index</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">extend()</td><td className="p-3">Append all items from an iterable</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">remove()</td><td className="p-3">Remove first occurrence of value</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">pop()</td><td className="p-3">Remove and return element at index (default last)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">clear()</td><td className="p-3">Remove all elements</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">copy()</td><td className="p-3">Return a shallow copy</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">reverse()</td><td className="p-3">Reverse order in place</td></tr>
                <tr><td className="p-3 font-mono">sort()</td><td className="p-3">Sort in place</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Access List Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Access List Items
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Pattern</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst[i]</td><td className="p-3">Single element at index i</td><td className="p-3 font-mono">vals[2] → element at index 2</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst[-1]</td><td className="p-3">Last element</td><td className="p-3 font-mono">vals[-1] → last element</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst[i:j]</td><td className="p-3">Slice from i to j (excl.)</td><td className="p-3 font-mono">vals[2:6] → indices 2, 3, 4, 5</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst[:n]</td><td className="p-3">First n elements</td><td className="p-3 font-mono">vals[:4] → indices 0–3</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst[n:]</td><td className="p-3">From index n to end</td><td className="p-3 font-mono">vals[4:] → indices 4 onward</td></tr>
                <tr><td className="p-3 font-mono">lst[::-1]</td><td className="p-3">Reversed (step -1)</td><td className="p-3 font-mono">vals[::-1] → reversed copy</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`vals = [2, 4, 6, 8, 10, 12, 14, 16]
vals[2]         # 6
vals[-1]        # 16
vals[2:6]       # [6, 8, 10, 12]
vals[:4]        # [2, 4, 6, 8]
vals[4:]        # [10, 12, 14, 16]
vals[::2]       # [2, 6, 10, 14]
vals[::-1]      # [16, 14, 12, 10, 8, 6, 4, 2]

4 in vals       # True
vals.count(6)   # 1
vals.index(8)   # 3`}
            </pre>
          </div>
        </section>

        {/* Reversing a List */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Reversing a List
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Returns</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Mutates original?</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst[::-1]</td><td className="p-3">New list</td><td className="p-3">No</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">list(reversed(lst))</td><td className="p-3">New list</td><td className="p-3">No</td></tr>
                <tr><td className="p-3 font-mono">lst.reverse()</td><td className="p-3">None</td><td className="p-3">Yes (in place)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`row = [2, 4, 6, 8]
 
# New list; original unchanged
reversed_slice = row[::-1]           # [8, 6, 4, 2]
reversed_builtin = list(reversed(row))  # [8, 6, 4, 2]
print(row)                           # [2, 4, 6, 8] - unchanged
 
# In place; returns None
row.reverse()                        # None
print(row)                           # [8, 6, 4, 2] - mutated
 
# To keep the original and get a reversed copy:
original = [2, 4, 6, 8]
copy_then_reverse = original.copy()
copy_then_reverse.reverse()          # copy_then_reverse is [8, 6, 4, 2]; original still [2, 4, 6, 8]`}
            </pre>
          </div>
        </section>

        {/* Unpacking Lists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Unpacking Lists
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Pattern</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Result / note</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Same length</td><td className="p-3 font-mono">a, b, c = [2, 4, 6]</td><td className="p-3">a=2, b=4, c=6</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">*rest</td><td className="p-3 font-mono">first, *rest = [2, 4, 6, 8]</td><td className="p-3">first=2, rest=[4, 6, 8]</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">*mid</td><td className="p-3 font-mono">first, *mid, last = [2, 4, 6]</td><td className="p-3">first=2, mid=[4], last=6</td></tr>
                <tr><td className="p-3">Spread into list</td><td className="p-3 font-mono">[*row, 10, 12]</td><td className="p-3">New list: elements of row plus 10, 12</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`row = [2, 4, 6, 8]
a, b, c, d = row           # a=2, b=4, c=6, d=8
head, *tail = row          # head=2, tail=[4, 6, 8]
first, *mid, last = row    # first=2, mid=[4, 6], last=8
 
for i, x in enumerate(row):
    print(i, x)             # 0 2, 1 4, 2 6, 3 8
 
combined = [*row, 10, 12]  # [2, 4, 6, 8, 10, 12]`}
            </pre>
          </div>
        </section>

        {/* Add List Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Add List Items
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.append(x)</td><td className="p-3">Add x at the end</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.insert(i, x)</td><td className="p-3">Insert x at index i</td></tr>
                <tr><td className="p-3 font-mono">lst.extend(iter)</td><td className="p-3">Append all items from an iterable (list, tuple, set, etc.)</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">+ vs extend()</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Aspect</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">lst + other</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">lst.extend(other)</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Returns</td><td className="p-3">New list</td><td className="p-3">None (mutates)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Mutates original</td><td className="p-3">No</td><td className="p-3">Yes</td></tr>
                <tr><td className="p-3">Operand types</td><td className="p-3">Both must be lists</td><td className="p-3">other = any iterable</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`evens = [2, 4, 6]
evens.append(8)         # [2, 4, 6, 8]
evens.insert(1, 0)      # [2, 0, 4, 6, 8]
more = [10, 12]
evens.extend(more)      # [2, 0, 4, 6, 8, 10, 12]
evens.extend((14, 16))  # [2, 0, 4, 6, 8, 10, 12, 14, 16]`}
            </pre>
          </div>
        </section>

        {/* Remove List Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Remove List Items
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method / Keyword</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.remove(x)</td><td className="p-3">Remove the first occurrence of x; raises ValueError if missing</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.pop(i)</td><td className="p-3">Remove and return the element at index i; default -1 (last)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">del lst[i]</td><td className="p-3">Delete element at index i (or slice i:j)</td></tr>
                <tr><td className="p-3 font-mono">lst.clear()</td><td className="p-3">Remove all elements in place; list becomes []</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">remove() vs pop()</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Aspect</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">lst.remove(x)</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">lst.pop(i)</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Identifies by</td><td className="p-3">Value x</td><td className="p-3">Index i; default -1 (last)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Returns</td><td className="p-3">None</td><td className="p-3">The removed element</td></tr>
                <tr><td className="p-3">If missing</td><td className="p-3">ValueError</td><td className="p-3">IndexError</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = [2, 4, 6, 4, 8]
data.remove(4)     # [2, 6, 4, 8] - first 4 removed
data.pop(2)        # returns 4; data = [2, 6, 8]
data.pop()         # returns 8; data = [2, 6]
 
del data[0]        # data = [6]
 
# Remove all occurrences of a value
data = [2, 4, 6, 4, 8]
data = [x for x in data if x != 4]   # [2, 6, 8] - new list`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">lst = [...] vs lst[:] = [...]</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Aspect</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">lst = [...]</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">lst[:] = [...]</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">Effect</td><td className="p-3">Rebinds variable to new list</td><td className="p-3">Replaces contents of existing list in place</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3">List object</td><td className="p-3">New object; original unchanged</td><td className="p-3">Same object; elements changed</td></tr>
                <tr><td className="p-3">Other refs</td><td className="p-3">Still see old list</td><td className="p-3">See updated list</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`original = [1, 2, 2, 3]
ref = original
original = [x for x in original if x != 2]    # rebind
# original = [1, 3], ref = [1, 2, 2, 3] - ref still sees old list
 
original = [1, 2, 2, 3]
ref = original
original[:] = [x for x in original if x != 2]  # replace contents
# original = [1, 3], ref = [1, 3] - same object, both updated`}
            </pre>
          </div>
        </section>

        {/* Loop Lists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Loop Lists
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = [2, 4, 6, 8]
 
for x in nums:
    print(x)        # 2, 4, 6, 8
 
for i in range(len(nums)):
    print(nums[i])  # 2, 4, 6, 8
 
i = 0
while i < len(nums):
    print(nums[i])
    i += 2          # 2, 6 - step by 2

# Loop in reverse
for x in reversed(nums):  # O(n)
    print(x)              # 8, 6, 4, 2
 
for i in range(len(nums) - 1, -1, -1):  # O(n)
    print(nums[i])                      # 8, 6, 4, 2 - index-based reverse`}
            </pre>
          </div>
        </section>

        {/* Comprehension */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Comprehension
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Syntax: [expr for item in iterable if condition]

evens = [2, 4, 6, 8, 10]
doubled = [x * 2 for x in evens]                   # [4, 8, 12, 16, 20]
filtered = [x for x in evens if x > 4]             # [6, 8, 10]
from_range = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
squared = [x ** 2 for x in evens]                  # [4, 16, 36, 64, 100]`}
            </pre>
          </div>
        </section>

        {/* Copy Lists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Copy Lists
          </h2>
          
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Shallow Copy</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Approach</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.copy()</td><td className="p-3 font-mono">new = orig.copy()</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">list(lst)</td><td className="p-3 font-mono">new = list(orig)</td></tr>
                <tr><td className="p-3 font-mono">Slice [:]</td><td className="p-3 font-mono">new = orig[:]</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mb-4">
            <p className="font-semibold text-gray-900 dark:text-white">When to use shallow copy:</p>
            <p className="text-gray-700 dark:text-gray-300">The list is flat or contains only immutable elements (ints, strings, tuples). Shallow copy is faster and sufficient for most cases.</p>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Deep Copy</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`import copy
 
# Shallow copy - fine for flat lists
a = [2, 4, 6]
b = a.copy() # b = [2, 4, 6]
a.append(8)  # a = [2, 4, 6, 8]; b unchanged
 
# Shallow copy - nested lists are shared
original = [[1, 2], [3, 4]]
shallow = original.copy()
shallow[0].append(99)  # original[0] is also [1, 2, 99] - shared reference
 
# Deep copy - fully independent
deep = copy.deepcopy(original)
deep[0].append(100)  # original unchanged; deep[0] = [1, 2, 99, 100]`}
            </pre>
          </div>
        </section>

        {/* Sort Lists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Sort Lists
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method / Built-in</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.sort()</td><td className="p-3">Sort in place; returns None</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.sort(reverse=True)</td><td className="p-3">Sort descending in place</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">lst.sort(key=fn)</td><td className="p-3">Sort by key function (e.g. abs, str.lower)</td></tr>
                <tr><td className="p-3 font-mono">sorted(lst)</td><td className="p-3">Returns new sorted list; original unchanged</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = [22, 4, 16, 8, 10]
nums.sort()                 # [4, 8, 10, 16, 22]
nums.sort(reverse=True)     # [22, 16, 10, 8, 4]
 
labels = ["Beta", "alpha", "Gamma"]
labels.sort()               # ['Beta', 'Gamma', 'alpha']
labels.sort(key=str.lower)  # ["alpha", "Beta", "Gamma"]

# sort() vs sorted()
row = [6, 2, 8, 4]
row.sort()              # row is now [2, 4, 6, 8]; returns None
out = sorted(row)       # out = [2, 4, 6, 8]; row unchanged
from_tuple = sorted((4, 2, 8))  # [2, 4, 8] - input can be any iterable`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mt-4">
            <p className="font-semibold text-gray-900 dark:text-white">The key Parameter</p>
            <p className="text-gray-700 dark:text-gray-300">For each element x, Python computes key(x) and compares those values instead of the elements themselves.</p>
            <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
              {`nums = [22, 4, 16, 8, 10]
# Sort by distance from 12: abs(22-12)=10, abs(4-12)=8, abs(16-12)=4, abs(8-12)=4, abs(10-12)=2
nums.sort(key=lambda n: abs(n - 12))  # [10, 16, 8, 4, 22]`}
            </pre>
          </div>
        </section>

        {/* List vs Other Collections */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            List vs Other Collections
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Type</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Ordered?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Mutable?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Duplicates?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Typical use</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">list</td><td className="p-3">Yes</td><td className="p-3">Yes</td><td className="p-3">Yes</td><td className="p-3">Ordered sequences you change</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">tuple</td><td className="p-3">Yes</td><td className="p-3">No</td><td className="p-3">Yes</td><td className="p-3">Fixed sequences (dict keys)</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">set</td><td className="p-3">No</td><td className="p-3">Yes</td><td className="p-3">No</td><td className="p-3">Unique items, membership tests</td></tr>
                <tr><td className="p-3 font-mono">dict</td><td className="p-3">Yes (insertion)</td><td className="p-3">Yes (values)</td><td className="p-3">No (keys)</td><td className="p-3">Key–value mapping</td></tr>
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
              <p className="font-semibold text-gray-900 dark:text-white">Access - index(x) when missing</p>
              <p className="text-gray-700 dark:text-gray-300">lst.index(x) raises ValueError if x is not in the list. Check with x in lst first, or wrap in try/except.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Add - lst + other with non-list</p>
              <p className="text-gray-700 dark:text-gray-300">Concatenation requires both sides to be lists. Use lst.extend((1, 2)) or lst + list((1, 2)).</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Loop - Modifying while iterating</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# WRONG - skip elements
lst = [1, 2, 3, 4]
for i, item in enumerate(lst):
    if item % 2 == 0:
        del lst[i]

# CORRECT - iterate over copy
for item in lst[:]:
    if item % 2 == 0:
        lst.remove(item)

# CORRECT - list comprehension
lst = [item for item in lst if item % 2 != 0]`}
              </pre>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Mutable default argument</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# WRONG - list persists across calls
def add_item(item, lst=[]):
    lst.append(item)
    return lst

# CORRECT
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst`}
              </pre>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Interview Questions
          </h2>
          <div className="space-y-5">
            <div><p className="font-semibold text-gray-900 dark:text-white">What is the time complexity of list append, insert, and index lookup?</p><p className="text-gray-700 dark:text-gray-300">Append O(1) amortized; insert O(n); index lookup O(1); in O(n). List is a dynamic array.</p></div>
            <div><p className="font-semibold text-gray-900 dark:text-white">How do you safely get the index of an item when it might not be in the list?</p><p className="text-gray-700 dark:text-gray-300">Check with x in lst first, then lst.index(x). Or use try/except ValueError.</p></div>
            <div><p className="font-semibold text-gray-900 dark:text-white">How do you unpack "first" and "rest" of a list?</p><p className="text-gray-700 dark:text-gray-300">Use first, *rest = lst.</p></div>
            <div><p className="font-semibold text-gray-900 dark:text-white">When to use + vs extend()?</p><p className="text-gray-700 dark:text-gray-300">Use + when you want a new list; both operands must be lists. Use extend() when you want to grow the list in place.</p></div>
            <div><p className="font-semibold text-gray-900 dark:text-white">What is the difference between lst.clear() and del lst?</p><p className="text-gray-700 dark:text-gray-300">lst.clear() empties the list in place; the list object still exists. del lst removes the name lst; the list may be garbage-collected.</p></div>
            <div><p className="font-semibold text-gray-900 dark:text-white">What is the difference between lst[::-1], list(reversed(lst)), and lst.reverse()?</p><p className="text-gray-700 dark:text-gray-300">lst[::-1] and list(reversed(lst)) produce a new list in reverse order; original unchanged. lst.reverse() reverses in place and returns None.</p></div>
            <div><p className="font-semibold text-gray-900 dark:text-white">sort() vs sorted() - in place vs new list?</p><p className="text-gray-700 dark:text-gray-300">sort() mutates the list in place and returns None. sorted() returns a new list and leaves the original unchanged.</p></div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/string"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: String
          </Link>
          <Link
            href="/learn/programming-fundamentals/dict"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Dictionary →
          </Link>
        </div>
      </div>
    </div>
  );
}