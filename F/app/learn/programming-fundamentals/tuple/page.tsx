// app/learn/programming-fundamentals/tuple/page.tsx

import Link from "next/link";

export default function TuplePage() {
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
        <div className="mb-8 border-b border-gray-200 dark:border-gray-700 pb-4">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Tuples
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Tuples are ordered, immutable sequences. They can hold mixed types and allow duplicate values.
          </p>
        </div>

        {/* Creating Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Creating Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            You can build a tuple in several ways:
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Approach</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Use when</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Literal `()`</td>
                  <td className="p-3 font-mono">row = (2, 4, 6)</td>
                  <td className="p-3">You know the elements</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Empty tuple</td>
                  <td className="p-3 font-mono">seq = ()</td>
                  <td className="p-3">Empty placeholder</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">tuple()</td>
                  <td className="p-3 font-mono">tuple() → ()</td>
                  <td className="p-3">Empty tuple (same as ())</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">tuple(iterable)</td>
                  <td className="p-3 font-mono">tuple([2, 4, 6]) → (2, 4, 6)</td>
                  <td className="p-3">Convert from list, range, str, set, etc.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Single-element tuple:</span> Requires a trailing comma: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2,)</code> - without the comma, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2)</code> is just an integer in parentheses.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10)
single = (2,)           # tuple
not_tuple = (2)         # int
built = tuple([2, 4, 6])  # (2, 4, 6)
empty = ()
from_list = tuple([2, 4, 6, 8])`}
            </pre>
          </div>

          <div className="mt-4 space-y-2">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <span className="font-semibold">Length:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">len(t)</code> to get the number of elements. For tuples this is O(1); the length is stored.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <span className="font-semibold">Truthiness:</span> An empty tuple <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">()</code> is falsy; a non-empty tuple is truthy. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if t:</code> to mean "if the tuple has at least one element" and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if not t:</code> for "if the tuple is empty."
              </p>
            </div>
          </div>
        </section>

        {/* Tuple Methods */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tuple Methods
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">count(x)</td>
                  <td className="p-3">Return the number of occurrences of x</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">index(x)</td>
                  <td className="p-3">Return the index of the first occurrence of x; raises ValueError if missing</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-gray-700 dark:text-gray-300 mt-4 text-sm">
            Tuples have only these two methods because they are immutable. No <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">append</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">remove</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sort</code>, or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reverse</code>.
          </p>
        </section>

        {/* Access Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Access Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Use <span className="font-semibold">indexing</span> and <span className="font-semibold">slicing</span> the same way as lists. Indices start at 0; negative indices count from the end.
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Pattern</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[i]</td>
                  <td className="p-3">Single element at index i</td>
                  <td className="p-3 font-mono">nums[2] → 6</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[-1]</td>
                  <td className="p-3">Last element</td>
                  <td className="p-3 font-mono">nums[-1] → 10</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[i:j]</td>
                  <td className="p-3">Slice from i to j (excl.)</td>
                  <td className="p-3 font-mono">nums[2:5] → (6, 8, 10)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[:n]</td>
                  <td className="p-3">First n elements</td>
                  <td className="p-3 font-mono">nums[:4] → (2, 4, 6, 8)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[n:]</td>
                  <td className="p-3">From index n to end</td>
                  <td className="p-3 font-mono">nums[4:] → (10,)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[-m:-n]</td>
                  <td className="p-3">Slice with negative indices</td>
                  <td className="p-3 font-mono">nums[-4:-2] → (4, 6)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Slicing with step:</span> The full form is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[start:stop:step]</code>. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">step</code> is the stride; omit it and it defaults to 1. Use a negative step to walk backward (e.g. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[::-1]</code> for a reversed copy).
            </p>
          </div>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Pattern</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Description</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[::k]</td>
                  <td className="p-3">Every k-th element</td>
                  <td className="p-3 font-mono">nums[::2] → (2, 6, 10, 14)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[i::k]</td>
                  <td className="p-3">Every k-th from index i</td>
                  <td className="p-3 font-mono">nums[1::2] → (4, 8, 12, 16)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">t[::-1]</td>
                  <td className="p-3">Reversed (step -1)</td>
                  <td className="p-3 font-mono">nums[::-1] → reversed copy</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10, 12, 14, 16)
nums[2]         # 6
nums[-1]        # 16
nums[2:6]       # (6, 8, 10, 12)
nums[:4]        # (2, 4, 6, 8)
nums[4:]        # (10, 12, 14, 16)
nums[-4:-2]     # (12, 14)
4 in nums       # True
nums.count(6)   # 1
nums.index(8)   # 3`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Value lookup:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">count(x)</code> to count occurrences and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index(x)</code> to get the first index. <span className="font-semibold">Note:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index(x)</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">ValueError</code> when the item is not present. Check with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x in t</code> first, or wrap in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">try</code>/<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">except</code>, if you need to handle the missing case.
            </p>
          </div>
        </section>

        {/* Reversing a Tuple */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Reversing a Tuple
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Create a new tuple with elements in reverse order. The original tuple is unchanged.
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Approach</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Returns</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Mutates original?</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">tuple(reversed(t))</td>
                  <td className="p-3 font-mono">tuple(reversed(nums))</td>
                  <td className="p-3">New tuple</td>
                  <td className="p-3">No</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Slice [::-1]</td>
                  <td className="p-3 font-mono">nums[::-1]</td>
                  <td className="p-3">New tuple</td>
                  <td className="p-3">No</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`original = (4, 2, 8, 6, 10)
# reversed() returns a reverse iterator; wrap in tuple() to get a tuple
nums = tuple(reversed(original))  # nums = (10, 6, 8, 2, 4); original = (4, 2, 8, 6, 10)

original = (4, 2, 8, 6, 10)
nums = original[::-1]  # nums = (10, 6, 8, 2, 4); original = (4, 2, 8, 6, 10)`}
            </pre>
          </div>

          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1 mt-4 text-sm">
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(reversed(t))</code> - <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed()</code> returns a reverse iterator; wrap in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple()</code> to get a tuple. O(n) time, O(n) space.</li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[::-1]</code> - Slice with step -1; creates a new tuple. O(n) time, O(n) space.</li>
          </ul>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              Tuples have no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reverse()</code> method (unlike lists) because they are immutable.
            </p>
          </div>
        </section>

        {/* Unpack Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Unpack Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Assign tuple elements to variables in one statement.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6)
a, b, c = nums  # a=2, b=4, c=6`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Asterisk `*`:</span> Collect remaining elements into a list.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10)
first, second, *rest = nums  # first=2, second=4, rest=[6, 8, 10]
print(type(rest))            # <class 'list'>`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Middle unpacking:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">*</code> at any position to capture the rest.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10)
a, *mid, z = nums  # a=2, mid=[4, 6, 8], z=10`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Mismatched count:</span> The number of variables must match the number of elements, unless <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">*</code> is used to absorb extras.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6)
a, b = nums        # ValueError: too many values to unpack
a, b, c, d = nums  # ValueError: not enough values to unpack

a, b, c, *d = (2, 4, 6)  # a=2, b=4, c=6, d=[]`}
            </pre>
          </div>

          <div className="mt-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Index with loop:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">enumerate(t)</code> when you need both index and value:
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`row = (2, 4, 6, 8)
for i, x in enumerate(row):
    print(i, x)  # 0 2, 1 4, 2 6, 3 8`}
              </pre>
            </div>
          </div>
        </section>

        {/* Update Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Update Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Tuples are <span className="font-semibold">immutable</span> - you cannot change, add, or remove elements in place. To "update" a tuple, create a new one.
          </p>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Replace an element:</span> Convert to list, modify, convert back.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10)
temp = list(nums)
temp[1] = 0
nums = tuple(temp)  # (2, 0, 6, 8, 10)`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Add an element:</span> Convert to list, append, convert back - or concatenate with a single-element tuple.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10)
temp = list(nums)
temp.append(12)
nums = tuple(temp)  # (2, 4, 6, 8, 10, 12)
print(nums)

nums = (2, 4, 6, 8, 10)
nums += (12,)  # (2, 4, 6, 8, 10, 12)`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Remove an element:</span> Convert to list, remove or filter, convert back.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10)
temp = list(nums)
temp.remove(6)
nums = tuple(temp)  # (2, 4, 8, 10)`}
              </pre>
            </div>
          </div>
        </section>

        {/* Join Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Join Tuples
          </h2>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Concatenation with `+`:</span> Both operands must be tuples. Returns a new tuple.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`a = (2, 4, 6)
b = (8, 10, 12)
c = a + b  # (2, 4, 6, 8, 10, 12)`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Repetition with `*`:</span> Repeat the tuple a given number of times.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6)
doubled = nums * 2  # (2, 4, 6, 2, 4, 6)`}
              </pre>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Note:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">+</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">*</code> create new tuples; they do not modify the original.
            </p>
          </div>
        </section>

        {/* Loop Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Loop Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Three common patterns: direct iteration, index-based, and while loop.
          </p>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Direct iteration:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)
for x in nums:
    print(x)  # 2, 4, 6, 8`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Index-based:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)
for i in range(len(nums)):
    print(nums[i])  # 2, 4, 6, 8`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">While loop:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)
i = 0
while i < len(nums):
    print(nums[i])
    i += 2  # 2, 6 - step by 2`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Loop in reverse:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)

for x in reversed(nums):  # O(n)
    print(x)              # 8, 6, 4, 2

for x in nums[::-1]:  # O(n)
    print(x)          # 8, 6, 4, 2 - slice with step -1

for i in range(len(nums) - 1, -1, -1):  # O(n)
    print(nums[i])                      # 8, 6, 4, 2 - index-based reverse`}
              </pre>
            </div>
          </div>

          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1 mt-4 text-sm">
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(t)</code> - Returns a reverse iterator; does not create a new tuple.</li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[::-1]</code> - Slice with step -1; creates a new tuple in reverse order.</li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">range(len(t)-1, -1, -1)</code> - Index-based; iterate from last index down to 0.</li>
          </ul>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Why LeetCode often prefers the index-based reverse:</span> When you need to <span className="font-semibold">modify</span> elements by index (swap, overwrite, remove), you must have the index. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed()</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[::-1]</code> give values, not indices. The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">range(len(t)-1, -1, -1)</code> pattern also avoids index shifting when removing elements in place, and it translates directly to other languages (C, Java, etc.).
            </p>
          </div>
        </section>

        {/* Sort Tuples */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Sort Tuples
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Tuples have no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sort()</code> method because they are immutable. To get a sorted tuple, create a new one using <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted()</code>.
          </p>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Note:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted(t)</code> returns a <span className="font-semibold">list</span>, not a tuple. Wrap in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple()</code> to get a tuple.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nums = (10, 4, 8, 2, 6)
tuple(sorted(nums))                # (2, 4, 6, 8, 10)
tuple(sorted(nums, reverse=True))  # (10, 8, 6, 4, 2)`}
            </pre>
          </div>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">With `key`:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">key</code> for custom sort order, same as for lists.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10, 12, 14, 16)
tuple(sorted(nums, key=lambda n: abs(n - 10)))  # sort by distance from 10
# (10, 8, 12, 6, 14, 4, 16, 2)`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Alternative:</span> Convert to list, sort in place, convert back. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted(t)</code> is simpler and preferred.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`nums = (10, 4, 8, 2, 6)
temp = list(nums)
temp.sort()
tuple(temp)  # (2, 4, 6, 8, 10)`}
              </pre>
            </div>
          </div>
        </section>

        {/* Tuple vs Other Collections */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tuple vs Other Collections
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Python has other built-in types for sequences and collections. Choosing the right one depends on order, mutability, and whether you need hashability.
          </p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Type</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Ordered?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Mutable?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Duplicates?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Hashable?</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Typical use</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">tuple</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Yes (if elements are)</td>
                  <td className="p-3">Fixed sequences; dict keys; return values</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">list</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Ordered sequences you change</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">set</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">No</td>
                  <td className="p-3">N/A</td>
                  <td className="p-3">Unique items, membership, set math</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">dict</td>
                  <td className="p-3">Yes (ins.)</td>
                  <td className="p-3">Yes (vals)</td>
                  <td className="p-3">No (keys)</td>
                  <td className="p-3">Keys only</td>
                  <td className="p-3">Key–value mapping</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Note:</span> A tuple is hashable only if every element is hashable (e.g. no lists or dicts inside). Lists and sets are not hashable.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden mt-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Hashable tuple (all elements hashable) - valid as dict key
key_ok = (2, 4)
d = {key_ok: "point"}   # {(2, 4): 'point'}

# Tuple with a list inside - not hashable
key_bad = (2, [4, 6])
# d2 = {key_bad: "x"}  # TypeError: unhashable type: 'list'

# List cannot be a dict key
# d3 = {[2, 4]: "x"}   # TypeError: unhashable type: 'list'`}
            </pre>
          </div>

          <div className="mt-4">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="font-semibold">When to use a tuple:</span> Fixed sequence (e.g. coordinates, record-like data), dict keys, set elements, or multiple return values from a function. Use a <span className="font-semibold">list</span> when you need to append, remove, or reorder. Use a <span className="font-semibold">set</span> when you care only about uniqueness or fast membership.
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-3 mt-4">
            <p className="text-sm text-green-800 dark:text-green-200">
              <span className="font-semibold">Common use cases:</span>
            </p>
            <ul className="list-disc list-inside text-sm text-green-800 dark:text-green-200 mt-2">
              <li><span className="font-semibold">Return values:</span> A function that "returns multiple values" actually returns a tuple. You can unpack it: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">def min_max(vals): return min(vals), max(vals)</code> → <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">lo, hi = min_max([2, 4, 6, 8])</code> gives <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">lo=2</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">hi=8</code>.</li>
              <li><span className="font-semibold">Dict keys and set elements:</span> Tuples are hashable when all elements are hashable, so they can be dict keys or set members. Lists cannot: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{(2, 4): "point"}'}</code> is valid; <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{[2, 4]: "point"}'}</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">TypeError: unhashable type: 'list'</code>.</li>
            </ul>
          </div>
        </section>

        {/* Tricky Points */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Creating - Single-element tuple</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2)</code> is an int; <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2,)</code> is a tuple. The comma is required so Python parses it as a tuple.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Access - <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index(x)</code> when missing</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t.index(x)</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">ValueError</code> if <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x</code> is not in the tuple. Check with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x in t</code> first, or use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">try</code>/<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">except</code>, or something like <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">next((i for i, v in enumerate(t) if v == x), None)</code> for a safe "index or None."</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Update - Immutability and nested mutables</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">The tuple itself cannot be changed, but if an element is mutable (e.g. a list), that object's contents can change. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t = (1, [2, 4])</code>; <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[1].append(6)</code> is allowed and makes <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t</code> equal to <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(1, [2, 4, 6])</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Join - <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t + other</code> with non-tuple</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Concatenation requires both sides to be tuples. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2, 4) + [6, 8]</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">TypeError</code>. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2, 4) + tuple([6, 8])</code> or convert the other iterable to a tuple first.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Sort - <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted(t)</code> returns a list</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted(t)</code> gives a list, not a tuple. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(sorted(t))</code> when you need a sorted tuple. Forgetting the wrapper is a common mistake when coming from other languages or when you assume <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted()</code> returns the same type.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Assignment is not copy</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">b = a</code> makes <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">b</code> refer to the same tuple as <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">a</code>. Since tuples are immutable, this rarely causes surprises, but if you need a new tuple from an existing one you can use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[:]</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(t)</code> - both produce a shallow copy (same elements, same tuple type).</p>
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
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you create a single-element tuple?</p>
              <p className="text-gray-700 dark:text-gray-300">Use a trailing comma: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2,)</code>. Without the comma, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(2)</code> is just an integer in parentheses.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Why does a tuple have only <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">count</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index</code>?</p>
              <p className="text-gray-700 dark:text-gray-300">Tuples are immutable, so they have no mutating methods. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">count</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index</code> are the only read-only operations needed; no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">append</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">remove</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sort</code>, or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reverse</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you reverse a tuple?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(reversed(t))</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">t[::-1]</code>. Both create a <span className="font-semibold">new</span> tuple in reverse order; the original is unchanged. Tuples have no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reverse()</code> method (unlike lists) because they are immutable.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">What does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">*</code> do in tuple unpacking?</p>
              <p className="text-gray-700 dark:text-gray-300">Collects remaining elements into a list: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">first, *rest = nums</code> → <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">rest = [4, 6, 8, 10]</code>. Use at any position: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">a, *mid, z = nums</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Can you "modify" a tuple? How?</p>
              <p className="text-gray-700 dark:text-gray-300">Tuples are immutable. To change, add, or remove elements, convert to list, modify, convert back: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(list(t))</code> or use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">+</code> for concatenation.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you sort a tuple?</p>
              <p className="text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(sorted(t))</code> - <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted()</code> returns a list, so wrap in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple()</code>. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">key</code> for custom order: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple(sorted(t, key=fn))</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Why use a tuple instead of a list?</p>
              <p className="text-gray-700 dark:text-gray-300">Immutability (safe as dict key, set element); slightly less memory; semantic "fixed structure." Tuples are hashable if all elements are hashable.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">When can a tuple be a dict key? Why can't a list?</p>
              <p className="text-gray-700 dark:text-gray-300">A tuple can be a dict key only if every element is <span className="font-semibold">hashable</span> (e.g. ints, strings, other tuples of hashable types). Then the tuple is hashable and can be used as a key. A list is mutable and unhashable, so <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{[2, 4]: "x"}'}</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">TypeError: unhashable type: 'list'</code>. Use a tuple when you need a sequence as a key: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{(2, 4): "point"}'}</code> works.</p>
              </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">What does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">return a, b</code> return from a function?</p>
              <p className="text-gray-700 dark:text-gray-300">A <span className="font-semibold">tuple</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(a, b)</code>. The comma makes it a tuple. Callers can unpack: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x, y = f()</code> or use as a single value: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">result = f()</code> → <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">result</code> is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">(a, b)</code>.</p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/set"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Set
          </Link>
          <Link
            href="/learn/programming-fundamentals/list"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Lists →
          </Link>
        </div>
      </div>
    </div>
  );
}