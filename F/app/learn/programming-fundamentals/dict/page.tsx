// app/learn/programming-fundamentals/dict/page.tsx

import Link from "next/link";

export default function DictPage() {
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
        {/* Header with metadata */}
        <div className="mb-8 border-b border-gray-200 dark:border-gray-700 pb-4">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Dictionaries
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Dictionaries store key–value pairs. Keys must be hashable; values can be any type. As of Python 3.7, dictionaries preserve insertion order.
          </p>
        </div>

        {/* Creating Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Creating Dictionaries
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Use curly braces <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{`{}`}</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">dict()</code>. Empty dictionary: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{`{}`}</code>.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
built = dict(a=2, b=4, c=6)  # keyword args; keys become strings
built = dict([("a", 2), ("b", 4), ("c", 6)])  # from list of pairs
empty = {}`}
            </pre>
          </div>

          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <span className="font-semibold">dict() keyword args:</span> Keys must be valid identifiers (no spaces, no quotes). <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict(2=4)</code> is invalid; use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{"2": 4}'}</code>.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <span className="font-semibold">Length:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">len(d)</code> for the number of key–value pairs.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <span className="font-semibold">Type:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">type(data)</code> → <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'<class "dict">'}</code>.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <span className="font-semibold">Truthiness:</span> An empty dictionary <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{`{}`}</code> is falsy; a non-empty dictionary is truthy. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if d:</code> for "at least one pair" and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if not d:</code> for "no keys".
              </p>
            </div>
          </div>

          <div className="mt-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Build from two sequences:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">zip()</code> to pair keys and values, then pass the result to <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">dict()</code>. The shorter sequence limits the number of pairs.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`labels = ["x", "y", "z"]
vals = [2, 4, 6]
mapping = dict(zip(labels, vals))   # {"x": 2, "y": 4, "z": 6}`}
              </pre>
            </div>
          </div>
        </section>

        {/* Dictionary Properties */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Dictionary Properties
          </h2>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-2 mb-4">
            <li><span className="font-semibold">Ordered</span> (Python 3.7+) - insertion order is preserved; iteration order is stable.</li>
            <li><span className="font-semibold">Changeable</span> - add, change, or remove items after creation.</li>
            <li><span className="font-semibold">Unique keys</span> - duplicate keys overwrite earlier values; only the last wins.</li>
          </ul>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6, "b": 8}  # "b": 4 overwritten
print(data)  # {"a": 2, "b": 8, "c": 6}`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Mixed value types:</span> Values can be strings, numbers, lists, dictionaries, or any type. Keys must be hashable (strings, numbers, tuples of hashables).
            </p>
          </div>
        </section>

        {/* Dictionary Methods */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Dictionary Methods
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
                  <td className="p-3 font-mono">get(key, default)</td>
                  <td className="p-3">Return value for <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">key</code>; <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">default</code> if missing (no <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">KeyError</code>)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">keys()</td>
                  <td className="p-3">View of keys</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">values()</td>
                  <td className="p-3">View of values</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">items()</td>
                  <td className="p-3">View of (key, value) pairs</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">fromkeys(seq, value)</td>
                  <td className="p-3">New dictionary with keys from <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">seq</code>, values <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">value</code> (default <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">None</code>)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">setdefault(key, default)</td>
                  <td className="p-3">Return value; if key missing, insert <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">default</code> and return it</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">update(other)</td>
                  <td className="p-3">Update dictionary with key–value pairs from <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">other</code></td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">pop(key, default)</td>
                  <td className="p-3">Remove <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">key</code>, return value; <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">KeyError</code> if missing (or <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">default</code>)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">popitem()</td>
                  <td className="p-3">Remove and return last inserted (key, value)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">clear()</td>
                  <td className="p-3">Remove all elements from the dictionary</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">copy()</td>
                  <td className="p-3">Return a copy of the dictionary (shallow)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Access Dictionary Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Access Dictionary Items
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-semibold">Bracket notation:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">d[key]</code> - raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">KeyError</code> if key is missing.
          </p>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-semibold">get(key, default):</span> returns <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">default</code> (or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">None</code>) if key is missing; no exception.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
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
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            keys(), values(), items()
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            These return <span className="font-semibold">view objects</span> - they reflect the current state of the dictionary. Changes to the dictionary are visible in the view; the view is not a snapshot.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
k = data.keys()
print(list(k))  # ["a", "b", "c"]
data["d"] = 8
print(list(k))  # ["a", "b", "c", "d"] - view updated`}
            </pre>
          </div>

          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1 mb-4">
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">keys()</code> - view of keys</li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">values()</code> - view of values</li>
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">items()</code> - view of (key, value) tuples</li>
          </ul>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              To get a snapshot (list), wrap in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list()</code>: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list(d.keys())</code>.
            </p>
          </div>

          <div className="mt-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">fromkeys(seq, value):</span> create a dictionary with keys from <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">seq</code>; all values <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">value</code> (default <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">None</code>).
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`dict.fromkeys(["a", "b", "c"])       # {"a": None, "b": None, "c": None}
dict.fromkeys(["a", "b", "c"], 0)    # {"a": 0, "b": 0, "c": 0}`}
              </pre>
            </div>
          </div>

          <div className="mt-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">setdefault(key, default):</span> if <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">key</code> exists, return its value. If not, set <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">d[key] = default</code> and return <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">default</code>. Useful for "get or create".
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`data = {"a": 2, "b": 4}
data.setdefault("c", 6)   # 6; data["c"] = 6
data.setdefault("a", 0)   # 2 - key exists, unchanged`}
              </pre>
            </div>
          </div>
        </section>

        {/* Check if Key Exists */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Check if Key Exists
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">in</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">not in</code> on the dictionary. <span className="font-semibold">Membership tests keys only, not values</span> - so a value like <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">2</code> is not "in" the dictionary unless it is a key.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
"b" in data     # True  - "b" is a key
"x" not in data # True
2 in data       # False - 2 is a value, not a key`}
            </pre>
          </div>
        </section>

        {/* Reversing a Dictionary */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Reversing a Dictionary
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            To get a new dictionary with keys in reverse insertion order, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">dict(reversed(d.items()))</code> (Python 3.8+). For older Python, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">dict(reversed(list(d.items())))</code>.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
rev = dict(reversed(data.items()))   # {"c": 6, "b": 4, "a": 2}`}
            </pre>
          </div>
        </section>

        {/* Unpacking Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Unpacking Dictionaries
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            In a dictionary literal, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">**other</code> unpacks the key–value pairs of <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">other</code> into the new dictionary. Combine with extra keys to override or add entries. Useful for merging or applying overrides without mutating the original.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`base = {"a": 2, "b": 4, "c": 6}
extended = {**base, "c": 8, "d": 10}   # c overwritten, d added
# {"a": 2, "b": 4, "c": 8, "d": 10}`}
            </pre>
          </div>
        </section>

        {/* Add Dictionary Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Add Dictionary Items
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Assign to a new key to add an item.
          </p>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4}
data["c"] = 6
print(data)  # {"a": 2, "b": 4, "c": 6}`}
            </pre>
          </div>
        </section>

        {/* Change Dictionary Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Change Dictionary Items
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Assign to a key to change its value.
          </p>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
data["b"] = 8
print(data)  # {"a": 2, "b": 8, "c": 6}`}
            </pre>
          </div>
        </section>

        {/* Update Dictionary */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Update Dictionary
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-semibold">update(other):</span> merge another dictionary or iterable of (key, value) pairs into the existing dictionary. Existing keys are overwritten; new keys are added. Accepts keyword args.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4}
data.update({"b": 8, "c": 6})       # dictionary: b overwritten, c added
data.update([("d", 8), ("e", 10)])  # iterable of pairs
data.update(f=12)                   # keyword: "f": 12
print(data)  # {"a": 2, "b": 8, "c": 6, "d": 8, "e": 10, "f": 12}`}
            </pre>
          </div>
        </section>

        {/* Remove Dictionary Items */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Remove Dictionary Items
          </h2>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Behavior</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">pop(key)</td>
                  <td className="p-3">Remove <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">key</code>, return value; <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">KeyError</code> if missing</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">pop(key, default)</td>
                  <td className="p-3">Remove <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">key</code>, return value; return <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">default</code> if missing</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">popitem()</td>
                  <td className="p-3">Remove and return last (key, value); Python 3.7+ LIFO; pre-Python 3.7: arbitrary</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">del d[key]</td>
                  <td className="p-3">Remove item; <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">KeyError</code> if missing</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">del d</td>
                  <td className="p-3">Delete the variable; name undefined; access raises <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">NameError</code></td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">clear()</td>
                  <td className="p-3">Remove all items; dictionary becomes <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">{`{}`}</code></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"a": 2, "b": 4, "c": 6}
data.pop("b")      # 4; data = {"a": 2, "c": 6}
data.pop("x", 0)   # 0 - key missing, default returned
data.popitem()     # ("c", 6) - last inserted in Python 3.7+
del data["a"]      # remove "a"
data.clear()       # {}`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">popitem() and version:</span> In Python 3.7+, removes the last inserted item (LIFO). In Python 3.6 and earlier, removes an arbitrary item.
            </p>
          </div>
        </section>

        {/* Loop Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Loop Dictionaries
          </h2>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-2 mb-4">
            <li><span className="font-semibold">By keys (default):</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">for k in d</code> iterates over keys.</li>
            <li><span className="font-semibold">By values:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">for v in d.values()</code>.</li>
            <li><span className="font-semibold">By keys and values:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">for k, v in d.items()</code>.</li>
          </ul>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
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
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Loop in reverse:</span> To iterate in reverse insertion order, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">reversed()</code> on the dictionary or on <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">d.items()</code> (Python 3.8+). On older Python, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">reversed(list(d.items()))</code>.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
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
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Dictionary Comprehension
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Build a dictionary from an iterable with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{'{key_expr: value_expr for ... in ...}'}</code>. You can add an <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">if</code> to filter. Keys and values can be computed from the loop variable.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`squares = {n: n * n for n in [2, 4, 6, 8]}   # {2: 4, 4: 16, 6: 36, 8: 64}
upper = {k.upper(): v for k, v in [("a", 2), ("b", 4)]}  # {"A": 2, "B": 4}
evens_only = {x: x * 2 for x in range(6) if x % 2 == 0}  # {0: 0, 2: 4, 4: 8}`}
            </pre>
          </div>
        </section>

        {/* Copy Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Copy Dictionaries
          </h2>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-2 mb-4">
            <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">d2 = d1</code> creates a <span className="font-semibold">reference</span>, not a copy. Changes to <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">d1</code> affect <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">d2</code>.</li>
            <li><span className="font-semibold">Shallow copy:</span> Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">copy()</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">dict(d)</code>.</li>
          </ul>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`original = {"a": 2, "b": 4, "c": 6}
ref = original        # reference - same object
copy = original.copy()  # shallow copy
also_copy = dict(original)

original["a"] = 0
# ref["a"] is 0; copy["a"] is still 2`}
            </pre>
          </div>

          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Shallow vs deep:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">copy()</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">dict()</code> copy the top level only. Nested dictionaries or lists are shared. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">copy.deepcopy()</code> for full copies.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`import copy

original = {"a": {"x": 2, "y": 4}, "b": {"x": 6, "y": 8}}
shallow = original.copy()
shallow["a"]["x"] = 0   # original["a"]["x"] is also 0 - shared

deep = copy.deepcopy(original)
deep["a"]["x"] = 0      # original unchanged - independent`}
              </pre>
            </div>
          </div>
        </section>

        {/* Sort Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Sort Dictionaries
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Dictionaries have no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sort()</code> method. To iterate in sorted order, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sorted()</code> on keys or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">items()</code>.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = {"c": 6, "a": 2, "b": 4}
for k in sorted(data):
    print(k, data[k])  # a 2, b 4, c 6

for k, v in sorted(data.items()):
    print(k, v)  # a 2, b 4, c 6

for k, v in sorted(data.items(), key=lambda p: p[1]):
    print(k, v)  # a 2, b 4, c 6 - sort by value`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">New dictionary from sorted items:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict(sorted(data.items()))</code> - creates a dictionary with keys in sorted order.
            </p>
          </div>
        </section>

        {/* Nested Dictionaries */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Nested Dictionaries
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Values can be dictionaries. Access with chained brackets.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`nested = {
    "p1": {"x": 2, "y": 4},
    "p2": {"x": 6, "y": 8},
}
nested["p1"]["x"]   # 2
nested["p2"]["y"]  # 8`}
            </pre>
          </div>

          <div className="mt-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Build from separate dictionaries:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`inner1 = {"x": 2, "y": 4}
inner2 = {"x": 6, "y": 8}
outer = {"p1": inner1, "p2": inner2}`}
              </pre>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Loop nested dictionaries:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`for key, obj in nested.items():
    print(key)
    for k, v in obj.items():
        print(f"  {k}: {v}")`}
              </pre>
            </div>
          </div>
        </section>

        {/* Reversing Key-Value Pairs */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Reversing Key–Value Pairs
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            When keys and values are unique, you can build a dictionary that maps each value back to its key: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{'{v: k for k, v in d.items()}'}</code>. Useful for reverse lookups (e.g. "which key has this value?").
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`code_to_name = {"uk": "United Kingdom", "de": "Germany", "fr": "France"}
name_to_code = {v: k for k, v in code_to_name.items()}
name_to_code["Germany"]   # "de"`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              If values are not unique, later keys overwrite earlier ones; only one key per value remains.
            </p>
          </div>
        </section>

        {/* Dictionary vs Other Collections */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Dictionary vs Other Collections
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Collection</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Order</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Indexing</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Key–value</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Use when</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">dictionary</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">By key only</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Lookup by name or id; config; mapping</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">list</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">By position</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Sequence; order matters; duplicates allowed</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">tuple</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">By position</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Immutable sequence; fixed layout; hashable</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">set</td>
                  <td className="p-3">No</td>
                  <td className="p-3">No</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Unique elements; fast membership; set operations</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Note: Dictionary order is guaranteed from Python 3.7 onward (insertion order). In Python 3.6 and earlier, iteration order was not guaranteed.
          </p>
        </section>

        {/* Common Use Cases */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Common Use Cases
          </h2>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-2">
            <li><span className="font-semibold">Config or options:</span> Store names and values (e.g. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{"timeout": 30, "retries": 2}'}</code>). Override with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{**defaults, **overrides}'}</code>.</li>
            <li><span className="font-semibold">Counting:</span> Track how many times something appears. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">counts[key] = counts.get(key, 0) + 1</code> or group with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">setdefault(key, []).append(item)</code>.</li>
            <li><span className="font-semibold">Grouping:</span> Collect items by a key (e.g. group people by role). <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">groups.setdefault(role, []).append(person)</code> keeps a list per role.</li>
            <li><span className="font-semibold">Lookup tables:</span> Map IDs or codes to names, or the reverse. Build from two lists with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict(zip(ids, names))</code> or invert with a comprehension when values are unique.</li>
          </ul>
        </section>

        {/* Tricky Points */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Creating - Empty dictionary</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{`{}`}</code> is falsy; use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if not d:</code> for "no keys". No literal for "empty" other than <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{`{}`}</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Creating - dict() keyword args</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Keys must be valid identifiers (no spaces, no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">2=4</code>); use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{"2": 4}'}</code> for numeric-like keys.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Creating - dict(zip(...))</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Length is the shorter of the two iterables; extra elements in either are ignored.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Dictionary Properties - Duplicate keys</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Later value overwrites earlier; no error; only the last value is kept.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Access - d[key] vs d.get(key)</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">[]</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">KeyError</code> if missing; <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">get()</code> returns <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code> or default.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">keys(), values(), items()</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Return views, not lists; reflect live changes; modifying dictionary size while iterating can raise <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">RuntimeError</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Check if Key Exists - in</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Membership tests <span className="font-semibold">keys</span> only; a value that is not a key (e.g. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'2 in {"a": 2}'}</code>) is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Reversing a Dictionary</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(d)</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(d.items())</code> require Python 3.8+; on older versions use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(list(d.items()))</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Unpacking (dictionary literal)</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Using <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">**other</code> in a dictionary literal builds a new dictionary; does not mutate <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">other</code>. Later keys overwrite.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Update - update() return value</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">update()</code> mutates the dictionary and returns <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>; do not assign the result and expect a new dictionary.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Remove - popitem()</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Python 3.7+: LIFO (last inserted); pre-Python 3.7: arbitrary item.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Remove - del d</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Deletes the variable; the name is gone; access raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">NameError</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Copy - d2 = d1</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Reference, not copy; changes to one affect the other. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">copy()</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict(d)</code> for a shallow copy.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Sort</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Dictionaries have no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sort()</code> method; use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted(d)</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sorted(d.items())</code> for a new ordering.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Nested - missing key</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d["a"]["b"]</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">KeyError</code> if <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">"a"</code> is missing; use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d.get("a", {}).get("b", default)</code> for safe access.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Reversing Key–Value</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">If values are not unique, later keys overwrite earlier; the inverse dictionary has only one key per value.</p>
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
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you create an empty dictionary? What about truthiness?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{`{}`}</code>. An empty dictionary is falsy; use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if not d:</code> for "no keys" and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if d:</code> for "at least one pair".</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you build a dictionary from two lists (keys and values)?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict(zip(keys_list, values_list))</code>. Pairs the first key with the first value, and so on. The length of the result is the length of the shorter list.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">When would you use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict.fromkeys()</code>?</p>
              <p className="text-gray-700 dark:text-gray-300">When you need a dictionary with the same value for many keys (e.g. default 0 or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>). <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">dict.fromkeys(keys, value)</code> is shorter than a loop or comprehension when every value is the same.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Why must dictionary keys be hashable?</p>
              <p className="text-gray-700 dark:text-gray-300">Dictionaries use a hash table; keys are hashed for O(1) lookup. Mutable types (lists, dictionaries) are unhashable and cannot be keys.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Are dictionaries ordered?</p>
              <p className="text-gray-700 dark:text-gray-300">Yes, since Python 3.7 (insertion order). In Python 3.6 it was a CPython implementation detail; before that, unordered. Order matters for predictable iteration and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">popitem()</code> LIFO behavior.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">What is the time complexity of dictionary get, set, and delete?</p>
              <p className="text-gray-700 dark:text-gray-300">O(1) average for get, set, and delete. Iteration is O(n).</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">When should you use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d[key]</code> vs <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d.get(key)</code>?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">get()</code> when the key may be missing and you want a default or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">[]</code> when the key must exist; let <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">KeyError</code> surface bugs.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Are <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">keys()</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">values()</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">items()</code> lists?</p>
              <p className="text-gray-700 dark:text-gray-300">No - they return view objects. Views reflect live changes to the dictionary. Modifying the dictionary while iterating over a view can raise <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">RuntimeError</code> if the dictionary size changes. For a safe snapshot, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list(d.items())</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">in</code> check keys or values?</p>
              <p className="text-gray-700 dark:text-gray-300">Keys only. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">value in d</code> is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">True</code> only if that value is a key; so <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'2 in {"a": 2}'}</code> is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">When should you use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">setdefault()</code> vs <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">get()</code> + assign?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">setdefault()</code> when you need get-or-create and then mutate in place (e.g. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">groups.setdefault(category, []).append(item)</code>). Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">get()</code> when you only need the value.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you iterate a dictionary in reverse insertion order?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(d)</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(d.items())</code> (Python 3.8+). On older Python, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed(list(d.items()))</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">What does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{**d1, **d2}'}</code> do? Does it mutate either dictionary?</p>
              <p className="text-gray-700 dark:text-gray-300">It builds a new dictionary by unpacking <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d1</code> and then <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d2</code>. Keys from <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d2</code> overwrite same-named keys from <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d1</code>. It does not mutate either dictionary.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you merge two dictionaries without mutating either?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{**d1, **d2}'}</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d1 | d2</code> (Python 3.9+). To mutate one of them, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">d1.update(d2)</code>.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">pop()</code> vs <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">del</code> vs <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">popitem()</code> - when to use each?</p>
              <p className="text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">pop(key)</code> returns the value; use when you need it. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">del d[key]</code> removes without returning. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">popitem()</code> removes the last inserted item (Python 3.7+); useful for LIFO processing or draining a dictionary.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">When would you use a dictionary comprehension instead of a loop?</p>
              <p className="text-gray-700 dark:text-gray-300">When you are building a new dictionary from an iterable in one expression (e.g. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{k: v * 2 for k, v in d.items()}'}</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{'{x: x ** 2 for x in range(4)}'}</code>). Keeps the code short and avoids creating an empty dictionary and updating it in a loop.</p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/list"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: List
          </Link>
          <Link
            href="/learn/programming-fundamentals/set"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Set →
          </Link>
        </div>
      </div>
    </div>
  );
}