// app/learn/programming-fundamentals/string/page.tsx

import Link from "next/link";

export default function StringPage() {
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
            Strings
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Strings are immutable sequences of Unicode characters. They support concatenation, formatting, searching, slicing, and type-checking methods. Mastery of these operations is essential for text processing and interview problems.
          </p>
        </div>

        {/* Creating Strings */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Creating Strings
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Quote Style</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Use when</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Single quotes</td>
                  <td className="p-3 font-mono">'hello'</td>
                  <td className="p-3">Most common, contains double quotes inside</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Double quotes</td>
                  <td className="p-3 font-mono">"world"</td>
                  <td className="p-3">Contains single quotes inside (e.g., "It's")</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">Triple quotes</td>
                  <td className="p-3 font-mono">"""multiline"""</td>
                  <td className="p-3">Multiline strings, docstrings</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Different quote styles
s1 = "hello"
s2 = 'world'
s3 = "It's a beautiful day"
s4 = 'He said "Hello"'
s5 = """This is
a multiline
string"""

# Empty string
empty = ""

# String from other types
s6 = str(123)           # "123"
s7 = str(3.14)          # "3.14"
s8 = str([1, 2, 3])     # "[1, 2, 3]"

# Important: Strings are IMMUTABLE
s = "hello"
# s[0] = "H"           # TypeError: 'str' object does not support item assignment
s = "Hello"              # This creates a new string`}
            </pre>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mt-4">
            <p className="font-semibold text-gray-900 dark:text-white">Immutability</p>
            <p className="text-gray-700 dark:text-gray-300">Strings cannot be changed in place. Operations such as <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">replace()</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">upper()</code> return new strings; the original is unchanged.</p>
          </div>
        </section>

        {/* Slicing and Indexing */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Slicing and Indexing
          </h2>
          
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Indexing</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[i]</td>
                  <td className="p-3">Single character at index i</td>
                  <td className="p-3 font-mono">"Python"[2] → "t"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[-1]</td>
                  <td className="p-3">Last character</td>
                  <td className="p-3 font-mono">"Python"[-1] → "n"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[-2]</td>
                  <td className="p-3">Second last character</td>
                  <td className="p-3 font-mono">"Python"[-2] → "o"</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Slicing</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[:n]</td>
                  <td className="p-3">First n characters</td>
                  <td className="p-3 font-mono">"Python"[:2] → "Py"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[n:]</td>
                  <td className="p-3">From index n to end</td>
                  <td className="p-3 font-mono">"Python"[2:] → "thon"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[i:j]</td>
                  <td className="p-3">Slice from i to j (exclusive)</td>
                  <td className="p-3 font-mono">"Python"[1:4] → "yth"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[-m:-n]</td>
                  <td className="p-3">Slice with negative indices</td>
                  <td className="p-3 font-mono">"Python"[-4:-2] → "th"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[::k]</td>
                  <td className="p-3">Every k-th character</td>
                  <td className="p-3 font-mono">"Python"[::2] → "Pto"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s[::-1]</td>
                  <td className="p-3">Reversed string</td>
                  <td className="p-3 font-mono">"Python"[::-1] → "nohtyP"</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`s = "Python"

# Indexing (0-based)
s[0]    # "P"
s[2]    # "t"
s[-1]   # "n" (last character)
s[-2]   # "o" (second last)

# Slicing [start:stop:step]
s[:2]       # "Py"
s[2:]       # "thon"
s[1:4]      # "yth"
s[-4:-2]    # "th"
s[::2]      # "Pto" (every 2nd character)
s[::-1]     # "nohtyP" (reverse)

# Length
len(s)      # 6

# Out of range - raises IndexError
# s[20]     # IndexError: string index out of range

# reversed() and join()
"".join(reversed(s))    # "nohtyP"`}
            </pre>
          </div>
        </section>

        {/* Concatenation */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Concatenation
          </h2>
          
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">The + Operator</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Join two strings into a new string
"Hello" + " " + "World"    # "Hello World"
"2" + "4" + "6"            # "246"

# Type error - both operands must be strings
# "2" + 4                  # TypeError: can only concatenate str to str

# Convert numbers to strings
"2" + str(4)               # "24"`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">The join() Method</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">sep.join(iterable)</td>
                  <td className="p-3">Joins strings in iterable with separator</td>
                  <td className="p-3 font-mono">"-".join(["a","b","c"]) → "a-b-c"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">"".join(iterable)</td>
                  <td className="p-3">Concatenates with no separator</td>
                  <td className="p-3 font-mono">"".join(["a","b","c"]) → "abc"</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# join() with different separators
"-".join(["a", "b", "c"])      # "a-b-c"
"".join(["2", "4", "6"])       # "246"
" | ".join(["x", "y", "z"])    # "x | y | z"
", ".join(["apple", "banana", "cherry"])  # "apple, banana, cherry"

# All elements must be strings
# ",".join([1, 2, 3])          # TypeError

# Convert to strings first
",".join(str(x) for x in [1, 2, 3])  # "1,2,3"

# join() with other iterables
"".join("abc")                 # "abc"
"-".join(("a", "b", "c"))      # "a-b-c"`}
            </pre>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mt-4">
            <p className="font-semibold text-gray-900 dark:text-white">Key Points about join()</p>
            <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1 ml-4 list-disc mt-2">
              <li><strong>Order:</strong> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">sep.join(parts)</code> produces <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">parts[0] + sep + parts[1] + sep + ...</code></li>
              <li><strong>Non-string elements:</strong> All elements must be strings</li>
              <li><strong>Empty separator:</strong> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">"".join(parts)</code> concatenates with no separator</li>
            </ul>
          </div>
        </section>

        {/* Formatting */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Formatting
          </h2>
          
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">F-Strings (Python 3.6+)</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`name = "Alice"
age = 25
score = 86.8

# Basic f-string
f"Name: {name}, Age: {age}"           # "Name: Alice, Age: 25"

# Format specifiers
f"Score: {score:.2f}"                 # "Score: 86.80"
f"Percentage: {score:.1%}"            # "Percentage: 8680.0%"

# Expressions inside braces
x, y = 4, 6
f"{x} * {y} = {x * y}"                # "4 * 6 = 24"

# Alignment and padding
f"{'hi':<10}"                         # "hi        " (left align)
f"{'hi':>10}"                         # "        hi" (right align)
f"{'hi':^10}"                         # "    hi    " (center)
f"{42:06d}"                           # "000042" (zero pad)`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">The format() Method</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Positional arguments
"{} and {}".format("a", "b")          # "a and b"
"{0} and {1}".format("x", "y")        # "x and y"
"{1} and {0}".format("x", "y")        # "y and x"

# Keyword arguments
"{name} is {age} years old".format(name="Alice", age=25)  # "Alice is 25 years old"

# Mixed
"{0} is {name}".format("Alice", name="Smith")  # "Alice is Smith"

# Format specifiers
"{:.2f}".format(3.14159)              # "3.14"
"{:06d}".format(42)                   # "000042"`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Alignment and Padding Methods</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.center(width)</td>
                  <td className="p-3">Center string in field of given width</td>
                  <td className="p-3 font-mono">"hi".center(6) → "  hi  "</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.ljust(width)</td>
                  <td className="p-3">Left-justify; pad on the right</td>
                  <td className="p-3 font-mono">"hi".ljust(6) → "hi    "</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.rjust(width)</td>
                  <td className="p-3">Right-justify; pad on the left</td>
                  <td className="p-3 font-mono">"hi".rjust(6) → "    hi"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.zfill(width)</td>
                  <td className="p-3">Pad with zeros on the left</td>
                  <td className="p-3 font-mono">"42".zfill(6) → "000042"</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Padding with custom fill character
"hi".center(6, "-")      # "--hi--"
"hi".ljust(6, "*")       # "hi****"
"hi".rjust(6, "_")       # "____hi"

# zfill preserves sign
"-42".zfill(6)           # "-00042"
"+42".zfill(6)           # "+00042"`}
            </pre>
          </div>
        </section>

        {/* Modify Strings */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Modify Strings
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">Strings are immutable; these methods return new strings.</p>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Case Conversion</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.upper()</td>
                  <td className="p-3">All characters to uppercase</td>
                  <td className="p-3 font-mono">"hello".upper() → "HELLO"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.lower()</td>
                  <td className="p-3">All characters to lowercase</td>
                  <td className="p-3 font-mono">"HELLO".lower() → "hello"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.casefold()</td>
                  <td className="p-3">Aggressive lowercase for caseless comparison</td>
                  <td className="p-3 font-mono">"Straße".casefold() → "strasse"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.capitalize()</td>
                  <td className="p-3">First character uppercase; rest lowercase</td>
                  <td className="p-3 font-mono">"hello".capitalize() → "Hello"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.title()</td>
                  <td className="p-3">First letter of each word uppercase</td>
                  <td className="p-3 font-mono">"hello world".title() → "Hello World"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.swapcase()</td>
                  <td className="p-3">Swap uppercase and lowercase</td>
                  <td className="p-3 font-mono">"HeLLo".swapcase() → "hEllO"</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4 mb-4">
            <p className="font-semibold text-gray-900 dark:text-white">casefold() vs lower()</p>
            <p className="text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">casefold()</code> handles more Unicode cases (e.g., German ß → ss). Use it for caseless comparison when <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">lower()</code> is insufficient.</p>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Stripping Whitespace and Characters</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.strip()</td>
                  <td className="p-3">Remove leading and trailing whitespace</td>
                  <td className="p-3 font-mono">"  hi  ".strip() → "hi"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.lstrip()</td>
                  <td className="p-3">Remove leading whitespace</td>
                  <td className="p-3 font-mono">"  hi".lstrip() → "hi"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.rstrip()</td>
                  <td className="p-3">Remove trailing whitespace</td>
                  <td className="p-3 font-mono">"hi  ".rstrip() → "hi"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.strip(chars)</td>
                  <td className="p-3">Remove leading/trailing chars in chars</td>
                  <td className="p-3 font-mono">"xxhixx".strip("x") → "hi"</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Replace</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Replace all occurrences
"aabb".replace("a", "x")                     # "xxbb"

# Replace with count limit
"one one one".replace("one", "two", 2)       # "two two one"

# Remove characters by replacing with empty
"hello world".replace(" ", "")               # "helloworld"

# Replace can remove substrings
"file.txt".replace(".txt", "")               # "file"

# Warning: replace with empty string inserts between every character
"abc".replace("", "x")                       # "xaxbxcx"`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Split and Partition</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.split(sep)</td>
                  <td className="p-3">Split by separator; returns list</td>
                  <td className="p-3 font-mono">"a,b,c".split(",") → ["a","b","c"]</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.split(sep, maxsplit)</td>
                  <td className="p-3">Split at most maxsplit times from left</td>
                  <td className="p-3 font-mono">"a,b,c".split(",", 1) → ["a","b,c"]</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.rsplit(sep, maxsplit)</td>
                  <td className="p-3">Split at most maxsplit times from right</td>
                  <td className="p-3 font-mono">"a,b,c".rsplit(",", 1) → ["a,b","c"]</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.splitlines()</td>
                  <td className="p-3">Split on line boundaries</td>
                  <td className="p-3 font-mono">"a\\nb".splitlines() → ["a","b"]</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.partition(sep)</td>
                  <td className="p-3">Split at first sep → (before, sep, after)</td>
                  <td className="p-3 font-mono">"a:b:c".partition(":") → ("a",":","b:c")</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.rpartition(sep)</td>
                  <td className="p-3">Split at last sep → (before, sep, after)</td>
                  <td className="p-3 font-mono">"a:b:c".rpartition(":") → ("a:b",":","c")</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# split() with no argument - splits on whitespace
"  a   b   c  ".split()                # ["a", "b", "c"]

# maxsplit examples
"a,b,c,d".split(",", 2)               # ["a", "b", "c,d"]
"a,b,c,d".rsplit(",", 2)              # ["a,b", "c", "d"]

# partition when sep not found
"hello".partition("x")                # ("hello", "", "")

# splitlines handles different line endings
"line1\\nline2\\r\\nline3".splitlines()  # ["line1", "line2", "line3"]`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">Prefix and Suffix Removal (Python 3.9+)</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# removeprefix() - removes prefix if present
"https://example.com".removeprefix("https://")    # "example.com"
"hello.txt".removeprefix("hello.")                # "txt"

# removesuffix() - removes suffix if present
"file.txt".removesuffix(".txt")                   # "file"
"archive.tar.gz".removesuffix(".gz")              # "archive.tar"

# If not present, returns original unchanged
"hello".removeprefix("world")                     # "hello"`}
            </pre>
          </div>
        </section>

        {/* Search */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Search
          </h2>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">find() and index()</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md mb-4">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Method</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">When found</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">When not found</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.find(sub)</td>
                  <td className="p-3">Index (int)</td>
                  <td className="p-3">-1</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.index(sub)</td>
                  <td className="p-3">Index (int)</td>
                  <td className="p-3 text-red-600">ValueError</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">rfind() and rindex()</h3>
          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Return highest index where substring is found
"hello world".rfind("l")    # 9
"hello world".rindex("l")   # 9

# rfind returns -1 when absent
"hello".rfind("x")          # -1

# rindex raises ValueError when absent
# "hello".rindex("x")       # ValueError`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mt-4 mb-2">count(), in, startswith(), endswith()</h3>
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
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.count(sub)</td>
                  <td className="p-3">Number of non-overlapping occurrences</td>
                  <td className="p-3 font-mono">"hello".count("l") → 2</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">sub in s</td>
                  <td className="p-3">True if substring exists</td>
                  <td className="p-3 font-mono">"ll" in "hello" → True</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.startswith(prefix)</td>
                  <td className="p-3">True if string starts with prefix</td>
                  <td className="p-3 font-mono">"hello.py".startswith("hello") → True</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-mono">s.endswith(suffix)</td>
                  <td className="p-3">True if string ends with suffix</td>
                  <td className="p-3 font-mono">"hello.py".endswith(".py") → True</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# count() - non-overlapping occurrences
"hello".count("l")              # 2
"aaa".count("aa")               # 1 (not 2 - non-overlapping)

# startswith/endswith with tuples (any match)
"hello.py".endswith((".py", ".txt"))  # True
"hello.jpg".endswith((".py", ".txt")) # False

# Using startswith and endswith with slices
s = "hello world"
s.startswith("wor", 6)          # True - start checking at index 6
s.endswith("wor", 0, 9)         # True - check in range [0,9)`}
            </pre>
          </div>
        </section>

        {/* Type-Checking Methods */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Type-Checking Methods
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">These methods return <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">True</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code> based on character properties. An empty string returns <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code> for all except <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">isspace()</code> (empty string → False).</p>

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
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.isalpha()</td><td className="p-3">All characters are alphabetic</td><td className="p-3 font-mono">"hello".isalpha() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.isalnum()</td><td className="p-3">All characters are alphanumeric</td><td className="p-3 font-mono">"abc123".isalnum() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.isdigit()</td><td className="p-3">All characters are digits (0–9, ², ₀, etc.)</td><td className="p-3 font-mono">"468".isdigit() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.isnumeric()</td><td className="p-3">All characters are numeric (incl. ½, Ⅷ, etc.)</td><td className="p-3 font-mono">"½¾".isnumeric() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.islower()</td><td className="p-3">All cased characters are lowercase</td><td className="p-3 font-mono">"hello".islower() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.isupper()</td><td className="p-3">All cased characters are uppercase</td><td className="p-3 font-mono">"HELLO".isupper() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.istitle()</td><td className="p-3">String is title-cased</td><td className="p-3 font-mono">"Hello World".istitle() → True</td></tr>
                <tr className="border-b border-gray-200 dark:border-gray-700"><td className="p-3 font-mono">s.isspace()</td><td className="p-3">All characters are whitespace</td><td className="p-3 font-mono">"   ".isspace() → True</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
            <p className="font-semibold text-gray-900 dark:text-white">isdigit() vs isnumeric()</p>
            <div className="overflow-x-auto mt-2">
              <table className="min-w-full">
                <thead><tr className="text-gray-700 dark:text-gray-300"><th className="text-left p-2">String</th><th className="text-left p-2">isdigit()</th><th className="text-left p-2">isnumeric()</th></tr></thead>
                <tbody className="text-sm">
                  <tr><td className="p-2 font-mono">"246"</td><td className="p-2">True</td><td className="p-2">True</td></tr>
                  <tr><td className="p-2 font-mono">"²"</td><td className="p-2">True</td><td className="p-2">True</td></tr>
                  <tr><td className="p-2 font-mono">"½"</td><td className="p-2">False</td><td className="p-2">True</td></tr>
                  <tr><td className="p-2 font-mono">"Ⅷ"</td><td className="p-2">False</td><td className="p-2">True</td></tr>
                </tbody>
              </table>
            </div>
            <p className="text-gray-700 dark:text-gray-300 text-sm mt-2"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">isdigit()</code> covers digits (0–9) and some Unicode digit symbols. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">isnumeric()</code> is broader: vulgar fractions, Roman numerals, and other numeric symbols.</p>
          </div>
        </section>

        {/* Common Use Cases */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Common Use Cases
          </h2>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">1. Check if string is palindrome</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]

is_palindrome("A man a plan a canal panama")  # True`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">2. Count vowels in a string</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`def count_vowels(s: str) -> int:
    vowels = set("aeiouAEIOU")
    return sum(1 for char in s if char in vowels)

count_vowels("Hello World")  # 3`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">3. Remove punctuation from string</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`import string
s = "Hello, World!"
clean = "".join(char for char in s if char not in string.punctuation)
# "Hello World"`}
              </pre>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">4. Extract file extension</p>
              <pre className="bg-gray-900 p-2 rounded text-green-400 text-sm overflow-x-auto">
                {`filename = "document.pdf"
extension = filename.split(".")[-1]  # "pdf"
# Or using removesuffix (Python 3.9+)
name = filename.removesuffix(f".{extension}")  # "document"`}
              </pre>
            </div>
          </div>
        </section>

        {/* Performance Tips */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Performance Tips
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">String concatenation in loops</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# SLOW - creates many intermediate strings
result = ""
for s in list_of_strings:
    result += s

# FAST - use join()
result = "".join(list_of_strings)`}
              </pre>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">Use in operator for substring check</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# Clean and fast
if substring in string:
    print("Found")

# Instead of find() != -1
if string.find(substring) != -1:
    print("Found")`}
              </pre>
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
              <p className="font-semibold text-gray-900 dark:text-white">Strings are immutable</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# WRONG - strings cannot be modified in place
s = "hello"
# s[0] = "H"        # TypeError

# CORRECT - assign the result of the method
s = s.upper()        # s becomes "HELLO"`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">split() with empty separator</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# "ab".split("")     # ValueError: empty separator
# Use list() instead
list("ab")           # ["a", "b"]`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">strip() removes characters, not substrings</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# strip() removes any characters in the argument string
"hello".strip("ho")    # "ell" - removes 'h' and 'o'
"hello".strip("he")    # "llo" - removes 'h' and 'e'

# To remove a specific substring, use replace
"hello".replace("he", "")  # "llo"`}
              </pre>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">find() vs index()</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">find()</code> when absence is expected; it returns -1. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index()</code> when absent is an error; it raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">ValueError</code>.</p>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white">replace() with count</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`# replace(old, new, count) replaces from the LEFT
"one one one one".replace("one", "two", 2)  # "two two one one"
# Not "one one two two"`}
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
            <div>
              <p className="font-semibold text-gray-900 dark:text-white">How do you reverse a string?</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`s[::-1]                          # Slicing approach
"".join(reversed(s))               # join with reversed iterator`}
              </pre>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">What does "".join(parts) do?</p>
              <p className="text-gray-700 dark:text-gray-300">Concatenates all strings in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">parts</code> with no separator. Equivalent to <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">parts[0] + parts[1] + ...</code> when all elements are strings.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">When to use casefold() instead of lower()?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">casefold()</code> for caseless comparison when <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">lower()</code> is insufficient (e.g., German ß → ss). <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">casefold()</code> produces a more aggressive normalization.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">What is the difference between find() and index()?</p>
              <p className="text-gray-700 dark:text-gray-300">Both return the lowest index of the substring. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">find()</code> returns -1 when not found; <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">index()</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">ValueError</code>.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">What is the difference between isdigit() and isnumeric()?</p>
              <p className="text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">isdigit()</code> is True for digits (0–9) and some Unicode digit symbols. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">isnumeric()</code> is broader: vulgar fractions (½), Roman numerals (Ⅷ), and other numeric symbols.</p>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">How to check if a string contains only whitespace?</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`s.isspace()                      # True if all characters are whitespace
len(s.strip()) == 0              # Alternative`}
              </pre>
            </div>

            <div>
              <p className="font-semibold text-gray-900 dark:text-white">How do you convert a list of integers to a comma-separated string?</p>
              <pre className="mt-2 text-sm bg-gray-900 p-2 rounded text-green-400 overflow-x-auto">
                {`numbers = [1, 2, 3, 4, 5]
result = ",".join(str(n) for n in numbers)  # "1,2,3,4,5"`}
              </pre>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/collections"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Collections
          </Link>
          <Link
            href="/learn/programming-fundamentals/tuple"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Tuple→
          </Link>
        </div>
      </div>
    </div>
  );
}