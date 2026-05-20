

import Link from "next/link";

export default function VariablesPage() {
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
            Variables & Data Types
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Variables store data in memory. Python is dynamically typed — variables can hold any type, and types are checked at runtime.
          </p>
        </div>

        {/* Variable Assignment */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Variable Assignment
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">=</code> to assign a value to a variable. Variable names can contain letters, numbers, and underscores, but cannot start with a number.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Basic assignment</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`x = 10
name = "Alice"
is_valid = True`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Multiple assignment:</span> Assign multiple variables in one line.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Multiple assignment</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`a, b, c = 1, 2, 3
x = y = z = 0  # All three reference 0`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Dynamic typing:</span> Variables can change type at runtime.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Dynamic typing</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`value = 42        # int
value = "hello"  # now str
value = [1, 2, 3] # now list`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Basic Data Types */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Basic Data Types
          </h2>

          <div className="space-y-6">
            {/* Integers */}
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">int — Integer</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                Whole numbers, unbounded in Python 3.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`age = 25
count = -10
binary = 0b1010   # 10 in decimal
hex_num = 0xFF    # 255 in decimal`}
                </pre>
              </div>
            </div>

            {/* Floats */}
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">float — Floating Point</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                Numbers with decimal points. Based on IEEE 754 double-precision.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`price = 19.99
pi = 3.14159
scientific = 1.5e-3  # 0.0015`}
                </pre>
              </div>
            </div>

            {/* Strings */}
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">str — String</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                Sequence of Unicode characters. Use single, double, or triple quotes.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`name = "Python"
single = 'Hello'
multi_line = """This spans
multiple lines"""`}
                </pre>
              </div>
            </div>

            {/* Booleans */}
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">bool — Boolean</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                Represents truth values. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">True</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">False</code> (capitalized).
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`is_active = True
is_done = False
is_greater = 10 > 5  # True`}
                </pre>
              </div>
            </div>

            {/* None */}
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">NoneType — None</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                Represents absence of a value. Similar to <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">null</code> in other languages.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`result = None
if result is None:
    print("No value")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Type Inspection */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Type Inspection
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">type() function:</span> Returns the type of an object.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`print(type(42))      # <class 'int'>
print(type("hello")) # <class 'str'>
print(type(3.14))    # <class 'float'>`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">isinstance() function:</span> Checks if an object is of a specific type (preferred over <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">type()</code> for type checking).
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`print(isinstance(42, int))     # True
print(isinstance("hi", str))   # True
print(isinstance(3.14, int))   # False`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Type Conversion */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Type Conversion (Casting)
          </h2>
          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`# String to int
num = int("123")      # 123

# Int to float
pi = float(3)         # 3.0

# Number to string
text = str(456)       # "456"

# String to list
chars = list("abc")   # ['a', 'b', 'c']`}
              </pre>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                ⚠️ <span className="font-semibold">Caution:</span> Invalid conversion raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">ValueError</code>.
                Example: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">int("hello")</code> fails.
              </p>
            </div>
          </div>
        </section>

        {/* Naming Conventions */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Naming Conventions (PEP 8)
          </h2>
          <div className="space-y-3">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">snake_case</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Variables and functions</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">PascalCase</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Classes</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">UPPER_SNAKE_CASE</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Constants</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">_leading_underscore</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Protected/internal (convention)</p>
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
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Dynamic typing can cause bugs</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Variables can change type unexpectedly. Use type hints and mypy to catch errors early.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Float precision issues</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">0.1 + 0.2 == 0.3</code> is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">False</code> due to binary representation. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">decimal.Decimal</code> for exact arithmetic.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Strings are immutable</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">s[0] = 'a'</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">TypeError</code>. Create new strings instead.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Assignment copies references, not values</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                For mutable objects, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">b = a</code> makes both reference the same object. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">copy()</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">deepcopy()</code> for independent copies.
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
                What is dynamic typing in Python?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                Variables don't have fixed types. A single variable can reference an integer, then a string, then a list. Type is checked at runtime.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is the difference between <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">type()</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">isinstance()</code>?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">isinstance()</code> considers inheritance (returns True for subclasses). <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">type()</code> does not. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">isinstance()</code> is preferred for type checking.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                Why does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">0.1 + 0.2 != 0.3</code>?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                Floating point numbers use binary representation. Some decimals cannot be represented exactly. The result is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">0.30000000000000004</code>.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What are Python's immutable data types?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">int</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">float</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">str</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">tuple</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">frozenset</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">bool</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">None</code>.
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/hello-world"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Hello World
          </Link>
          <Link
            href="/learn/programming-fundamentals//basic-io"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Basic I/O →
          </Link>
        </div>
      </div>
    </div>
  );
}