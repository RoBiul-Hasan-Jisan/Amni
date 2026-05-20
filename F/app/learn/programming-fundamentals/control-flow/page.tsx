// app/learn/programming-fundamentals/control-flow/page.tsx

import Link from "next/link";

export default function ControlFlowPage() {
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
            Control Flow
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Control flow statements determine the order in which code executes. Python provides conditionals (if/elif/else) and pattern matching (match/case) for branching logic.
          </p>
        </div>

        {/* if Statement */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            if Statement
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">if</code> statement executes a block only when a condition is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">True</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Basic if</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`age = 18
if age >= 18:
    print("You can vote")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">if-else:</span> Execute one block if condition is True, another if False.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">if-else</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`temperature = 30
if temperature > 25:
    print("It's hot outside")
else:
    print("It's cool outside")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">if-elif-else:</span> Check multiple conditions in sequence.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">if-elif-else</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Grade: {grade}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison Operators */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Comparison Operators
          </h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">==</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Equal to</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">!=</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Not equal to</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">&gt;</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Greater than</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">&lt;</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Less than</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">&gt;=</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Greater than or equal to</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-mono text-gray-900 dark:text-white">&lt;=</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">Less than or equal to</p>
            </div>
          </div>
        </section>

        {/* Logical Operators */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Logical Operators
          </h2>
          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">and, or, not</div>
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`age = 25
has_license = True

# AND - both conditions must be True
if age >= 18 and has_license:
    print("You can drive")

# OR - at least one condition must be True
if age < 18 or age > 65:
    print("Discount applies")

# NOT - reverses the condition
if not has_license:
    print("Get a license first")

# Combining operators
if (age >= 18 and has_license) or age >= 65:
    print("Eligible")`}
              </pre>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                💡 <span className="font-semibold">Short-circuit evaluation:</span> <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">and</code> stops at first False, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">or</code> stops at first True.
              </p>
            </div>
          </div>
        </section>

        {/* match/case Statement (Python 3.10+) */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            match/case Statement (Python 3.10+)
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Structural pattern matching — more powerful than switch statements in other languages.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Basic match/case</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`command = "start"
match command:
    case "start":
        print("Starting...")
    case "stop":
        print("Stopping...")
    case "restart":
        print("Restarting...")
    case _:  # Default case (underscore)
        print("Unknown command")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Matching with guards:</span> Add conditions to patterns.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Guards (if conditions)</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`value = 10
match value:
    case x if x < 0:
        print(f"Negative: {x}")
    case x if x == 0:
        print("Zero")
    case x if x > 0:
        print(f"Positive: {x}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Matching sequences:</span> Match against tuples or lists.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Sequence patterns</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`point = (2, 3)
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"On Y-axis at {y}")
    case (x, 0):
        print(f"On X-axis at {x}")
    case (x, y):
        print(f"Point at ({x}, {y})")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Matching with OR patterns:</span> Multiple patterns for one case.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">OR patterns (|)</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`status = 404
match status:
    case 200 | 201:
        print("Success")
    case 400 | 404:
        print("Client error")
    case 500 | 502 | 503:
        print("Server error")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Truthiness and Falsy Values */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Truthiness and Falsy Values
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Any value can be used as a condition. Some values are considered <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">False</code> (falsy), others are <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">True</code> (truthy).
          </p>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-red-600 dark:text-red-400 mb-2">Falsy Values</p>
              <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1">
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code></li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code></li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">0</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">0.0</code></li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">""</code> (empty string)</li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">[]</code> (empty list)</li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{}</code> (empty dict)</li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">set()</code> (empty set)</li>
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">range(0)</code></li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
              <p className="font-semibold text-green-600 dark:text-green-400 mb-2">Truthy Values</p>
              <ul className="text-gray-700 dark:text-gray-300 text-sm space-y-1">
                <li><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">True</code></li>
                <li>Non-zero numbers (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">1</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">-5</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">3.14</code>)</li>
                <li>Non-empty strings (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">"Hello"</code>)</li>
                <li>Non-empty containers (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">[1, 2]</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{"{'a': 1}"}</code>)</li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden mt-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Useful pattern: check if container has items
items = []
if items:  # False for empty list
    print("Has items")
else:
    print("Empty")`}
            </pre>
          </div>
        </section>

        {/* Conditional Expressions (Ternary Operator) */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Conditional Expressions (Ternary Operator)
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            One-line if-else expressions for simple assignments.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Syntax: value_if_true if condition else value_if_false
age = 20
status = "Adult" if age >= 18 else "Minor"
print(status)  # Adult

# Nested ternary (use sparingly)
score = 85
result = "A" if score >= 90 else "B" if score >= 80 else "C"
print(result)  # B`}
            </pre>
          </div>
        </section>

        {/* Chained Comparisons */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Chained Comparisons
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Python allows chaining comparison operators for cleaner code.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Instead of: if x > 5 and x < 10:
x = 7
if 5 < x < 10:
    print("x is between 5 and 10")

# Works with other operators too
y = 15
if 0 <= y <= 100:
    print("y is in range")

# Chained comparisons are short-circuited
a = 5
if a == a < 10:  # Equivalent to a == a and a < 10
    print("True")`}
            </pre>
          </div>
        </section>

        {/* Tricky Points */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Assignment vs Comparison</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if x = 5:</code> is a syntax error. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">==</code> for comparison. Python doesn't allow assignment in conditions (unlike C).
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Indentation Errors</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Inconsistent indentation within a block raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">IndentationError</code>. Always use consistent spaces (4 per level).
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Dangling Else</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Python's <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">else</code> attaches to the nearest <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if</code>. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">elif</code> for explicit chaining.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">match/case is not switch</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Unlike C-style switch, Python's <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match/case</code> doesn't fall through. No <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code> needed. It supports structural pattern matching.
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
                What is short-circuit evaluation?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                In <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">and</code>, if the first operand is False, the second isn't evaluated. In <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">or</code>, if the first is True, the second isn't evaluated. Useful for avoiding errors: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if x is not None and x &gt; 0:</code>
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                Difference between <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">==</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">is</code>?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">==</code> compares values. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">is</code> compares identity (memory address). Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">is</code> with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>, True, False.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What values are falsy in Python?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">0</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">0.0</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">""</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">[]</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">{}</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">set()</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">range(0)</code>. Everything else is truthy.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                How does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match/case</code> differ from switch in C/Java?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                No fall-through (no break needed). Supports destructuring, guards, and complex pattern matching on any data structure, not just primitive values.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                Write a ternary operator equivalent of <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if x &gt; 0: y = 1 else: y = -1</code>
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">y = 1 if x &gt; 0 else -1</code>
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/basic-io"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Basic I/O
          </Link>
          <Link
            href="/learn/programming-fundamentals/loops"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Loops →
          </Link>
        </div>
      </div>
    </div>
  );
}