
import Link from "next/link";

export default function HelloWorldPage() {
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
            Hello World
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            A minimal Python program prints output to the console. This lesson focuses on{" "}
            <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono text-sm">
              print()
            </code>{" "}
            basics so you can run and verify your first scripts quickly.
          </p>
        </div>

        {/* The print() Function */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            The print() Function
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print()</code>{" "}
            sends text to standard output. It accepts one or more arguments and converts them to strings before writing.
          </p>

          <div className="space-y-5">
            {/* Basic */}
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">print("Hello, World!")</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  Output: Hello, World!
                </pre>
              </div>
            </div>

            {/* Multiple arguments */}
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Multiple arguments:</span> Arguments are separated by spaces by default.{" "}
                The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sep</code> parameter controls the separator.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">print("Hello", "World", "!")</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  Output: Hello World !
                </pre>
              </div>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-3">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">print("2", "4", "6", sep="-")</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  Output: 2-4-6
                </pre>
              </div>
            </div>

            {/* End parameter */}
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">End parameter:</span> The{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">end</code> parameter controls what is printed after the last argument.{" "}
                The default is a newline (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">\n</code>).
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">
                  print("Line one", end="")
                  {"\n"}print("Line two")
                </div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  Output: Line oneLine two
                </pre>
              </div>
            </div>

            {/* No arguments */}
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">No arguments:</span>{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print()</code> with no arguments prints a single newline.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">print()</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  Output: (blank line)
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Common Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Common Patterns
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Minimal executable script:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  print("Hello, World!")
                </pre>
              </div>
            </div>
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Multiple values in one line:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`name = "Alice"
score = 82
print("Name:", name, "Score:", score)`}
                </pre>
              </div>
            </div>
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Custom separator for compact output:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  print("2", "4", "6", sep="-")
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Python Syntax Basics */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Python Syntax Basics
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Indentation:</span> Python uses indentation to define blocks. Four spaces per level is conventional. Mixing tabs and spaces can cause errors.
              </p>
            </div>
            <div>
              <p className="text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Colon:</span> A colon (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">:</code>) starts a block. The block must be indented.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`if True:
    print("indented block")`}
                </pre>
              </div>
            </div>
            <div>
              <p className="text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Comments:</span> Lines starting with{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">#</code> are comments and are ignored.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# This is a comment
print("Hello")  # inline comment`}
                </pre>
              </div>
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
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Multiple arguments are converted to strings</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print()</code> stringifies each argument. You can pass numbers, booleans, and other objects directly.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Separator affects readability</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Using a custom <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sep</code> is useful, but overusing separators can make logs harder to scan.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">print() adds a newline by default</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print("x")</code> ends with <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">\n</code>. Use{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">end=""</code> when you need to continue on the same line.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Indentation errors</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Any block statement in Python must be indented consistently, or{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">IndentationError</code> is raised.
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
                How do you pass multiple values to print()?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                Pass them as separate arguments: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print("a", "b", "c")</code>. They are separated by spaces by default; use{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sep</code> to change this.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is the difference between sep and end in print()?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sep</code> controls text between arguments.{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">end</code> controls what is printed after the last argument.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is the default value of end in print()?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                A newline character (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">\n</code>). Use{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">end=""</code> to suppress the trailing newline.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What does print() do with non-string values?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                It calls string conversion and prints the result. For example,{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print(2, True, [4, 6])</code> is valid.
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <div className="text-gray-400">Previous: Getting Started</div>
          <Link
            href="/learn/programming-fundamentals/variables"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Variables & Data Types →
          </Link>
        </div>
      </div>
    </div>
  );
}