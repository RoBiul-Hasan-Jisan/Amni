

import Link from "next/link";

export default function BasicIOPage() {
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
            Basic I/O
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Input and output operations let programs interact with users. Python provides <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print()</code> for output and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">input()</code> for user input.
          </p>
        </div>

        {/* Output with print() */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Output with print()
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print()</code> function writes text to standard output (usually the console).
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Basic output</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`print("Hello, World!")
print(42)
print(3.14159)
print(True)`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Separator (sep):</span> Control how multiple arguments are joined.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Using sep parameter</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`print("apple", "banana", "cherry")           # apple banana cherry
print("apple", "banana", "cherry", sep=", ")  # apple, banana, cherry
print("apple", "banana", "cherry", sep=" | ") # apple | banana | cherry`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">End character (end):</span> Control what is printed after the last argument.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Using end parameter</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`print("First line")           # Ends with newline
print("Same", end=" ")         # Ends with space
print("line")                  # Continues on same line
print("No newline", end="")    # No newline at all`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Formatted strings (f-strings):</span> Embed expressions inside strings.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">f-string formatting</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old")
print(f"Next year I will be {age + 1}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">String formatting methods:</span> Alternative ways to format strings.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">.format() method</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# .format() method
print("My name is {} and I am {} years old".format(name, age))
print("{1} is {0} years old".format(age, name))  # Positional

# %-formatting (old style)
print("My name is %s and I am %d years old" % (name, age))`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Input with input() */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Input with input()
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">input()</code> function reads a line from standard input (usually the keyboard) and returns it as a <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">str</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Basic input</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`name = input()
print("Hello", name)`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Input with prompt:</span> Display a message before reading input.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Prompt parameter</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`name = input("Enter your name: ")
age = input("Enter your age: ")
print(f"Hello {name}, you are {age} years old")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Type conversion:</span> Convert input from string to other types.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Converting input</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`age = int(input("Enter your age: "))        # Convert to int
price = float(input("Enter price: "))        # Convert to float
is_student = input("Are you a student? (y/n): ").lower() == 'y'`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Multiple inputs in one line:</span> Split and parse multiple values.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Multiple values</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Input: "Alice 25 Engineer"
data = input().split()
name = data[0]
age = int(data[1])
job = data[2]

# Using tuple unpacking
x, y = map(int, input("Enter two numbers: ").split())
print(f"Sum: {x + y}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Complete Example */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Complete Example
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Interactive program</div>
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Simple calculator
print("=== Simple Calculator ===")
num1 = float(input("Enter first number: "))
operator = input("Enter operator (+, -, *, /): ")
num2 = float(input("Enter second number: "))

if operator == '+':
    result = num1 + num2
elif operator == '-':
    result = num1 - num2
elif operator == '*':
    result = num1 * num2
elif operator == '/':
    result = num1 / num2 if num2 != 0 else "Error: Division by zero"
else:
    result = "Invalid operator"

print(f"{num1} {operator} {num2} = {result}")`}
            </pre>
          </div>
        </section>

        {/* Redirecting Output */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Redirecting Output (file parameter)
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">file</code> parameter lets you write to files instead of the console.
          </p>

          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`# Write to a file
with open("output.txt", "w") as f:
    print("Hello, file!", file=f)
    print("This goes to the file", file=f)

# Write to stderr (error output)
import sys
print("Error message", file=sys.stderr)`}
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
              <p className="font-semibold text-gray-900 dark:text-white mb-1">input() always returns a string</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Even if the user enters a number, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">input()</code> returns it as a string. Always convert using <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">int()</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">float()</code> for numeric operations.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Trailing newline in input</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">input()</code> strips the trailing newline but keeps other whitespace. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">.strip()</code> to remove leading/trailing spaces.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Print adds spaces between arguments</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print("a", "b")</code> outputs <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">"a b"</code>. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">sep=""</code> to remove spaces.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Input can raise EOFError</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                When reading from a file or pipe that ends unexpectedly, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">input()</code> raises <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">EOFError</code>. Handle with try/except if needed.
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
                What is the difference between <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print("a", "b")</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">print("a" + "b")</code>?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                First prints <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">"a b"</code> (space separated). Second concatenates strings first, printing <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">"ab"</code>.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                How do you read a number from user input?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">num = int(input("Enter a number: "))</code>. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">float()</code> for decimals. Wrap in try/except to handle invalid input.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">end=""</code> do in a print statement?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                It prevents the default newline from being added after printing. Useful for building output gradually on the same line.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                How can you read multiple integers from a single input line?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">a, b = map(int, input().split())</code>. This splits the input by whitespace and converts each part to an integer.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What is an f-string and why is it useful?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                f-strings (formatted string literals) allow embedding expressions directly in strings using <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">{}</code>. They're faster and more readable than <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">.format()</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">%</code>-formatting.
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/variables"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Variables & Data Types
          </Link>
          <Link
            href="/learn/programming-fundamentals/control-flow"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Control Flow →
          </Link>
        </div>
      </div>
    </div>
  );
}