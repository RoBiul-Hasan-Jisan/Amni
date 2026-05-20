// app/learn/programming-fundamentals/loops/page.tsx

import Link from "next/link";

export default function LoopsPage() {
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
            Loops
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Loops repeat a block of code multiple times. Python provides <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">for</code> loops for iteration and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">while</code> loops for condition-based repetition.
          </p>
        </div>

        {/* for Loop */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            for Loop
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">for</code> loop iterates over any iterable (list, tuple, string, range, etc.).
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Iterating over a list</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">range() function:</span> Generate sequences of numbers.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">range(stop), range(start, stop), range(start, stop, step)</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# range(stop) - 0 to stop-1
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(start, stop) - start to stop-1
for i in range(2, 6):
    print(i)  # 2, 3, 4, 5

# range(start, stop, step) - with step value
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Count backwards
for i in range(5, 0, -1):
    print(i)  # 5, 4, 3, 2, 1`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Iterating over strings:</span> Strings are sequences of characters.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">String iteration</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`for char in "Python":
    print(char)  # P, y, t, h, o, n`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">enumerate():</span> Get both index and value while iterating.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Using enumerate</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`colors = ["red", "green", "blue"]
for index, color in enumerate(colors):
    print(f"{index}: {color}")

# Start index from 1
for index, color in enumerate(colors, start=1):
    print(f"{index}: {color}`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">zip():</span> Iterate over multiple sequences in parallel.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Parallel iteration</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# zip stops at shortest iterable
a = [1, 2, 3]
b = ["x", "y"]
for x, y in zip(a, b):
    print(x, y)  # (1, 'x'), (2, 'y')`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">Iterating over dictionaries:</span> Keys, values, or items.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Dictionary iteration</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`person = {"name": "Alice", "age": 30, "city": "New York"}

# Iterate over keys
for key in person:
    print(key)

# Iterate over values
for value in person.values():
    print(value)

# Iterate over key-value pairs
for key, value in person.items():
    print(f"{key}: {value}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* while Loop */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            while Loop
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">while</code> loop repeats as long as a condition is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">True</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Basic while loop</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`count = 0
while count < 5:
    print(count)
    count += 1`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">User input validation:</span> Common use case for while loops.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Input validation</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`while True:
    age = input("Enter your age: ")
    if age.isdigit() and 0 < int(age) < 120:
        age = int(age)
        break
    print("Invalid input. Please enter a valid age.")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Loop Control Statements */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Loop Control Statements
          </h2>

          <div className="space-y-5">
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">break:</span> Exit the loop immediately.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">continue:</span> Skip the rest of the current iteration and move to the next.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">else with loops:</span> Executes only if loop completes normally (no break).
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <div className="bg-gray-800 px-4 py-2 text-gray-300 font-mono text-sm">Loop else clause</div>
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Example: Check if number is prime
num = 7
for i in range(2, num):
    if num % i == 0:
        print(f"{num} is not prime")
        break
else:
    print(f"{num} is prime")  # Executes when no break

# else does NOT run if break occurs
for i in range(3):
    if i == 1:
        break
else:
    print("This won't print because loop broke")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2">
                <span className="font-semibold">pass:</span> Null statement - does nothing. Placeholder for future code.
              </p>
              <div className="bg-gray-900 rounded-lg overflow-hidden mt-2">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`for i in range(5):
    if i == 2:
        pass  # TODO: Add logic here
    print(i)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Nested Loops */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Nested Loops
          </h2>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i*j}")
    print("-" * 10)

# Matrix iteration
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # New line after each row`}
            </pre>
          </div>
        </section>

        {/* Loop Optimization */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Loop Optimization Tips
          </h2>
          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`# BAD: Avoid attribute lookup in loop
for i in range(len(my_list)):
    if my_list[i] > 0:  # my_list lookup each time
        pass

# GOOD: Cache length and use direct iteration
n = len(my_list)  # Cache once
for item in my_list:  # Direct iteration faster
    if item > 0:
        pass

# BAD: Expensive operation inside loop
for i in range(1000):
    result = expensive_function(i)  # Called 1000 times

# GOOD: Precompute if possible
results = [expensive_function(i) for i in range(1000)]  # Still 1000 calls but list comp faster

# Use local variables for speed
my_list = [1, 2, 3]
append = my_list.append  # Cache method lookup
for i in range(100):
    append(i)`}
              </pre>
            </div>
          </div>
        </section>

        {/* Common Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Common Loop Patterns
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Summing values:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`numbers = [1, 2, 3, 4, 5]
total = 0
for num in numbers:
    total += num
print(f"Sum: {total}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Finding maximum value:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`numbers = [3, 7, 2, 9, 5]
max_value = numbers[0]
for num in numbers:
    if num > max_value:
        max_value = num
print(f"Max: {max_value}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Filtering with condition:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = []
for num in numbers:
    if num % 2 == 0:
        evens.append(num)
print(f"Even numbers: {evens}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-gray-700 dark:text-gray-300 mb-2 font-semibold">Looping with index access:</p>
              <div className="bg-gray-900 rounded-lg overflow-hidden">
                <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                  {`# Using range and len
fruits = ["apple", "banana", "cherry"]
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

# Better: Use enumerate
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Infinite Loops */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Infinite Loops
          </h2>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-red-500 p-4 mb-4">
            <p className="font-semibold text-gray-900 dark:text-white mb-1">⚠️ Warning</p>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              Infinite loops never terminate and will freeze your program. Always ensure your loop condition becomes False at some point.
            </p>
          </div>
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`# Infinite loop (don't run without break)
# while True:
#     print("This runs forever")

# Controlled infinite loop with break
while True:
    user_input = input("Type 'quit' to exit: ")
    if user_input == 'quit':
        break
    print(f"You typed: {user_input}")`}
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
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Don't modify a list while iterating over it</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Modifying a list while iterating causes skipped or repeated elements. Iterate over a copy instead: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">for item in list[:]:</code>
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">range() doesn't create a list in Python 3</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">range()</code> returns a lazy sequence (range object), not a list. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list(range(10))</code> if you need an actual list.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">else clause executes after normal completion</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">else</code> block runs only if the loop wasn't terminated by <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code>. Useful for search operations.
              </p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Variable leakage in loops</p>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                Loop variables remain in scope after the loop ends. Unlike some languages, Python doesn't create a new scope for loops.
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
                What is the difference between <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">for</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">while</code> loops?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">for</code> is used when the number of iterations is known or when iterating over a sequence. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">while</code> is used when the number of iterations depends on a condition that may change during execution.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                When does the <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">else</code> clause in a loop execute?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">else</code> block executes when the loop completes normally (without encountering a <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code> statement). It does not execute if the loop is terminated by <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code>.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                How do you iterate over a list backwards?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">reversed()</code>: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">for item in reversed(my_list):</code> or slicing <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">my_list[::-1]</code>.
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                What's the difference between <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">pass</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">continue</code>, and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code>?
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code> exits the loop entirely. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">continue</code> skips to the next iteration. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">pass</code> does nothing (placeholder).
              </p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">
                Write a loop that prints all even numbers from 1 to 20.
              </p>
              <p className="text-gray-700 dark:text-gray-300">
                <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">for i in range(2, 21, 2): print(i)</code>
              </p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/control-flow"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Control Flow
          </Link>
          <Link
            href="/learn/programming-fundamentals/collections"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Collections →
          </Link>
        </div>
      </div>
    </div>
  );
}