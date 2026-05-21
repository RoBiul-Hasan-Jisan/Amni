import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb } from "lucide-react";

export default function HelloWorldPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "hello-world");
  if (!result) return null;

  const { topic, subtopic } = result;

 const codeExamples = [
  {
    language: "python",
    label: "Python",
    code: `# Basic Hello World
print("Hello, World!")

# Multiple arguments
print("Hello", "World", "!")

# Custom separator
print("2", "4", "6", sep="-")

# Custom end character
print("Line one", end="")
print("Line two")

# With variables
name = "Alice"
print(f"Hello, {name}!")`,
  },
  {
    language: "javascript",
    label: "JavaScript",
    code: `// Basic Hello World
console.log("Hello, World!");

// Multiple arguments
console.log("Hello", "World", "!");

// Template literals (escaped backticks)
const name = "Alice";
console.log("Hello, " + name + "!");

// No newline (Node.js)
process.stdout.write("Line one");
process.stdout.write("Line two");`,
  },
  {
    language: "java",
    label: "Java",
    code: `public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World!");

    System.out.print("Line one");
    System.out.print("Line two");

    String name = "Alice";
    System.out.printf("Hello, %s!%n", name);
  }
}`,
  },
  {
    language: "cpp",
    label: "C++",
    code: `#include <iostream>
using namespace std;

int main() {
  cout << "Hello, World!" << endl;
  cout << "Hello" << " " << "World" << "!" << endl;

  cout << "Line one";
  cout << "Line two";

  string name = "Alice";
  cout << "Hello, " << name << "!" << endl;

  return 0;
}`,
  },
];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What does the `print()` function do in Python?",
      options: [
        "Reads input from the user",
        "Writes text to standard output",
        "Creates a new variable",
        "Opens a file"
      ],
      correctAnswer: 1,
      explanation: "The `print()` function outputs text to the console (standard output), allowing programs to communicate with users.",
    },
    {
      id: 2,
      question: "What is the default separator between arguments in `print()`?",
      options: ["Comma (,)", "Space ( )", "Nothing ('')", "Hyphen (-)"],
      correctAnswer: 1,
      explanation: "By default, `print()` separates multiple arguments with a space. You can change this using the `sep` parameter.",
    },
    {
      id: 3,
      question: "How do you prevent `print()` from adding a newline at the end?",
      options: ["Use newline=False", "Use end=''", "Use no_newline=True", "Use sep=''"],
      correctAnswer: 1,
      explanation: "Using `end=''` replaces the default newline character with an empty string, so subsequent output continues on the same line.",
    },
    {
      id: 4,
      question: "What happens when you call `print()` with no arguments?",
      options: [
        "Nothing happens",
        "It prints an empty string",
        "It prints a blank line",
        "It raises an error"
      ],
      correctAnswer: 2,
      explanation: "`print()` with no arguments prints a blank line (just a newline character). This is useful for adding vertical spacing.",
    },
    {
      id: 5,
      question: "Which of the following is the correct way to write a comment in Python?",
      options: [
        "// This is a comment",
        "<!-- This is a comment -->",
        "# This is a comment",
        "/* This is a comment */"
      ],
      correctAnswer: 2,
      explanation: "In Python, comments start with the `#` symbol and continue to the end of the line.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is Hello World?</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">"Hello, World!"</strong> is traditionally the first program written when 
            learning a new programming language. It simply prints a message to the console, confirming that your 
            development environment is set up correctly and that you can run code.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Why "Hello World"?</h4>
                <p className="text-sm text-muted-foreground">
                  The tradition started with the 1978 book "The C Programming Language" by Kernighan and Ritchie. 
                  It's a simple, confidence-building first step that proves your development environment works 
                  and you can execute code successfully.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* The print() Function */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">The print() Function</h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">print()</code> function sends text to standard output (usually the console). 
            It accepts one or more arguments and automatically converts them to strings before displaying.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Basic Usage</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`print("Hello, World!")
print(42)
print(3.14159)`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Multiple Arguments</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`print("Hello", "World")
# Output: Hello World

print("a", "b", "c", sep="-")
# Output: a-b-c`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Custom End Character</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`print("Same", end="")
print("line")
# Output: Same line`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Print Empty Line</h4>
              <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
                {`print()  # Prints blank line
print("After gap")`}
              </pre>
            </div>
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Code Examples Across Languages</h2>
          <p className="text-muted-foreground mb-4">
            See how "Hello, World!" and basic output operations look in different programming languages:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Python Syntax Basics */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Python Syntax Basics</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Indentation</h4>
              <p className="text-sm text-muted-foreground">
                Python uses indentation to define code blocks. Four spaces per level is the convention.
              </p>
              <pre className="bg-muted p-2 rounded text-xs mt-2">
                {`if True:
    print("Indented!")`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Comments</h4>
              <p className="text-sm text-muted-foreground">
                Lines starting with <code className="bg-muted px-1 rounded">#</code> are comments and are ignored.
              </p>
              <pre className="bg-muted p-2 rounded text-xs mt-2">
                {`# This is a comment
print("Hello")  # Inline comment`}
              </pre>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Colon</h4>
              <p className="text-sm text-muted-foreground">
                A colon <code className="bg-muted px-1 rounded">:</code> starts a block. The following lines must be indented.
              </p>
              <pre className="bg-muted p-2 rounded text-xs mt-2">
                {`if condition:
    # Block starts here
    print("Inside block")`}
              </pre>
            </div>
          </div>
        </section>

        {/* Common Patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Usage Patterns</h2>
          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2"> Simple Greeting</h4>
              <pre className="bg-muted p-3 rounded-lg text-sm overflow-x-auto">
                {`name = "Alice"
print(f"Hello, {name}! Welcome to Python.")`}
              </pre>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2"> Progress Indicator</h4>
              <pre className="bg-muted p-3 rounded-lg text-sm overflow-x-auto">
                {`import time

print("Loading", end="")
for i in range(3):
    print(".", end="", flush=True)
    time.sleep(0.5)
print(" Done!")`}
              </pre>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2"> Formatted Table</h4>
              <pre className="bg-muted p-3 rounded-lg text-sm overflow-x-auto">
                {`print("Name".ljust(10) + "Score".rjust(10))
print("-" * 20)
print("Alice".ljust(10) + "95".rjust(10))
print("Bob".ljust(10) + "87".rjust(10))`}
              </pre>
            </div>
          </div>
        </section>

        {/* Time & Space Complexity Note */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Performance Considerations</h2>
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Clock className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">I/O Operations are Expensive</h4>
                <p className="text-sm text-muted-foreground">
                  While <code className="bg-muted px-1 rounded">print()</code> is fine for learning and debugging, 
                  frequent console output in production can slow down your program. Each I/O operation is costly 
                  compared to in-memory operations. For performance-critical applications, minimize console output 
                  or use logging with appropriate levels.
                </p>
                <div className="mt-3 p-3 bg-muted rounded text-sm">
                  <p className="font-mono text-xs"># Bad for performance in loops:</p>
                  <p className="font-mono text-xs text-muted-foreground">for i in range(10000): print(i)</p>
                  <p className="font-mono text-xs mt-2"># Better: Collect and print once</p>
                  <p className="font-mono text-xs text-muted-foreground">result = [str(i) for i in range(10000)]</p>
                  <p className="font-mono text-xs text-muted-foreground">print("\n".join(result))</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Mistakes to Avoid</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Missing Parentheses</h4>
                <p className="text-sm text-muted-foreground">
                  In Python 3, <code className="bg-muted px-1 rounded">print</code> is a function and requires parentheses.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong (Python 2 style)
print "Hello"

# Correct
print("Hello")`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Quote Mismatch</h4>
                <p className="text-sm text-muted-foreground">
                  Strings must start and end with matching quotes.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong
print("Hello')  # Mixed quotes

# Correct
print("Hello")   # Double quotes
print('Hello')   # Single quotes`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Indentation Errors</h4>
                <p className="text-sm text-muted-foreground">
                  Python requires consistent indentation. Mixing tabs and spaces causes <code className="bg-muted px-1 rounded">IndentationError</code>.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong - inconsistent indentation
if True:
  print("Two spaces")
    print("Four spaces")

# Correct - consistent
if True:
    print("Four spaces")
    print("Also four spaces")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Best Practices */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Best Practices</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use f-strings for Formatting</h4>
                <p className="text-sm text-muted-foreground">
                  Modern Python (3.6+) supports f-strings, which are cleaner and faster than other formatting methods.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`name = "Alice"
score = 95
print(f"{name} scored {score}%")  # Recommended`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Add Descriptive Messages</h4>
                <p className="text-sm text-muted-foreground">
                  When printing for users, add clear context to your output.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Good
print(f"Processing file: {filename}")

# Bad
print(filename)`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use flush for Real-time Output</h4>
                <p className="text-sm text-muted-foreground">
                  Use <code className="bg-muted px-1 rounded">flush=True</code> to force immediate output, especially in progress indicators.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`for i in range(10):
    print(f"\rProgress: {i}/10", end="", flush=True)
    time.sleep(0.1)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Hello World Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}