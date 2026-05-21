import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb } from "lucide-react";

export default function BasicIOPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "basic-io");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python",
      label: "Python",
      code: `# Basic print output
print("Hello, World!")
print(42)
print(3.14159)

# Using sep parameter
print("apple", "banana", "cherry")           # apple banana cherry
print("apple", "banana", "cherry", sep=", ")  # apple, banana, cherry

# Using end parameter
print("First line")           # Ends with newline
print("Same", end=" ")         # Ends with space
print("line")                  # Continues on same line

# f-string formatting
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old")

# Basic input
user_input = input("Enter your name: ")
print(f"Hello {user_input}")

# Type conversion
age = int(input("Enter your age: "))
price = float(input("Enter price: "))

# Multiple inputs in one line
x, y = map(int, input("Enter two numbers: ").split())
print(f"Sum: {x + y}")`,
    },
    {
      language: "javascript",
      label: "JavaScript",
      code: `// Basic console output
console.log("Hello, World!");
console.log(42);
console.log(3.14159);

// Multiple arguments
console.log("apple", "banana", "cherry");

// Template literals (like f-strings)
const name = "Alice";
const age = 25;
console.log(\`My name is \${name} and I am \${age} years old\`);

// Basic input (Node.js with readline)
const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.question("Enter your name: ", (answer) => {
  console.log(\`Hello \${answer}\`);
  rl.close();
});

// Type conversion
const num = parseInt("42");
const floatNum = parseFloat("3.14");`,
    },
    {
      language: "java",
      label: "Java",
      code: `import java.util.Scanner;

public class Main {
  public static void main(String[] args) {
    // Basic output
    System.out.println("Hello, World!");
    System.out.println(42);
    System.out.print("No newline");
    
    // Formatted output
    String name = "Alice";
    int age = 25;
    System.out.printf("My name is %s and I am %d years old%n", name, age);
    
    // Basic input
    Scanner scanner = new Scanner(System.in);
    System.out.print("Enter your name: ");
    String input = scanner.nextLine();
    System.out.println("Hello " + input);
    
    // Type conversion
    System.out.print("Enter your age: ");
    int age2 = scanner.nextInt();
    
    scanner.close();
  }
}`,
    },
    {
      language: "cpp",
      label: "C++",
      code: `#include <iostream>
#include <string>
using namespace std;

int main() {
  // Basic output
  cout << "Hello, World!" << endl;
  cout << 42 << endl;
  cout << 3.14159 << endl;
  
  // Formatted output
  string name = "Alice";
  int age = 25;
  cout << "My name is " << name << " and I am " << age << " years old" << endl;
  
  // Basic input
  string userInput;
  cout << "Enter your name: ";
  getline(cin, userInput);
  cout << "Hello " << userInput << endl;
  
  // Type conversion
  int number;
  cout << "Enter a number: ";
  cin >> number;
  
  return 0;
}`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What does the `input()` function return in Python?",
      options: ["Integer", "Float", "String", "Boolean"],
      correctAnswer: 2,
      explanation: "The `input()` function always returns a string, regardless of what the user enters. You need to convert it using int() or float() for numeric operations.",
    },
    {
      id: 2,
      question: "What is the default value of the `sep` parameter in `print()`?",
      options: ["Comma (,)", "Space ( )", "Nothing ('')", "Newline (\\n)"],
      correctAnswer: 1,
      explanation: "The default separator between arguments in print() is a space. You can change it using sep parameter.",
    },
    {
      id: 3,
      question: "How do you prevent `print()` from adding a newline at the end?",
      options: ["Use newline=False", "Use end=''", "Use no_newline=True", "Use sep=''"],
      correctAnswer: 1,
      explanation: "Using end='' replaces the default newline character with an empty string, so subsequent output continues on the same line.",
    },
    {
      id: 4,
      question: "What is an f-string used for?",
      options: ["File input/output", "Function definitions", "String formatting with embedded expressions", "Finding substrings"],
      correctAnswer: 2,
      explanation: "f-strings (formatted string literals) allow you to embed expressions directly inside string literals using curly braces {}.",
    },
    {
      id: 5,
      question: "How do you read multiple integers from a single input line in Python?",
      options: ["input().split().to_int()", "int(input().split())", "map(int, input().split())", "input().split(int)"],
      correctAnswer: 2,
      explanation: "Use map(int, input().split()) to split the input and convert each part to an integer.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is Basic I/O?</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Input and Output (I/O)</strong> operations allow programs to interact with users. 
            Python provides <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">print()</code> for output and{' '}
            <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">input()</code> for user input, forming the foundation of interactive programming.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Think of I/O like a conversation: <strong>output (print)</strong> is like speaking to someone, 
                  while <strong>input</strong> is like listening and waiting for a response. The program speaks first, 
                  then listens, then responds based on what it heard.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Concepts */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Key I/O Concepts</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Standard Output (stdout)</h4>
              <p className="text-sm text-muted-foreground">
                The <code className="bg-muted px-1 py-0.5 rounded">print()</code> function writes to the console by default. 
                You can customize separators and line endings.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Standard Input (stdin)</h4>
              <p className="text-sm text-muted-foreground">
                The <code className="bg-muted px-1 py-0.5 rounded">input()</code> function reads user input as strings. 
                Always convert numeric input to appropriate types.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">String Formatting</h4>
              <p className="text-sm text-muted-foreground">
                f-strings offer the most readable way to embed expressions in strings. 
                Alternative methods include <code>.format()</code> and <code>%</code>-formatting.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-2 text-foreground">Error Output (stderr)</h4>
              <p className="text-sm text-muted-foreground">
                Use <code>file=sys.stderr</code> to print error messages separately from normal output.
              </p>
            </div>
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Code Examples</h2>
          <p className="text-muted-foreground mb-4">
            Here are examples of basic I/O operations across different programming languages:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Common Use Cases */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Use Cases</h2>
          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2"> Simple Calculator</h4>
              <pre className="bg-muted p-3 rounded-lg text-sm overflow-x-auto">
                {`# Get user input
num1 = float(input("Enter first number: "))
operator = input("Enter operator (+, -, *, /): ")
num2 = float(input("Enter second number: "))

# Perform calculation
if operator == '+':
    result = num1 + num2
elif operator == '-':
    result = num1 - num2
elif operator == '*':
    result = num1 * num2
elif operator == '/':
    result = num1 / num2 if num2 != 0 else "Error: Division by zero"

# Output result
print(f"{num1} {operator} {num2} = {result}")`}
              </pre>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2"> Data Collection Form</h4>
              <pre className="bg-muted p-3 rounded-lg text-sm overflow-x-auto">
                {`# Collect user information
name = input("Enter your name: ")
age = int(input("Enter your age: "))
email = input("Enter your email: ")

# Display formatted output
print("\\n=== User Profile ===")
print(f"Name: {name}")
print(f"Age: {age}")
print(f"Email: {email}")
print("===================")`}
              </pre>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2"> Progress Indicator</h4>
              <pre className="bg-muted p-3 rounded-lg text-sm overflow-x-auto">
                {`import time

# Print progress on same line
print("Loading", end="")
for i in range(5):
    print(".", end="", flush=True)
    time.sleep(0.5)
print(" Done!")`}
              </pre>
            </div>
          </div>
        </section>

        {/* Formatting Methods Comparison */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">String Formatting Methods</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Method</th>
                  <th className="border border-border p-3 text-left text-foreground">Syntax</th>
                  <th className="border border-border p-3 text-left text-foreground">When to Use</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-border p-3 font-mono text-sm text-foreground">f-strings (Python 3.6+)</td>
                <td className="border border-border p-3 font-mono text-sm text-primary">
  {"Hello World"}
</td>
                  <td className="border border-border p-3 text-muted-foreground text-sm">Most readable, fastest, recommended</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-mono text-sm text-foreground">.format()</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">"Hello {}".format(name)</td>
                  <td className="border border-border p-3 text-muted-foreground text-sm">Compatible with older Python versions</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-mono text-sm text-foreground">%-formatting</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">"Hello %s" % name</td>
                  <td className="border border-border p-3 text-muted-foreground text-sm">Legacy code, C-style formatting</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-mono text-sm text-foreground">Concatenation</td>
                  <td className="border border-border p-3 font-mono text-sm text-primary">"Hello " + name</td>
                  <td className="border border-border p-3 text-muted-foreground text-sm">Simple cases with few variables</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Mistakes to Avoid</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Forgetting Type Conversion</h4>
                <p className="text-sm text-muted-foreground">
                  <code className="bg-muted px-1 py-0.5 rounded">input()</code> returns a string. Always convert numeric input using 
                  <code className="bg-muted px-1 py-0.5 rounded">int()</code> or <code className="bg-muted px-1 py-0.5 rounded">float()</code> before math operations.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`# Wrong
age = input("Enter age: ")
next_age = age + 1  # TypeError!

# Correct
age = int(input("Enter age: "))
next_age = age + 1  # Works!`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Not Handling Invalid Input</h4>
                <p className="text-sm text-muted-foreground">
                  User input can be invalid (letters when expecting numbers). Use try/except for robust programs.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`try:
    age = int(input("Enter your age: "))
except ValueError:
    print("Please enter a valid number!")`}
                </pre>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Whitespace Issues</h4>
                <p className="text-sm text-muted-foreground">
                  Input may contain leading/trailing spaces. Use <code className="bg-muted px-1 py-0.5 rounded">.strip()</code> to clean input.
                </p>
                <pre className="mt-2 bg-muted p-2 rounded text-sm">
                  {`name = input("Enter name: ").strip()  # Removes extra spaces`}
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
                  f-strings are more readable, faster, and less error-prone than other formatting methods.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Provide Clear Prompts</h4>
                <p className="text-sm text-muted-foreground">
                  Always tell users what input you expect. Include expected format and valid ranges.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Validate Input</h4>
                <p className="text-sm text-muted-foreground">
                  Never trust user input. Always validate, convert, and handle errors appropriately.
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Descriptive Messages</h4>
                <p className="text-sm text-muted-foreground">
                  Error messages should explain what went wrong and how to fix it.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Basic I/O Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}