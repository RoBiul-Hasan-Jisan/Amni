import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function ErrorHandlingPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "error-handling");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Exception Handling
          </h1>
          <p className="text-muted-foreground text-lg">
            <strong className="text-foreground">Exception handling</strong> allows you to respond to runtime errors gracefully. Instead of crashing, your program can catch, log, and recover from unexpected situations.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              <strong>Why handle exceptions?</strong> User input is unpredictable, files may be missing, network connections can fail. Exception handling makes your program robust and user-friendly instead of crashing with an ugly traceback.
            </p>
          </div>
        </div>

        {/* What are Exceptions? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What are Exceptions?
          </h2>
          <p className="text-muted-foreground mb-4">
            An <strong className="text-foreground">exception</strong> is an error that occurs during program execution. When an exception is raised, normal flow stops and Python looks for a handler.
          </p>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Common Exception</th>
                  <th className="text-left p-3 font-semibold text-foreground">When it occurs</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">ZeroDivisionError</td><td className="p-3">Division by zero</td><td className="p-3 font-mono">10 / 0</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">TypeError</td><td className="p-3">Wrong type for operation</td><td className="p-3 font-mono">"2" + 4</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">ValueError</td><td className="p-3">Correct type, invalid value</td><td className="p-3 font-mono">int("abc")</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">IndexError</td><td className="p-3">Index out of range</td><td className="p-3 font-mono">[1,2][5]</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">KeyError</td><td className="p-3">Dictionary key missing</td><td className="p-3 font-mono">{'{"a":1}'}["b"]</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">FileNotFoundError</td><td className="p-3">File doesn't exist</td><td className="p-3 font-mono">open("missing.txt")</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">AttributeError</td><td className="p-3">Attribute doesn't exist</td><td className="p-3 font-mono">"string".invalid()</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">ImportError</td><td className="p-3">Module not found</td><td className="p-3 font-mono">import nonexistent</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Basic try/except */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Basic try/except
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`try:
    num = int(input("Enter a number: "))
    result = 100 / num
    print(f"Result: {result}")
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("Can't divide by zero!")

# Multiple exceptions in one except
try:
    risky_operation()
except (ValueError, ZeroDivisionError) as e:
    print(f"An error occurred: {e}")`}
            </pre>
          </div>
        </section>

        {/* Catching All Exceptions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Catching All Exceptions
          </h2>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">⚠️ Warning</p>
                <p className="text-muted-foreground text-sm">
                  Catching <code className="bg-muted px-1.5 py-0.5 rounded">Exception</code> catches almost everything, including <code className="bg-muted px-1.5 py-0.5 rounded">KeyboardInterrupt</code> (Ctrl+C) and <code className="bg-muted px-1.5 py-0.5 rounded">SystemExit</code>. Avoid bare <code className="bg-muted px-1.5 py-0.5 rounded">except:</code> unless you have a very good reason.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`try:
    file = open("data.txt")
    content = file.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")
except Exception as e:
    # Catches any other exception
    print(f"Unexpected error: {e}")

# Bare except (not recommended)
try:
    dangerous()
except:  # Catches everything, even KeyboardInterrupt
    print("Something went wrong")`}
            </pre>
          </div>
        </section>

        {/* else Clause */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            else Clause
          </h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded">else</code> block runs <strong>only if no exception was raised</strong> in the <code className="bg-muted px-1.5 py-0.5 rounded">try</code> block.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`try:
    num = int(input("Enter a positive number: "))
except ValueError:
    print("Not a number")
else:
    # Runs only if conversion succeeded
    if num > 0:
        print(f"Square root: {num ** 0.5}")
    else:
        print("Number must be positive")

# Why use else? It separates success code from error-handling code
# Makes it clear which code expects exceptions and which doesn't`}
            </pre>
          </div>
        </section>

        {/* finally Clause */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            finally Clause
          </h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> block runs <strong>no matter what</strong> — whether an exception was raised or not, and even if there's a <code className="bg-muted px-1.5 py-0.5 rounded">return</code> or <code className="bg-muted px-1.5 py-0.5 rounded">break</code>. Perfect for cleanup (closing files, releasing resources).
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`try:
    file = open("data.txt", "r")
    content = file.read()
    print(content)
except FileNotFoundError:
    print("File not found")
finally:
    # Always closes the file if it was opened
    try:
        file.close()
    except NameError:
        pass  # file was never opened

# Better: use with statement (handles cleanup automatically)
with open("data.txt", "r") as file:
    content = file.read()`}
            </pre>
          </div>
        </section>

        {/* Raising Exceptions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Raising Exceptions
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <code className="bg-muted px-1.5 py-0.5 rounded">raise</code> to manually trigger an exception.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age too high (max 150)")
    return age

try:
    age = validate_age(-5)
except ValueError as e:
    print(f"Invalid age: {e}")

# Re-raise exception
try:
    risky_operation()
except ValueError:
    print("Logging error...")
    raise  # Re-raise the same exception`}
            </pre>
          </div>
        </section>

        {/* Custom Exceptions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Custom Exceptions
          </h2>
          <p className="text-muted-foreground mb-4">
            Create your own exception classes by inheriting from <code className="bg-muted px-1.5 py-0.5 rounded">Exception</code>.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class InsufficientFundsError(Exception):
    """Raised when account balance is insufficient."""
    pass

class NegativeAmountError(Exception):
    """Raised when amount is negative."""
    def __init__(self, amount, message="Amount cannot be negative"):
        self.amount = amount
        self.message = message
        super().__init__(self.message)

def withdraw(balance, amount):
    if amount < 0:
        raise NegativeAmountError(amount)
    if amount > balance:
        raise InsufficientFundsError(f"Need {amount}, have {balance}")
    return balance - amount

try:
    new_balance = withdraw(100, 200)
except InsufficientFundsError as e:
    print(f"Withdrawal failed: {e}")
except NegativeAmountError as e:
    print(f"Invalid amount: {e.amount}")`}
            </pre>
          </div>
        </section>

        {/* Exception Chaining */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Exception Chaining
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <code className="bg-muted px-1.5 py-0.5 rounded">from</code> to chain exceptions, preserving the original cause.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def load_config(filepath):
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"Config file {filepath} missing") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {filepath}") from e

try:
    config = load_config("config.json")
except RuntimeError as e:
    print(f"Error: {e}")
    print(f"Caused by: {e.__cause__}")`}
            </pre>
          </div>
        </section>

        {/* Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Best Practices
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">1. Be specific about exceptions</p>
                  <p className="text-muted-foreground">Catch specific exceptions rather than a bare <code className="bg-muted px-1.5 py-0.5 rounded">except:</code>. This prevents hiding unexpected bugs.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">2. Use <code className="bg-muted px-1.5 py-0.5 rounded">else</code> for success code</p>
                  <p className="text-muted-foreground">Put code that expects no exceptions in <code className="bg-muted px-1.5 py-0.5 rounded">else</code> to separate success from error handling.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">3. Always close resources</p>
                  <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> or the <code className="bg-muted px-1.5 py-0.5 rounded">with</code> statement to ensure resources are cleaned up.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">4. Don't swallow exceptions silently</p>
                  <p className="text-muted-foreground">Always log or report exceptions; silent failures make debugging impossible.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Tricky Points */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">else vs finally</p>
                  <p className="text-muted-foreground text-sm"><code className="bg-muted px-1.5 py-0.5 rounded">else</code> runs only on success; <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> always runs. <code className="bg-muted px-1.5 py-0.5 rounded">else</code> is skipped if there's an exception; <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> runs even after <code className="bg-muted px-1.5 py-0.5 rounded">return</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Exception in <code className="bg-muted px-1.5 py-0.5 rounded">except</code> or <code className="bg-muted px-1.5 py-0.5 rounded">else</code></p>
                  <p className="text-muted-foreground text-sm">Exceptions raised inside <code className="bg-muted px-1.5 py-0.5 rounded">except</code> or <code className="bg-muted px-1.5 py-0.5 rounded">else</code> are not caught by that <code className="bg-muted px-1.5 py-0.5 rounded">except</code> block. Use nested <code className="bg-muted px-1.5 py-0.5 rounded">try</code> or handle separately.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Bare except kills KeyboardInterrupt</p>
                  <p className="text-muted-foreground text-sm"><code className="bg-muted px-1.5 py-0.5 rounded">except:</code> (without exception type) catches <code className="bg-muted px-1.5 py-0.5 rounded">KeyboardInterrupt</code>, making Ctrl+C unable to stop your program.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Interview Questions
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between <code className="bg-muted px-1.5 py-0.5 rounded">except:</code> and <code className="bg-muted px-1.5 py-0.5 rounded">except Exception:</code>?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">except:</code> catches <strong>all</strong> exceptions including <code className="bg-muted px-1.5 py-0.5 rounded">KeyboardInterrupt</code> and <code className="bg-muted px-1.5 py-0.5 rounded">SystemExit</code>. <code className="bg-muted px-1.5 py-0.5 rounded">except Exception:</code> catches only exceptions that inherit from <code className="bg-muted px-1.5 py-0.5 rounded">Exception</code> (most normal errors). Avoid bare <code className="bg-muted px-1.5 py-0.5 rounded">except:</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">When does <code className="bg-muted px-1.5 py-0.5 rounded">else</code> execute in a try/except block?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">else</code> executes only if the <code className="bg-muted px-1.5 py-0.5 rounded">try</code> block completes without raising an exception. It's useful for code that should only run on success.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Does <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> run if there's a <code className="bg-muted px-1.5 py-0.5 rounded">return</code> in <code className="bg-muted px-1.5 py-0.5 rounded">try</code>?</p>
                  <p className="text-muted-foreground">Yes. <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> runs even after a <code className="bg-muted px-1.5 py-0.5 rounded">return</code>, <code className="bg-muted px-1.5 py-0.5 rounded">break</code>, or <code className="bg-muted px-1.5 py-0.5 rounded">continue</code>. The <code className="bg-muted px-1.5 py-0.5 rounded">finally</code> block executes before the function actually returns.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How do you create a custom exception?</p>
                  <p className="text-muted-foreground">Create a class that inherits from <code className="bg-muted px-1.5 py-0.5 rounded">Exception</code>. Add a docstring and optionally override <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code> to accept custom arguments.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is exception chaining and how do you do it?</p>
                  <p className="text-muted-foreground">Exception chaining preserves the original exception when raising a new one. Use <code className="bg-muted px-1.5 py-0.5 rounded">raise NewException() from original_exception</code>. The original is accessible via <code className="bg-muted px-1.5 py-0.5 rounded">e.__cause__</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

    
      </div>
    </TopicContent>
  );
}