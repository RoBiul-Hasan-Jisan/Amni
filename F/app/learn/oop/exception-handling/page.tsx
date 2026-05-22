"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
  FileQuestion,
  Shield,
  AlertCircle,
  XCircle,
  CheckCircle,
  Flag,
} from "lucide-react";

const javaExceptionCode = `// Exception Handling in Java

import java.io.*;

public class ExceptionDemo {
    
    // Method declaring that it throws an exception
    public static void readFile(String filename) throws IOException {
        FileReader file = new FileReader(filename);
        BufferedReader br = new BufferedReader(file);
        System.out.println(br.readLine());
        br.close();
    }
    
    public static void main(String[] args) {
        // Try-catch-finally
        try {
            int[] arr = new int[5];
            arr[10] = 100;  // ArrayIndexOutOfBoundsException
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Array index error: " + e.getMessage());
        } finally {
            System.out.println("This always executes");
        }
        
        // Multiple catch blocks
        try {
            String str = null;
            System.out.println(str.length());  // NullPointerException
        } catch (NullPointerException e) {
            System.out.println("Null pointer: " + e);
        } catch (Exception e) {
            System.out.println("Generic exception: " + e);
        }
        
        // Using throws
        try {
            readFile("nonexistent.txt");
        } catch (IOException e) {
            System.out.println("IO Error: " + e.getMessage());
        }
        
        // Custom exception
        try {
            validateAge(15);
        } catch (InvalidAgeException e) {
            System.out.println("Custom exception: " + e.getMessage());
        }
    }
    
    public static void validateAge(int age) throws InvalidAgeException {
        if (age < 18) {
            throw new InvalidAgeException("Age must be 18 or above");
        }
        System.out.println("Valid age: " + age);
    }
}

// Custom Exception
class InvalidAgeException extends Exception {
    public InvalidAgeException(String message) {
        super(message);
    }
}`;

const cppExceptionCode = `// Exception Handling in C++

#include <iostream>
#include <exception>
#include <stdexcept>
using namespace std;

// Custom exception class
class InvalidAgeException : public exception {
private:
    string message;
public:
    InvalidAgeException(const string& msg) : message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
};

class BankAccount {
private:
    double balance;
public:
    BankAccount(double initial) : balance(initial) {}
    
    void withdraw(double amount) {
        if (amount > balance) {
            throw runtime_error("Insufficient balance!");
        }
        if (amount < 0) {
            throw invalid_argument("Amount cannot be negative!");
        }
        balance -= amount;
        cout << "Withdrawn: " << amount << ", Remaining: " << balance << endl;
    }
    
    void display() {
        cout << "Balance: " << balance << endl;
    }
};

double divide(int a, int b) {
    if (b == 0) {
        throw "Division by zero!";
    }
    return (double)a / b;
}

int main() {
    // Basic try-catch
    try {
        cout << divide(10, 2) << endl;
        cout << divide(10, 0) << endl;
    } catch (const char* msg) {
        cerr << "Error: " << msg << endl;
    }
    
    // Multiple catch blocks
    BankAccount account(1000);
    try {
        account.withdraw(500);
        account.withdraw(600);  // This will throw
        account.withdraw(-50);   // This will throw
    } catch (const runtime_error& e) {
        cerr << "Runtime error: " << e.what() << endl;
    } catch (const invalid_argument& e) {
        cerr << "Invalid argument: " << e.what() << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }
    
    // Custom exception
    try {
        int age;
        cout << "Enter age: ";
        cin >> age;
        if (age < 18) {
            throw InvalidAgeException("Age must be 18 or above");
        }
        cout << "Valid age: " << age << endl;
    } catch (const InvalidAgeException& e) {
        cerr << "Custom exception: " << e.what() << endl;
    }
    
    account.display();
    return 0;
}`;

const pythonExceptionCode = `# Exception Handling in Python

class InvalidAgeException(Exception):
    """Custom exception class"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError(f"Insufficient balance! Balance: {self.balance}")
        if amount < 0:
            raise ValueError("Amount cannot be negative!")
        self.balance -= amount
        print(f"Withdrawn: {amount}, Remaining: {self.balance}")

def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    else:
        print(f"Division successful: {result}")
        return result
    finally:
        print("Division operation completed")

def validate_age(age):
    if age < 18:
        raise InvalidAgeException(f"Age {age} is below 18")
    print(f"Valid age: {age}")

def main():
    # Basic exception handling
    try:
        num = int(input("Enter a number: "))
        result = 10 / num
        print(f"Result: {result}")
    except ValueError:
        print("Error: Please enter a valid number")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
    except Exception as e:
        print(f"Unexpected error: {e}")
    else:
        print("No exceptions occurred")
    finally:
        print("This always executes")
    
    # Multiple exceptions
    account = BankAccount(1000)
    try:
        account.withdraw(500)
        account.withdraw(600)  # This will raise error
    except ValueError as e:
        print(f"Error: {e}")
    
    # Custom exception
    try:
        age = int(input("Enter age: "))
        validate_age(age)
    except InvalidAgeException as e:
        print(f"Custom exception: {e}")
    except ValueError:
        print("Please enter a valid number")

if __name__ == "__main__":
    main()`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is exception handling?",
    options: [
      "A way to ignore errors",
      "A mechanism to handle runtime errors gracefully",
      "A method to optimize code",
      "A type of inheritance",
    ],
    correctAnswer: 1,
    explanation: "Exception handling is a mechanism to detect and handle runtime errors without terminating the program abruptly.",
  },
  {
    id: 2,
    question: "What is the purpose of the finally block?",
    options: [
      "To catch exceptions",
      "To throw exceptions",
      "To execute code regardless of whether an exception occurs",
      "To declare exceptions",
    ],
    correctAnswer: 2,
    explanation: "The finally block always executes whether an exception occurs or not, typically used for cleanup operations like closing files.",
  },
];

// Question Card Component
interface QuestionCardProps {
  number: number;
  title: string;
  question: string;
  answer: string;
  marks: number;
  children: React.ReactNode;
  icon?: React.ReactNode;
}

const QuestionCard: React.FC<QuestionCardProps> = ({ 
  number, 
  title, 
  question, 
  answer, 
  marks, 
  children,
  icon 
}) => {
  return (
    <div className="mb-8 rounded-lg border border-border bg-card overflow-hidden">
      <div className="bg-primary/10 p-4 border-b border-border">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
              {number}
            </div>
            <h3 className="text-lg font-semibold text-foreground">{title}</h3>
          </div>
          <div className="bg-primary/20 text-primary px-3 py-1 rounded-full text-sm font-medium">
            {marks} Marks
          </div>
        </div>
      </div>
      
      <div className="p-5">
        <div className="mb-4">
          <div className="flex items-start gap-2">
            <FileQuestion className="h-5 w-5 text-primary mt-0.5 shrink-0" />
            <div>
              <span className="text-sm font-medium text-muted-foreground">Question:</span>
              <p className="text-foreground font-medium mt-1">{question}</p>
            </div>
          </div>
        </div>
        
        <div className="mb-4">
          <div className="flex items-start gap-2">
            <div className="w-5 h-5 rounded-full bg-green-500/20 text-green-500 flex items-center justify-center text-xs font-bold mt-0.5 shrink-0">
              ✓
            </div>
            <div className="flex-1">
              <span className="text-sm font-medium text-muted-foreground">Answer:</span>
              <div className="text-foreground mt-1 space-y-2">
                <p>{answer}</p>
              </div>
            </div>
          </div>
        </div>
        
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Code className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-muted-foreground">Example:</span>
          </div>
          <div className="rounded-lg overflow-hidden border border-border">
            {children}
          </div>
        </div>
        
        {icon && (
          <div className="mt-4 pt-3 border-t border-border flex justify-end">
            <div className="text-xs text-muted-foreground flex items-center gap-1">
              {icon}
              <span>Key Concept Highlighted</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Difference Table Component
const DifferenceTable: React.FC<{
  title: string;
  headers: string[];
  rows: Array<Array<string | React.ReactNode>>;
}> = ({ title, headers, rows }) => {
  return (
  <div className="mb-6">
    <h4 className="font-medium text-foreground mb-3">{title}</h4>

    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-muted">
            {headers.map((header, idx) => (
              <th
                key={idx}
                className="border border-border p-2 text-left font-semibold text-foreground"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} className="even:bg-muted/30">
              {row.map((cell, cellIdx) => (
                <td
                  key={cellIdx}
                  className="border border-border p-2 text-muted-foreground"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);
};

export default function ExceptionHandlingPage() {
  const result = getSubtopicBySlug("oop", "exception-handling");

  if (!result) {
    return (
      <div className="p-8 text-center text-muted-foreground">
        Topic not found
      </div>
    );
  }

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <BookOpen className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            What is Exception Handling?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Exception handling</strong> is a mechanism to handle 
            runtime errors gracefully without crashing the program. It separates error-handling code 
            from regular code, making programs more robust and maintainable.
          </p>
        </div>
      </section>

      {/* Code Implementation */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Code className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            Code Implementation
          </h2>
        </div>
        <MultiLanguageCode
          codes={[
            { language: "java", label: "Java", code: javaExceptionCode },
            { language: "cpp", label: "C++", code: cppExceptionCode },
            { language: "python", label: "Python", code: pythonExceptionCode },
          ]}
        />
      </section>

      {/* Practice Questions */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-primary/10">
            <Target className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            Practice Questions (5 Marks Each)
          </h2>
        </div>

        {/* Question 40: Exception Handling Basics */}
        <QuestionCard
          number={40}
          title="Exception Handling - try, catch, throw, finally"
          question="What is exception handling? Explain try, catch, throw, finally with examples."
          answer="Exception handling is a mechanism to handle runtime errors. Keywords: try - contains code that might throw exception; catch - handles specific exception types; throw - explicitly throws an exception; finally - always executes (cleanup). This separates error handling from normal code flow."
          marks={5}
          icon={<Shield className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class ExceptionBasics {
    public static void main(String[] args) {
        try {
            // Code that might throw exception
            int result = divide(10, 0);
            System.out.println("Result: " + result);
        } 
        catch (ArithmeticException e) {
            // Handle specific exception
            System.out.println("Cannot divide by zero!");
            System.out.println("Exception: " + e.getMessage());
        }
        catch (Exception e) {
            // Handle any other exception
            System.out.println("Some error occurred: " + e);
        }
        finally {
            // Always executes
            System.out.println("Cleanup - close files, connections");
        }
        
        // Using throw
        try {
            validateAge(15);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
        }
    }
    
    static int divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return a / b;
    }
    
    static void validateAge(int age) {
        if (age < 18) {
            throw new IllegalArgumentException("Age must be 18+");
        }
        System.out.println("Valid age: " + age);
    }
}

/* Output:
Cannot divide by zero!
Cleanup - close files, connections
Age must be 18+
*/`}
          />
        </QuestionCard>

        {/* Question 41: Checked vs Unchecked Exceptions */}
        <QuestionCard
          number={41}
          title="Checked vs Unchecked Exceptions"
          question="Difference between checked and unchecked exceptions in Java."
          answer="Checked exceptions are checked at compile-time (e.g., IOException, SQLException). The compiler forces you to handle or declare them. Unchecked exceptions (RuntimeException and subclasses) are checked at runtime (e.g., NullPointerException, ArrayIndexOutOfBoundsException). Unchecked exceptions are programmer errors and don't require mandatory handling."
          marks={5}
          icon={<AlertCircle className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Checked vs Unchecked Exceptions"
            headers={["Feature", "Checked Exception", "Unchecked Exception"]}
            rows={[
              ["Checked at", "Compile time", "Runtime"],
              ["Must handle?", "Yes (try-catch or throws)", "No (optional)"],
              ["Parent class", "Exception (except RuntimeException)", "RuntimeException"],
              ["Examples", "IOException, SQLException", "NullPointerException, ArrayIndexOutOfBoundsException"],
              ["Recovery", "Often recoverable", "Usually programming error"],
            ]}
          />
          <CodeBlock
            language="java"
            code={`import java.io.*;

public class CheckedVsUnchecked {
    
    // CHECKED EXCEPTION - Must handle or declare
    public void readFile() {
        // Compiler error if not handled!
        // FileReader fr = new FileReader("file.txt"); // ERROR
        
        // Must handle checked exception
        try {
            FileReader fr = new FileReader("file.txt");
            BufferedReader br = new BufferedReader(fr);
            System.out.println(br.readLine());
            br.close();
        } catch (IOException e) {
            System.out.println("Handled checked exception: " + e);
        }
    }
    
    // Or declare with throws
    public void readFile2() throws IOException {
        FileReader fr = new FileReader("file.txt");
        BufferedReader br = new BufferedReader(fr);
        System.out.println(br.readLine());
        br.close();
    }
    
    // UNCHECKED EXCEPTION - No need to handle
    public void causeNullPointer() {
        String str = null;
        // Compiles fine, but throws at runtime
        System.out.println(str.length()); // NullPointerException at runtime
    }
    
    public void causeArrayIndexError() {
        int[] arr = new int[5];
        // Compiles fine, but throws at runtime
        System.out.println(arr[10]); // ArrayIndexOutOfBoundsException
    }
    
    public static void main(String[] args) {
        CheckedVsUnchecked demo = new CheckedVsUnchecked();
        demo.readFile();  // Handled
        // demo.causeNullPointer(); // Unhandled but compiles
    }
}`}
          />
        </QuestionCard>

        {/* Question 42: Multiple Catch Blocks */}
        <QuestionCard
          number={42}
          title="Multiple Catch Blocks"
          question="Can we have multiple catch blocks? Write an example."
          answer="Yes, we can have multiple catch blocks to handle different types of exceptions separately. The order matters - catch more specific exceptions first, then general ones. Java 7+ allows multi-catch: catch (IOException | SQLException e)."
          marks={5}
          icon={<Flag className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class MultipleCatchDemo {
    public static void processInput(String input) {
        try {
            // Multiple potential exceptions
            int num = Integer.parseInt(input);
            int result = 100 / num;
            int[] arr = new int[5];
            arr[num] = result;
            
            System.out.println("Result: " + result);
        } 
        // Order matters - specific first
        catch (ArithmeticException e) {
            System.out.println("Arithmetic: Cannot divide by zero");
        }
        catch (NumberFormatException e) {
            System.out.println("Number Format: Invalid number format");
        }
        catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Array Index: Index out of bounds");
        }
        catch (Exception e) {
            System.out.println("Generic: " + e.getMessage());
        }
    }
    
    // Java 7+ multi-catch
    public static void multiCatch(String input) {
        try {
            int num = Integer.parseInt(input);
            int result = 100 / num;
        } 
        catch (ArithmeticException | NumberFormatException e) {
            System.out.println("Arithmetic or Number format error: " + e);
        }
    }
    
    public static void main(String[] args) {
        System.out.println("=== Testing multiple catch blocks ===");
        
        System.out.println("\\nTest 1: Valid input");
        processInput("5");
        
        System.out.println("\\nTest 2: Division by zero");
        processInput("0");
        
        System.out.println("\\nTest 3: Invalid number");
        processInput("abc");
        
        System.out.println("\\nTest 4: Array index");
        processInput("10");
    }
}

/* Output:
=== Testing multiple catch blocks ===

Test 1: Valid input
Result: 20

Test 2: Division by zero
Arithmetic: Cannot divide by zero

Test 3: Invalid number
Number Format: Invalid number format

Test 4: Array index
Array Index: Index out of bounds
*/`}
          />
        </QuestionCard>

        {/* Question 43: Finally Block */}
        <QuestionCard
          number={43}
          title="Finally Block Purpose"
          question="What is the purpose of finally block? Does it always execute?"
          answer="The finally block is used for cleanup operations like closing files, database connections, or releasing resources. It ALWAYS executes regardless of whether an exception occurs or not, with very few exceptions: System.exit(), JVM crash, or infinite loop before finally. Finally is optional but recommended for resource management."
          marks={5}
          icon={<CheckCircle className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">When does finally NOT execute?</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>System.exit() called</li>
              <li>JVM crashes</li>
              <li>Infinite loop before finally</li>
              <li>Power failure / system shutdown</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`import java.io.*;

public class FinallyDemo {
    
    public static void testWithException() {
        try {
            System.out.println("Inside try");
            int result = 10 / 0;
        } catch (ArithmeticException e) {
            System.out.println("Inside catch");
            return; // Even with return, finally executes!
        } finally {
            System.out.println("Inside finally - ALWAYS executes");
        }
        System.out.println("After try-catch-finally");
    }
    
    public static void testWithoutException() {
        try {
            System.out.println("\\nInside try - no exception");
        } finally {
            System.out.println("Finally executes even without catch");
        }
    }
    
    public static void testResourceManagement() {
        FileReader fr = null;
        try {
            fr = new FileReader("test.txt");
            // Read file
        } catch (IOException e) {
            System.out.println("Error reading file: " + e);
        } finally {
            // Cleanup - always executed
            try {
                if (fr != null) {
                    fr.close();
                    System.out.println("File closed in finally");
                }
            } catch (IOException e) {
                System.out.println("Error closing file: " + e);
            }
        }
    }
    
    // Java 7+ try-with-resources (automatic cleanup)
    public static void tryWithResources() {
        try (FileReader fr = new FileReader("test.txt");
             BufferedReader br = new BufferedReader(fr)) {
            System.out.println(br.readLine());
            // Automatically closes resources
        } catch (IOException e) {
            System.out.println("Error: " + e);
        }
    }
    
    public static void main(String[] args) {
        testWithException();
        testWithoutException();
    }
}`}
          />
        </QuestionCard>

        {/* Question 44: Throws Keyword */}
        <QuestionCard
          number={44}
          title="Throws Keyword"
          question="What is throws keyword? Explain with example."
          answer="The 'throws' keyword is used in method signatures to declare that a method may throw one or more exceptions. It delegates exception handling to the calling method. This is mandatory for checked exceptions in Java. Multiple exceptions can be declared separated by commas. 'throws' doesn't throw exceptions - it just declares the possibility."
          marks={5}
          icon={<Flag className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`import java.io.*;

public class ThrowsDemo {
    
    // Method declares that it may throw IOException
    // Caller must handle or declare further
    public static void readFile(String filename) throws IOException {
        FileReader fr = new FileReader(filename);
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        System.out.println("First line: " + line);
        br.close();
    }
    
    // Multiple exceptions
    public static void processData(String data) 
            throws IOException, NumberFormatException, ArithmeticException {
        
        int num = Integer.parseInt(data);
        int result = 100 / num;
        System.out.println("Result: " + result);
    }
    
    // Method that calls another throws method
    // Can handle or declare further
    public static void safeReadFile(String filename) {
        try {
            readFile(filename);
        } catch (IOException e) {
            System.out.println("Handled in safeReadFile: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        // Caller must handle or declare
        try {
            readFile("test.txt");
        } catch (IOException e) {
            System.out.println("Handled in main: " + e.getMessage());
        }
        
        // Using safe wrapper
        safeReadFile("test.txt");
        
        // Multiple exceptions
        try {
            processData("10");
            processData("0");   // Throws ArithmeticException
            processData("abc"); // Throws NumberFormatException
        } catch (IOException | NumberFormatException | ArithmeticException e) {
            System.out.println("Caught: " + e.getClass().getSimpleName());
        }
    }
}

// Rule: Overriding methods cannot throw broader checked exceptions
class Parent {
    void method() throws IOException {}
}

class Child extends Parent {
    // OK - same exception
    void method() throws IOException {}
    
    // OK - more specific (subclass)
    void method2() throws FileNotFoundException {}
    
    // ERROR - broader exception
    // void method3() throws Exception {}  // Compile error!
}`}
          />
        </QuestionCard>

        {/* Question 45: Custom Exception */}
        <QuestionCard
          number={45}
          title="Custom Exception"
          question="Write a class with a custom exception (user-defined exception)."
          answer="Custom exceptions are created by extending Exception (checked) or RuntimeException (unchecked). They allow application-specific error handling with meaningful names and custom messages. Provide constructors to set error messages and optionally cause. Custom exceptions improve code readability and maintainability."
          marks={5}
          icon={<XCircle className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Custom Checked Exception
class InsufficientFundsException extends Exception {
    private double amount;
    private double balance;
    
    public InsufficientFundsException(String message) {
        super(message);
    }
    
    public InsufficientFundsException(String message, double amount, double balance) {
        super(message);
        this.amount = amount;
        this.balance = balance;
    }
    
    public double getAmount() { return amount; }
    public double getBalance() { return balance; }
    public double getDeficit() { return amount - balance; }
}

// Custom Unchecked Exception
class InvalidAccountException extends RuntimeException {
    private String accountNumber;
    
    public InvalidAccountException(String message, String accountNumber) {
        super(message);
        this.accountNumber = accountNumber;
    }
    
    public String getAccountNumber() { return accountNumber; }
}

// BankAccount class using custom exceptions
class BankAccount {
    private String accountNumber;
    private String accountHolder;
    private double balance;
    
    public BankAccount(String accNo, String holder, double initialBalance) {
        if (accNo == null || accNo.trim().isEmpty()) {
            throw new InvalidAccountException("Account number cannot be empty", accNo);
        }
        this.accountNumber = accNo;
        this.accountHolder = holder;
        this.balance = initialBalance;
    }
    
    public void withdraw(double amount) throws InsufficientFundsException {
        if (amount <= 0) {
            throw new IllegalArgumentException("Withdrawal amount must be positive");
        }
        
        if (amount > balance) {
            throw new InsufficientFundsException(
                "Insufficient funds! Needed: " + amount + ", Available: " + balance,
                amount, balance
            );
        }
        
        balance -= amount;
        System.out.println("Withdrawn: " + amount + ", New balance: " + balance);
    }
    
    public void deposit(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Deposit amount must be positive");
        }
        balance += amount;
        System.out.println("Deposited: " + amount + ", New balance: " + balance);
    }
    
    public void display() {
        System.out.println("Account: " + accountNumber + ", Holder: " + accountHolder + 
                           ", Balance: " + balance);
    }
}

public class CustomExceptionDemo {
    public static void main(String[] args) {
        try {
            BankAccount account = new BankAccount("ACC123", "John Doe", 500);
            account.display();
            
            account.deposit(200);
            account.withdraw(300);
            account.withdraw(500);  // This will throw custom exception
            
        } catch (InsufficientFundsException e) {
            System.out.println("\\n=== Custom Exception Caught ===");
            System.out.println("Error: " + e.getMessage());
            System.out.println("Requested amount: " + e.getAmount());
            System.out.println("Current balance: " + e.getBalance());
            System.out.println("Deficit: " + e.getDeficit());
            System.out.println("Please deposit " + e.getDeficit() + " to proceed");
            
        } catch (InvalidAccountException e) {
            System.out.println("Invalid account: " + e.getAccountNumber());
            System.out.println("Error: " + e.getMessage());
            
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid operation: " + e.getMessage());
        }
        
        // Testing unchecked custom exception
        try {
            BankAccount invalid = new BankAccount("", "Test", 100);
        } catch (InvalidAccountException e) {
            System.out.println("\\nCaught unchecked exception: " + e.getMessage());
        }
    }
}

/* Output:
Account: ACC123, Holder: John Doe, Balance: 500.0
Deposited: 200.0, New balance: 700.0
Withdrawn: 300.0, New balance: 400.0

=== Custom Exception Caught ===
Error: Insufficient funds! Needed: 500.0, Available: 400.0
Requested amount: 500.0
Current balance: 400.0
Deficit: 100.0
Please deposit 100.0 to proceed
*/`}
          />
        </QuestionCard>

        {/* Summary */}
        <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q40:</span>
              <span className="text-muted-foreground ml-1">try-catch-throw-finally for runtime error handling</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q41:</span>
              <span className="text-muted-foreground ml-1">Checked (compile-time) vs Unchecked (runtime)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q42:</span>
              <span className="text-muted-foreground ml-1">Multiple catch blocks - specific before general</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q43:</span>
              <span className="text-muted-foreground ml-1">Finally always executes (except System.exit())</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q44:</span>
              <span className="text-muted-foreground ml-1">throws = declares exceptions, caller must handle</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q45:</span>
              <span className="text-muted-foreground ml-1">Custom exception = extends Exception/RuntimeException</span>
            </div>
          </div>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="Exception Handling Quiz" />
    </TopicContent>
  );
}