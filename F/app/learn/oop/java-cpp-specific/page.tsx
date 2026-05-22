"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { Zap } from "lucide-react";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
  FileQuestion,
  GitBranch,
  Box,
  Eye,
  Hash,
  Lock,
  Coffee,
  Cpu,
  Shield,
  Link,
  Copy,
  Layers,
} from "lucide-react";

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is the difference between == and .equals() in Java?",
    options: [
      "They are the same",
      "== compares references, .equals() compares content",
      "== compares content, .equals() compares references",
      ".equals() is faster than ==",
    ],
    correctAnswer: 1,
    explanation: "== compares memory references (whether they point to same object). .equals() compares actual content/value.",
  },
  {
    id: 2,
    question: "Why are Strings immutable in Java?",
    options: [
      "To save memory",
      "For security, caching, and thread-safety",
      "To make them slower",
      "No specific reason",
    ],
    correctAnswer: 1,
    explanation: "String immutability provides security (class loading), enables string pooling, and ensures thread-safety.",
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

export default function JavaCppSpecificOOPPage() {
  const result = getSubtopicBySlug("oop", "java-cpp-specific");

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
      {/* Header */}
      <section className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <BookOpen className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            Java & C++ Specific OOP Concepts
          </h2>
        </div>
        <p className="text-muted-foreground">
          Language-specific features that are important for interviews and exams.
        </p>
      </section>

      {/* ==================== JAVA-SPECIFIC (67-75) ==================== */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-orange-500/10">
            <Coffee className="h-5 w-5 text-orange-500" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            Java-Specific OOP (Questions 67-75)
          </h2>
        </div>

        {/* Question 67: == vs .equals() */}
        <QuestionCard
          number={67}
          title="== vs .equals() in Java"
          question="What is the difference between == and .equals() in Java?"
          answer="== is a reference comparison operator that checks if two references point to the same memory location. .equals() is a method that compares the actual content/value of objects. For Strings, == compares references, while .equals() compares character sequences. For primitive types, == compares values. Always use .equals() for content comparison of objects."
          marks={5}
          icon={<Hash className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class EqualsVsDoubleEquals {
    public static void main(String[] args) {
        // For primitive types - == compares values
        int a = 10;
        int b = 10;
        System.out.println(a == b);  // true (values equal)
        
        // For String literals - string pool
        String s1 = "Hello";
        String s2 = "Hello";
        String s3 = new String("Hello");
        String s4 = new String("Hello");
        
        // == compares references
        System.out.println(s1 == s2);     // true (same object from string pool)
        System.out.println(s1 == s3);     // false (different objects)
        System.out.println(s3 == s4);     // false (different objects)
        
        // .equals() compares content
        System.out.println(s1.equals(s2)); // true (same content)
        System.out.println(s1.equals(s3)); // true (same content)
        System.out.println(s3.equals(s4)); // true (same content)
        
        // For objects - default equals() behaves like ==
        class Person {
            String name;
            Person(String name) { this.name = name; }
            // Need to override equals() for content comparison
        }
        
        Person p1 = new Person("John");
        Person p2 = new Person("John");
        System.out.println(p1 == p2);      // false (different objects)
        System.out.println(p1.equals(p2)); // false (default equals uses ==)
        
        // Override equals() properly
        class PersonCorrect {
            String name;
            PersonCorrect(String name) { this.name = name; }
            
            @Override
            public boolean equals(Object obj) {
                if (this == obj) return true;
                if (obj == null || getClass() != obj.getClass()) return false;
                PersonCorrect p = (PersonCorrect) obj;
                return name.equals(p.name);
            }
            
            @Override
            public int hashCode() {
                return name.hashCode();
            }
        }
        
        PersonCorrect pc1 = new PersonCorrect("John");
        PersonCorrect pc2 = new PersonCorrect("John");
        System.out.println(pc1.equals(pc2)); // true (content compared)
        
        // Common mistake with StringBuilder
        StringBuilder sb1 = new StringBuilder("Hello");
        StringBuilder sb2 = new StringBuilder("Hello");
        System.out.println(sb1.equals(sb2)); // false! StringBuilder doesn't override equals()
        System.out.println(sb1.toString().equals(sb2.toString())); // true
    }
}

// Best Practices:
// 1. Use == for primitives and enums
// 2. Use == to check if two references point to same object
// 3. Use .equals() for content comparison
// 4. Always override equals() and hashCode() together
// 5. For Strings from user input, use .equals() not ==`}
          />
        </QuestionCard>

        {/* Question 68: String Interning */}
        <QuestionCard
          number={68}
          title="String Interning"
          question="What is String interning?"
          answer="String interning is a method of storing only one copy of each distinct string value in a pool (String Pool/String Intern Pool). When you create a string literal, Java checks the pool; if it exists, the reference is returned; otherwise, a new string is added to the pool. This saves memory and improves performance. intern() method manually adds a string to the pool."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class StringInterningDemo {
    public static void main(String[] args) {
        // String literals - automatically interned
        String s1 = "Hello";
        String s2 = "Hello";
        String s3 = "He" + "llo";  // Compile-time constant, interned
        
        System.out.println(s1 == s2);     // true (same object from pool)
        System.out.println(s1 == s3);     // true
        
        // String objects - NOT automatically interned
        String s4 = new String("Hello");
        String s5 = new String("Hello");
        
        System.out.println(s1 == s4);     // false (different objects)
        System.out.println(s4 == s5);     // false
        
        // Manual interning
        String s6 = s4.intern();  // Returns reference from pool
        System.out.println(s1 == s6);     // true (s6 references pool object)
        
        // Dynamic strings - not interned
        String s7 = "Hel";
        String s8 = "lo";
        String s9 = s7 + s8;  // Creates new object (not interned)
        System.out.println(s1 == s9);     // false
        
        // intern() with dynamic string
        String s10 = s9.intern();
        System.out.println(s1 == s10);    // true
        
        // Memory and performance example
        System.out.println("\\n=== Memory Optimization ===");
        String[] names = new String[10000];
        for (int i = 0; i < 10000; i++) {
            // Without interning: 10000 separate objects
            names[i] = new String("CommonName");
        }
        // With interning: only one object in pool, 10000 references
        for (int i = 0; i < 10000; i++) {
            names[i] = new String("CommonName").intern();
        }
        
        // Visual representation:
        // String Pool (Heap - PermGen/Metaspace)
        // ┌─────────────────────────┐
        // │ "Hello" ← s1, s2, s3   │
        // │ "World"                 │
        // └─────────────────────────┘
        //        ▲
        //        │ intern()
        // ┌───────┴───────┐
        // │ "Hello"      │ ← s4, s5 (heap objects before interning)
        // └───────────────┘
    }
}

// When to use intern():
// 1. When processing large amounts of repeated string data
// 2. When you have many duplicate strings
// 3. To reduce memory footprint
// Caution: Too much interning can cause memory issues in PermGen/Metaspace`}
          />
        </QuestionCard>

        {/* Question 69: String Immutability */}
        <QuestionCard
          number={69}
          title="Why Strings are Immutable"
          question="Why are Strings immutable in Java?"
          answer="Strings are immutable for several reasons: (1) Security - class loading, network connections, file paths use strings; (2) String Pool - immutability allows safe sharing; (3) Thread-safety - immutable objects are automatically thread-safe; (4) Caching - hashcode can be cached; (5) Performance - no need for defensive copies. Immutability means once created, a String's value cannot be changed."
          marks={5}
          icon={<Lock className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class StringImmutabilityDemo {
    public static void main(String[] args) {
        // String is immutable - operations create new strings
        String s = "Hello";
        System.out.println(System.identityHashCode(s));
        
        s = s + " World";  // Creates NEW string, doesn't modify original
        System.out.println(System.identityHashCode(s)); // Different hashcode
        
        s = s.toUpperCase(); // Creates another NEW string
        System.out.println(System.identityHashCode(s)); // Different again
        
        // Why immutability matters:
        
        // 1. SECURITY EXAMPLE
        String filename = "config.txt";
        // If String were mutable, malicious code could change it
        // during file operations after security check!
        
        // 2. STRING POOL EXAMPLE
        String s1 = "Java";
        String s2 = "Java";
        // Both reference same object because immutable
        System.out.println(s1 == s2); // true
        
        // 3. THREAD-SAFETY EXAMPLE
        // Multiple threads can share same string without synchronization
        
        // 4. HASHCODE CACHING
        String str = "HelloWorld";
        int hash1 = str.hashCode(); // Computed once
        int hash2 = str.hashCode(); // Returns cached value
        System.out.println(hash1 == hash2); // true (cached)
        
        // String vs StringBuilder (mutable)
        StringBuilder sb = new StringBuilder("Hello");
        System.out.println(System.identityHashCode(sb));
        sb.append(" World");  // Modifies same object
        System.out.println(System.identityHashCode(sb)); // Same hashcode!
        
        // Performance implication
        // BAD: Creates many intermediate strings
        String result = "";
        for (int i = 0; i < 1000; i++) {
            result += i;  // Creates new String each iteration!
        }
        
        // GOOD: Use StringBuilder for multiple concatenations
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb2.append(i);  // Modifies same object
        }
        String goodResult = sb2.toString();
    }
}

// How String immutability is achieved:
// 1. String class is final (cannot be extended)
// 2. char[] value is private and final
// 3. No setter methods
// 4. Methods that modify return new String objects
// 5. String class is thread-safe`}
          />
        </QuestionCard>

        {/* Question 70: final keyword */}
        <QuestionCard
          number={70}
          title="final Keyword in Java"
          question="What is the final keyword in Java? (final class, final method, final variable)"
          answer="final is a modifier that restricts modification. final variable: cannot be reassigned (constant). final method: cannot be overridden by subclasses. final class: cannot be extended/inherited. final parameters: cannot be changed inside method. final helps with security, optimization, and design intent."
          marks={5}
          icon={<Lock className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class FinalKeywordDemo {
    
    // final variable - constant
    public static final double PI = 3.14159;
    private final int MAX_VALUE = 100;
    
    // final variable - blank final (initialized in constructor)
    private final int id;
    
    public FinalKeywordDemo(int id) {
        this.id = id;  // Blank final initialized here
    }
    
    // final method - cannot be overridden
    public final void display() {
        System.out.println("This cannot be overridden");
    }
    
    // final parameter - cannot be modified
    public void process(final int value) {
        // value = 10;  // COMPILE ERROR!
        System.out.println(value);
    }
    
    public static void main(String[] args) {
        // final local variable
        final int LOCAL_CONST = 50;
        // LOCAL_CONST = 60;  // COMPILE ERROR!
        
        // final reference - reference cannot change, but object can
        final StringBuilder sb = new StringBuilder("Hello");
        sb.append(" World");  // OK - object content changes
        // sb = new StringBuilder("New");  // COMPILE ERROR!
        
        System.out.println(PI);
    }
}

// final class - cannot be extended
final class UtilityClass {
    public static void helper() {
        System.out.println("Helper method");
    }
}
// class Child extends UtilityClass {}  // COMPILE ERROR!

// final method example
class Parent {
    public final void cannotOverride() {
        System.out.println("Parent final method");
    }
    
    public void canOverride() {
        System.out.println("Parent normal method");
    }
}

class Child extends Parent {
    // public void cannotOverride() {}  // COMPILE ERROR!
    
    @Override
    public void canOverride() {  // OK
        System.out.println("Child overridden method");
    }
}

// Best Practices:
// 1. Use final for constants (static final)
// 2. Use final for immutable class design
// 3. Use final methods for template method pattern
// 4. Use final classes for utility classes
// 5. Use final parameters for clarity

// Performance benefits:
// - Compiler can inline final methods
// - JVM can optimize final variables
// - Better for multi-threading (immutability)`}
          />
        </QuestionCard>

        {/* Question 71: Abstract Class vs Interface */}
        <QuestionCard
          number={71}
          title="Abstract Class vs Interface"
          question="What is an abstract class vs interface? When to use which?"
          answer="Abstract class can have state (instance variables), constructors, and method implementations. Interface (Java 8+) can have default/static methods but no instance variables. Use abstract class when classes share common code/state (is-a relationship). Use interface to define capabilities (can-do relationship). Java supports multiple interfaces but single inheritance."
          marks={5}
          icon={<Layers className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Abstract Class vs Interface"
            headers={["Feature", "Abstract Class", "Interface"]}
            rows={[
              ["Keyword", "abstract", "interface"],
              ["Multiple inheritance", "No", "Yes (multiple interfaces)"],
              ["Instance variables", "Yes", "No (only static final)"],
              ["Constructors", "Yes", "No"],
              ["Method implementation", "Can have", "Default/static methods (Java 8+)"],
              ["Access modifiers", "All", "Public (default)"],
              ["When to use", "Common code/state", "Define contract/capability"],
              ["Relationship", "is-a", "can-do"],
            ]}
          />
          <CodeBlock
            language="java"
            code={`// ABSTRACT CLASS - for related classes with shared code
abstract class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    // Abstract method - must be implemented
    public abstract void makeSound();
    
    // Concrete method - shared implementation
    public void eat() {
        System.out.println(name + " is eating");
    }
    
    public void sleep() {
        System.out.println(name + " is sleeping");
    }
}

class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " says Woof!");
    }
}

// INTERFACE - for capabilities/cross-cutting concerns
interface Flyable {
    void fly();
    
    // Default method (Java 8+)
    default void takeOff() {
        System.out.println("Taking off...");
    }
    
    // Static method (Java 8+)
    static void showWings() {
        System.out.println("Showing wings");
    }
}

interface Swimmable {
    void swim();
}

interface Eatable {
    void eat();
}

// Class can implement multiple interfaces
class Duck extends Animal implements Flyable, Swimmable {
    public Duck(String name) {
        super(name);
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " says Quack!");
    }
    
    public void fly() {
        System.out.println(name + " is flying");
    }
    
    public void swim() {
        System.out.println(name + " is swimming");
    }
}

// WHEN TO USE WHAT:
// Use Abstract Class when:
// - You have common code to share
// - You have common state (instance variables)
// - You want non-public members
// - You expect subclasses to be closely related

// Use Interface when:
// - You need multiple inheritance of type
// - You want to define a contract/capability
// - You have unrelated classes that share behavior
// - You want to achieve loose coupling

// Java 8+ Interface with default methods
interface Payment {
    void pay(double amount);
    
    default void validatePayment() {
        System.out.println("Validating payment...");
    }
    
    static void paymentGateway() {
        System.out.println("Payment Gateway: Stripe");
    }
}

class CreditCard implements Payment {
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " via Credit Card");
    }
}`}
          />
        </QuestionCard>

        {/* Question 72: Functional Interface */}
        <QuestionCard
          number={72}
          title="Functional Interface"
          question="What is a functional interface?"
          answer="A functional interface is an interface with exactly one abstract method (SAM - Single Abstract Method). It can have multiple default/static methods. Used as target type for lambda expressions and method references. Annotated with @FunctionalInterface (optional but recommended). Examples: Runnable, Comparator, Callable, Consumer, Predicate, Supplier."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`import java.util.function.*;

// Defining functional interfaces
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);  // Single abstract method
    // int another();  // ERROR: can't have multiple abstract methods
    
    // Default and static methods are allowed
    default void display() {
        System.out.println("Calculator");
    }
    
    static void info() {
        System.out.println("Functional Interface Demo");
    }
}

// Built-in functional interfaces
public class FunctionalInterfaceDemo {
    public static void main(String[] args) {
        // Using lambda with custom functional interface
        Calculator add = (a, b) -> a + b;
        Calculator multiply = (a, b) -> a * b;
        Calculator subtract = (a, b) -> a - b;
        
        System.out.println(add.calculate(5, 3));      // 8
        System.out.println(multiply.calculate(5, 3)); // 15
        System.out.println(subtract.calculate(5, 3)); // 2
        
        // Built-in functional interfaces
        
        // 1. Predicate<T> - boolean test(T t)
        Predicate<String> isEmpty = s -> s == null || s.isEmpty();
        System.out.println(isEmpty.test(""));   // true
        System.out.println(isEmpty.test("Hi")); // false
        
        // 2. Consumer<T> - void accept(T t)
        Consumer<String> printer = s -> System.out.println("Print: " + s);
        printer.accept("Hello");
        
        // 3. Supplier<T> - T get()
        Supplier<Double> random = () -> Math.random();
        System.out.println(random.get());
        
        // 4. Function<T,R> - R apply(T t)
        Function<String, Integer> length = s -> s.length();
        System.out.println(length.apply("Hello")); // 5
        
        // 5. BiFunction<T,U,R> - R apply(T t, U u)
        BiFunction<Integer, Integer, Integer> max = (a, b) -> a > b ? a : b;
        System.out.println(max.apply(10, 20)); // 20
        
        // 6. Runnable - void run()
        Runnable task = () -> System.out.println("Running...");
        task.run();
        
        // 7. Comparator<T> - int compare(T o1, T o2)
        Comparator<String> byLength = (s1, s2) -> s1.length() - s2.length();
        
        // Custom functional interface with generics
        @FunctionalInterface
        interface Transformer<T, R> {
            R transform(T input);
        }
        
        Transformer<String, Integer> toLength = s -> s.length();
        Transformer<Integer, String> toString = i -> "Number: " + i;
        
        System.out.println(toLength.transform("Java"));   // 4
        System.out.println(toString.transform(42));       // Number: 42
    }
}

// @FunctionalInterface annotation benefits:
// 1. Compiler checks that interface has only one abstract method
// 2. Documents intent
// 3. Enables lambda support`}
          />
        </QuestionCard>

        {/* Question 73: Lambda Expression */}
        <QuestionCard
          number={73}
          title="Lambda Expression"
          question="What is a lambda expression? Write an example."
          answer="Lambda expression is an anonymous function (without class/method name) that can be passed as data. Syntax: (parameters) -> expression or { statements }. Used with functional interfaces. They enable functional programming in Java, reduce boilerplate code, and work with Stream API."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`import java.util.*;
import java.util.stream.*;

public class LambdaDemo {
    public static void main(String[] args) {
        // Lambda syntax variations
        
        // 1. No parameters
        Runnable run = () -> System.out.println("Running");
        run.run();
        
        // 2. One parameter (no parentheses needed)
        java.util.function.Consumer<String> print = s -> System.out.println(s);
        print.accept("Hello Lambda");
        
        // 3. Multiple parameters
        java.util.function.BinaryOperator<Integer> add = (a, b) -> a + b;
        System.out.println(add.apply(5, 3));
        
        // 4. Multiple statements (use braces and return)
        java.util.function.Function<Integer, Integer> square = x -> {
            int result = x * x;
            return result;
        };
        
        // Practical examples with collections
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
        
        // Old way (anonymous class)
        names.sort(new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                return a.compareTo(b);
            }
        });
        
        // Lambda way
        names.sort((a, b) -> a.compareTo(b));
        names.sort(String::compareTo);  // Method reference (even shorter)
        
        // Iterate with lambda
        names.forEach(name -> System.out.println(name));
        
        // Stream API with lambdas
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Chain of operations
        int result = numbers.stream()
            .filter(n -> n % 2 == 0)           // Keep even numbers
            .map(n -> n * n)                    // Square each
            .filter(n -> n > 10)                // Keep > 10
            .reduce(0, (a, b) -> a + b);        // Sum
        
        System.out.println("Result: " + result); // 4²=16 + 6²=36 + 8²=64 + 10²=100 = 216
        
        // Lambda with custom functional interface
        @FunctionalInterface
        interface StringOperation {
            String operate(String s);
        }
        
        StringOperation toUpper = s -> s.toUpperCase();
        StringOperation reverse = s -> new StringBuilder(s).reverse().toString();
        
        System.out.println(toUpper.operate("hello"));  // HELLO
        System.out.println(reverse.operate("world"));  // dlrow
        
        // Lambda capturing effectively final variables
        int multiplier = 10;
        java.util.function.Function<Integer, Integer> times = x -> x * multiplier;
        // multiplier = 20;  // Would cause error if uncommented (not effectively final)
        
        // Common lambda patterns
        List<Integer> nums = Arrays.asList(5, 2, 8, 1, 9);
        
        // Sort descending
        nums.sort((a, b) -> b - a);
        
        // Find max
        Optional<Integer> max = nums.stream().max((a, b) -> a - b);
        
        // Filter and collect
        List<Integer> evenSquares = nums.stream()
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .collect(Collectors.toList());
    }
}`}
          />
        </QuestionCard>

        {/* Question 74: Method Reference */}
        <QuestionCard
          number={74}
          title="Method Reference (:: operator)"
          question="What is method reference? (:: operator)"
          answer="Method reference is a shorthand syntax for lambda expressions that call an existing method. Types: (1) Static method reference: ClassName::staticMethod, (2) Instance method of specific object: object::instanceMethod, (3) Instance method of arbitrary type: ClassName::instanceMethod, (4) Constructor reference: ClassName::new. They make code more concise and readable."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class MethodReferenceDemo {
    
    // Static method
    public static boolean isEven(int n) {
        return n % 2 == 0;
    }
    
    // Instance method
    public String toUpperCase(String s) {
        return s.toUpperCase();
    }
    
    public static void main(String[] args) {
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
        
        // 1. Static method reference: ClassName::staticMethod
        // Lambda: n -> MethodReferenceDemo.isEven(n)
        // Method ref: MethodReferenceDemo::isEven
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        numbers.stream()
            .filter(MethodReferenceDemo::isEven)  // Method reference
            .forEach(System.out::println);         // 2, 4
        
        // 2. Instance method of specific object: object::instanceMethod
        MethodReferenceDemo demo = new MethodReferenceDemo();
        // Lambda: s -> demo.toUpperCase(s)
        // Method ref: demo::toUpperCase
        names.stream()
            .map(demo::toUpperCase)
            .forEach(System.out::println);
        
        // 3. Instance method of arbitrary type: ClassName::instanceMethod
        // Lambda: (s1, s2) -> s1.compareTo(s2)
        // Method ref: String::compareTo
        names.sort(String::compareTo);
        
        // Lambda: s -> System.out.println(s)
        // Method ref: System.out::println
        names.forEach(System.out::println);
        
        // 4. Constructor reference: ClassName::new
        // Lambda: () -> new ArrayList<>()
        // Method ref: ArrayList::new
        Supplier<List<String>> listSupplier = ArrayList::new;
        List<String> newList = listSupplier.get();
        
        // Constructor reference with parameter
        Function<String, Integer> toInteger = Integer::new;
        Integer num = toInteger.apply("123");
        
        // Array constructor reference
        IntFunction<int[]> arrayCreator = int[]::new;
        int[] arr = arrayCreator.apply(5);
        
        // Complete example with streams
        List<String> words = Arrays.asList("hello", "world", "java");
        
        // Using lambda
        words.stream()
            .map(s -> s.toUpperCase())
            .filter(s -> s.length() > 4)
            .forEach(s -> System.out.println(s));
        
        // Using method reference (cleaner)
        words.stream()
            .map(String::toUpperCase)          // Instance method of String
            .filter(s -> s.length() > 4)
            .forEach(System.out::println);      // Instance method of PrintStream
        
        // Static method reference with custom class
        class MathUtils {
            static int max(int a, int b) {
                return a > b ? a : b;
            }
        }
        
        BinaryOperator<Integer> maxOp = MathUtils::max;
        System.out.println(maxOp.apply(10, 20)); // 20
        
        // When to use method reference vs lambda:
        // Use method reference when:
        // 1. Lambda just calls existing method
        // 2. Code becomes more readable
        // 3. You want to reuse existing methods
        
        // Use lambda when:
        // 1. Method reference would be unclear
        // 2. Need multiple statements
        // 3. Need to capture variables
    }
}`}
          />
        </QuestionCard>

        {/* Question 75: instanceof operator */}
        <QuestionCard
          number={75}
          title="instanceof Operator"
          question="What is the instanceof operator?"
          answer="instanceof is a binary operator that tests whether an object is an instance of a specific class, interface, or subclass. Returns true if the object is an instance, false otherwise. Used for safe downcasting, type checking before casting to avoid ClassCastException. Also works with interfaces and null (returns false for null)."
          marks={5}
          icon={<Eye className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class InstanceofDemo {
    
    static class Animal { }
    static class Dog extends Animal { 
        void bark() { System.out.println("Woof!"); }
    }
    static class Cat extends Animal {
        void meow() { System.out.println("Meow!"); }
    }
    
    static interface Flyable { }
    static class Bird implements Flyable { }
    
    public static void main(String[] args) {
        Animal animal = new Dog();
        
        // Basic instanceof
        System.out.println(animal instanceof Dog);     // true
        System.out.println(animal instanceof Animal);  // true
        System.out.println(animal instanceof Cat);     // false
        System.out.println(animal instanceof Object);  // true
        
        // null returns false
        Animal nullAnimal = null;
        System.out.println(nullAnimal instanceof Dog); // false
        
        // Interface checking
        Bird bird = new Bird();
        System.out.println(bird instanceof Flyable);   // true
        
        // Safe downcasting pattern
        if (animal instanceof Dog) {
            Dog dog = (Dog) animal;  // Safe cast
            dog.bark();
        }
        
        // Pattern matching (Java 16+)
        if (animal instanceof Dog d) {
            d.bark();  // No explicit cast needed
        }
        
        // Multiple checks in one condition
        class MyClass { }
        
        Object obj = "Hello";
        if (obj instanceof String && ((String) obj).length() > 0) {
            System.out.println("Non-empty string");
        }
        
        // Using instanceof in a method
        processAnimal(new Dog());
        processAnimal(new Cat());
        processAnimal(new Animal());
        
        // Array type checking
        Object[] arr = new String[10];
        System.out.println(arr instanceof String[]);  // true
        System.out.println(arr instanceof Object[]);  // true
    }
    
    static void processAnimal(Animal animal) {
        if (animal instanceof Dog) {
            Dog dog = (Dog) animal;
            dog.bark();
        } else if (animal instanceof Cat) {
            Cat cat = (Cat) animal;
            cat.meow();
        } else {
            System.out.println("Unknown animal");
        }
        
        // Better with pattern matching (Java 16+)
        if (animal instanceof Dog d) {
            d.bark();
        } else if (animal instanceof Cat c) {
            c.meow();
        }
    }
    
    // Common use case: equals() method
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (!(obj instanceof InstanceofDemo)) return false;  // instanceof check
        InstanceofDemo other = (InstanceofDemo) obj;
        return true; // compare fields
    }
}

// Best Practices:
// 1. Always use instanceof before downcasting
// 2. Use pattern matching (Java 16+) for cleaner code
// 3. Overuse may indicate design issue - prefer polymorphism
// 4. instanceof with null always returns false`}
          />
        </QuestionCard>
      </section>

      {/* ==================== C++-SPECIFIC (76-82) ==================== */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-blue-500/10">
            <Cpu className="h-5 w-5 text-blue-500" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            C++-Specific OOP (Questions 76-82)
          </h2>
        </div>

        {/* Question 76: C++ vs Java OOP */}
        <QuestionCard
          number={76}
          title="C++ vs Java OOP"
          question="What is the difference between C++ and Java OOP?"
          answer="C++ supports multiple inheritance, operator overloading, pointers, and manual memory management. Java supports single inheritance (multiple via interfaces), automatic garbage collection, no pointers, and runs on JVM. C++ compiles to native code, Java to bytecode. C++ has destructors, Java has finalize() (deprecated). Java has built-in threading and reflection."
          marks={5}
          icon={<Cpu className="h-3 w-3" />}
        >
          <DifferenceTable
            title="C++ vs Java OOP"
            headers={["Feature", "C++", "Java"]}
            rows={[
              ["Multiple Inheritance", "✓ Supported (with virtual)", "✗ (via interfaces only)"],
              ["Operator Overloading", "✓ Yes", "✗ No (except String +)"],
              ["Memory Management", "Manual (delete)", "Automatic (GC)"],
              ["Pointers", "✓ Yes", "✗ No (references only)"],
              ["Virtual Functions", "✓ (virtual keyword)", "✓ (non-static methods are virtual)"],
              ["Destructors", "✓ Yes", "✗ (finalize deprecated)"],
              ["Garbage Collection", "✗ No", "✓ Yes"],
              ["Compilation", "To native code", "To bytecode (JVM)"],
              ["Platform", "Platform dependent", "Platform independent"],
              ["Reflection", "Limited (RTTI)", "✓ Full support"],
              ["Threading", "External libraries", "✓ Built-in"],
            ]}
          />
          <CodeBlock
            language="cpp"
            code={`// C++ Features
#include <iostream>
using namespace std;

// Multiple inheritance
class A { };
class B { };
class C : public A, public B { };  // Multiple inheritance

// Operator overloading
class Complex {
    double real, imag;
public:
    Complex operator+(const Complex& other) {
        Complex result;
        result.real = real + other.real;
        return result;
    }
};

// Manual memory management
int main() {
    int* ptr = new int(10);
    delete ptr;  // Must manually delete
}

// Java (conceptual)
/*
public class JavaExample {
    // Single inheritance only
    class Child extends Parent { }  // Only one parent
    
    // Multiple inheritance via interfaces
    class MyClass implements Interface1, Interface2 { }
    
    // No operator overloading
    // int result = a + b;  // Only works for primitives and String
    
    // Automatic garbage collection
    // No need to delete objects
}`}
          />
        </QuestionCard>

        {/* Question 77: Multiple Inheritance & Diamond Problem */}
        <QuestionCard
          number={77}
          title="Multiple Inheritance & Diamond Problem"
          question="What is multiple inheritance in C++? Explain the diamond problem and virtual inheritance."
          answer="Multiple inheritance allows a class to inherit from multiple base classes. The diamond problem occurs when a class inherits from two classes that share a common ancestor, causing ambiguity about which parent's members to inherit. Virtual inheritance (virtual keyword) solves this by ensuring only one copy of the common base class is inherited."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <pre className="text-xs font-mono">
{`        Animal
        /    \\
      Mammal  Bird
        \\    /
         Bat (Diamond Problem)

Without virtual inheritance: Bat has two copies of Animal
With virtual inheritance: Bat has one copy of Animal`}
            </pre>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

// Base class
class Animal {
protected:
    string name;
public:
    Animal() { cout << "Animal constructor" << endl; }
    void breathe() { cout << "Breathing..." << endl; }
};

// Without virtual inheritance - Diamond Problem
class Mammal : public Animal {
public:
    Mammal() { cout << "Mammal constructor" << endl; }
    void walk() { cout << "Walking..." << endl; }
};

class Bird : public Animal {
public:
    Bird() { cout << "Bird constructor" << endl; }
    void fly() { cout << "Flying..." << endl; }
};

// Diamond problem! Bat has two copies of Animal
class Bat : public Mammal, public Bird {
public:
    Bat() { cout << "Bat constructor" << endl; }
};

// WITH VIRTUAL INHERITANCE - Solution
class AnimalV {
protected:
    string name;
public:
    AnimalV() { cout << "AnimalV constructor" << endl; }
    void breathe() { cout << "Breathing..." << endl; }
};

class MammalV : virtual public AnimalV {
public:
    MammalV() { cout << "MammalV constructor" << endl; }
};

class BirdV : virtual public AnimalV {
public:
    BirdV() { cout << "BirdV constructor" << endl; }
};

// Now BatV has only ONE copy of AnimalV
class BatV : public MammalV, public BirdV {
public:
    BatV() { cout << "BatV constructor" << endl; }
};

int main() {
    // Diamond Problem demonstration
    cout << "=== Without Virtual Inheritance ===" << endl;
    Bat bat;
    // bat.breathe();  // ERROR! Ambiguous - which Animal?
    bat.Mammal::breathe();  // Must specify
    bat.Bird::breathe();    // Two different Animal objects!
    
    cout << "\\n=== With Virtual Inheritance ===" << endl;
    BatV batv;
    batv.breathe();  // Works! Only one AnimalV object
    
    // Memory layout difference:
    // Without virtual: Bat contains Mammal(Animal) + Bird(Animal)
    // With virtual: Bat contains Mammal + Bird + single Animal
    
    return 0;
}

// Virtual inheritance ensures:
// 1. Only one instance of common base class
// 2. Constructor of virtual base called only once
// 3. Resolves ambiguity automatically`}
          />
        </QuestionCard>

        {/* Question 78: Virtual Destructors */}
        <QuestionCard
          number={78}
          title="Virtual Destructors"
          question="What are virtual destructors? Why are they needed?"
          answer="Virtual destructors ensure that the correct destructor is called when deleting a derived object through a base pointer. Without virtual destructor, only base destructor runs, causing memory leaks. Virtual destructors enable proper cleanup of derived class resources. Rule: If a class has virtual methods, make destructor virtual."
          marks={5}
          icon={<Shield className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

// WITHOUT virtual destructor - PROBLEM
class BaseBad {
public:
    BaseBad() { cout << "BaseBad constructor" << endl; }
    ~BaseBad() { cout << "BaseBad destructor" << endl; }
};

class DerivedBad : public BaseBad {
private:
    int* data;
public:
    DerivedBad() {
        data = new int[1000];
        cout << "DerivedBad constructor - allocated memory" << endl;
    }
    ~DerivedBad() {
        delete[] data;
        cout << "DerivedBad destructor - freed memory" << endl;
    }
};

// WITH virtual destructor - SOLUTION
class BaseGood {
public:
    BaseGood() { cout << "BaseGood constructor" << endl; }
    virtual ~BaseGood() { cout << "BaseGood destructor" << endl; }
};

class DerivedGood : public BaseGood {
private:
    int* data;
public:
    DerivedGood() {
        data = new int[1000];
        cout << "DerivedGood constructor - allocated memory" << endl;
    }
    ~DerivedGood() override {
        delete[] data;
        cout << "DerivedGood destructor - freed memory" << endl;
    }
};

int main() {
    cout << "=== WITHOUT VIRTUAL DESTRUCTOR ===" << endl;
    BaseBad* ptr1 = new DerivedBad();
    delete ptr1;  // Only BaseBad destructor runs!
    // MEMORY LEAK! DerivedBad's destructor never called
    
    cout << "\\n=== WITH VIRTUAL DESTRUCTOR ===" << endl;
    BaseGood* ptr2 = new DerivedGood();
    delete ptr2;  // Both destructors run correctly
    
    cout << "\\n=== RULE OF THREE/FIVE ===" << endl;
    // If you have virtual functions, make destructor virtual
    class Correct {
    public:
        virtual ~Correct() = default;  // Virtual destructor
        virtual void doSomething() { }
    };
    
    return 0;
}

// When to use virtual destructor:
// 1. Class has any virtual functions
// 2. Class is meant to be inherited from
// 3. Deleting derived objects via base pointer
// 4. Polymorphic base classes

// Cost: Virtual destructor adds vtable overhead (8 bytes per object)
// But worth it for proper cleanup`}
          />
        </QuestionCard>

        {/* Question 79: Friend Function */}
        <QuestionCard
          number={79}
          title="Friend Function"
          question="What is a friend function? When is it used?"
          answer="Friend function is a non-member function that has access to private and protected members of a class. Declared with 'friend' keyword inside the class. Used for operator overloading (<<, >>), when a function needs access to multiple classes' private data, or for testing/debugging. Friend functions break encapsulation but are necessary in some cases."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class A {
private:
    int secret;
protected:
    int protectedData;
public:
    A() : secret(100), protectedData(50) {}
    
    // Declare friend function
    friend void showSecret(A& obj);
    
    // Friend class
    friend class B;
    
    // Friend function that accesses multiple classes
    friend void exchange(A& a, class C& c);
};

class B {
public:
    void accessA(A& a) {
        cout << "B accessing A's secret: " << a.secret << endl;
        cout << "B accessing A's protected: " << a.protectedData << endl;
    }
};

class C {
private:
    int value;
public:
    C(int v) : value(v) {}
    friend void exchange(A& a, C& c);
};

// Friend function definition
void showSecret(A& obj) {
    cout << "Friend function accessing secret: " << obj.secret << endl;
    cout << "Friend function accessing protected: " << obj.protectedData << endl;
}

void exchange(A& a, C& c) {
    cout << "Before exchange: A.secret=" << a.secret << ", C.value=" << c.value << endl;
    int temp = a.secret;
    a.secret = c.value;
    c.value = temp;
    cout << "After exchange: A.secret=" << a.secret << ", C.value=" << c.value << endl;
}

// Operator overloading with friend
class Complex {
private:
    double real, imag;
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // Friend function for operator<<
    friend ostream& operator<<(ostream& out, const Complex& c);
    friend Complex operator+(const Complex& a, const Complex& b);
};

ostream& operator<<(ostream& out, const Complex& c) {
    out << c.real << " + " << c.imag << "i";
    return out;
}

Complex operator+(const Complex& a, const Complex& b) {
    return Complex(a.real + b.real, a.imag + b.imag);
}

int main() {
    A a;
    showSecret(a);  // Friend function call
    
    B b;
    b.accessA(a);   // Friend class access
    
    C c(200);
    exchange(a, c);  // Friend accessing two classes
    
    Complex c1(3, 4), c2(1, 2);
    Complex c3 = c1 + c2;
    cout << "Complex: " << c3 << endl;
    
    return 0;
}

// Friend vs Public:
// Friend: Non-member access to private
// Public: Member function access to private

// When to use friends:
// 1. Operator overloading (<<, >>)
// 2. Functions needing access to multiple classes
// 3. Testing/debugging helper functions
// 4. Performance-critical operations

// Caution: Overuse breaks encapsulation`}
          />
        </QuestionCard>

        {/* Question 80: this pointer */}
        <QuestionCard
          number={80}
          title="this pointer"
          question="What is the this pointer?"
          answer="this is an implicit pointer available in non-static member functions that points to the current object. It's used to access members when there are naming conflicts (parameter names same as member variables), to return *this for method chaining, and to pass current object to other functions. It's a const pointer (can't be reassigned)."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class Counter {
private:
    int count;
    string name;
    
public:
    // Constructor using this to resolve naming conflict
    Counter(int count, string name) {
        this->count = count;  // this->count is member, count is parameter
        this->name = name;
    }
    
    // Method chaining using this
    Counter& increment() {
        this->count++;
        return *this;  // Return current object
    }
    
    Counter& decrement() {
        this->count--;
        return *this;
    }
    
    Counter& setName(const string& name) {
        this->name = name;
        return *this;
    }
    
    void display() {
        cout << name << " count: " << this->count << endl;
    }
    
    // Comparing this with another object
    bool isEqual(const Counter& other) {
        return this == &other;  // Compare addresses
    }
    
    // Using this to pass current object
    void logCurrentObject() {
        logObject(this);  // Pass current object
    }
    
    static void logObject(Counter* obj) {
        cout << "Object at address: " << obj << endl;
    }
};

class ThisDemo {
private:
    int x, y;
    
public:
    ThisDemo& setX(int x) {
        this->x = x;
        return *this;
    }
    
    ThisDemo& setY(int y) {
        this->y = y;
        return *this;
    }
    
    ThisDemo& setBoth(int x, int y) {
        this->x = x;
        this->y = y;
        return *this;
    }
    
    void display() const {
        // const member function - this is const pointer to const
        cout << "x=" << x << ", y=" << y << endl;
    }
};

int main() {
    // Method chaining example
    Counter c(0, "MyCounter");
    c.increment().increment().decrement().setName("UpdatedCounter");
    c.display();  // UpdatedCounter count: 1
    
    // Using this for chaining in initializers
    ThisDemo demo;
    demo.setX(10).setY(20).setBoth(30, 40);
    demo.display();  // x=30, y=40
    
    // This pointer address
    Counter c1(1, "A"), c2(2, "B");
    cout << "c1 equals itself: " << c1.isEqual(c1) << endl;  // true
    cout << "c1 equals c2: " << c1.isEqual(c2) << endl;      // false
    
    // const member functions and this
    const Counter c3(5, "Const");
    // c3.increment();  // ERROR! const object can't call non-const method
    
    return 0;
}

// 'this' is not available in:
// 1. Static member functions
// 2. Non-member functions
// 3. Friend functions

// 'this' pointer type:
// - In non-const member function: ClassName* const this
// - In const member function: const ClassName* const this`}
          />
        </QuestionCard>

        {/* Question 81: Templates vs Generics */}
        <QuestionCard
          number={81}
          title="Templates vs Generics"
          question="What are templates in C++? How are they different from generics in Java?"
          answer="Templates are a C++ feature for generic programming that generate code at compile-time (compile-time polymorphism). Java generics use type erasure (runtime). Templates create separate code for each type; generics use single bytecode. Templates support primitive types and specialization; generics work only with objects. Templates can cause code bloat; generics have no overhead."
          marks={5}
          icon={<Hash className="h-3 w-3" />}
        >
          <DifferenceTable
            title="C++ Templates vs Java Generics"
            headers={["Feature", "C++ Templates", "Java Generics"]}
            rows={[
              ["Implementation", "Compile-time code generation", "Type erasure (runtime)"],
              ["Type checking", "At compile time (strict)", "At compile time"],
              ["Specialization", "✓ Full template specialization", "✗ Not supported"],
              ["Primitive types", "✓ int, char, etc.", "✗ Use wrappers (Integer)"],
              ["Code generation", "Separate code per type", "Single bytecode"],
              ["Runtime overhead", "No (compile-time)", "Boxing/unboxing overhead"],
              ["Code size", "Larger (code bloat)", "Smaller"],
              ["Debugging", "Harder (complex errors)", "Easier"],
            ]}
          />
          <CodeBlock
            language="cpp"
            code={`// C++ TEMPLATES
#include <iostream>
#include <string>
using namespace std;

// Function template
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Class template
template<typename T>
class Box {
private:
    T value;
public:
    Box(T v) : value(v) {}
    
    T getValue() { return value; }
    
    // Template specialization (C++ specific)
    void display() {
        cout << "Generic: " << value << endl;
    }
};

// Template specialization for string
template<>
void Box<string>::display() {
    cout << "String specialization: " << value << endl;
}

// Template with multiple parameters
template<typename T1, typename T2>
class Pair {
    T1 first;
    T2 second;
public:
    Pair(T1 f, T2 s) : first(f), second(s) {}
    void display() {
        cout << "(" << first << ", " << second << ")" << endl;
    }
};

// Template with non-type parameter
template<typename T, int size>
class Array {
    T arr[size];
public:
    T& operator[](int index) { return arr[index]; }
    int getSize() { return size; }
};

// Java GENERICS (conceptual)
/*
// Type erasure - becomes Object at runtime
public class Box<T> {
    private T value;
    public Box(T v) { value = v; }
    public T getValue() { return value; }
}

// Wildcards for flexibility
public void process(List<? extends Number> list) { }
*/

int main() {
    // C++ Templates
    cout << max(10, 20) << endl;        // int version generated
    cout << max(3.14, 2.71) << endl;    // double version generated
    cout << max<string>("Hello", "World") << endl;  // string version
    
    Box<int> intBox(100);      // Generates Box<int> class
    Box<double> doubleBox(3.14);  // Generates Box<double> class
    Box<string> strBox("Hello");   // Uses specialization
    
    intBox.display();    // Generic: 100
    strBox.display();    // String specialization: Hello
    
    Pair<int, string> p(1, "One");
    p.display();  // (1, One)
    
    Array<int, 5> arr;
    arr[0] = 10;
    cout << "Array size: " << arr.getSize() << endl;
    
    // Compile-time vs runtime
    // C++: Different types generate different code
    // Java: List<Integer> and List<String> are same at runtime
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 82: Deep Copy vs Shallow Copy */}
        <QuestionCard
          number={82}
          title="Deep Copy vs Shallow Copy"
          question="What is the difference between deep copy and shallow copy?"
          answer="Shallow copy copies the object's immediate members, but pointers/references still point to the same dynamically allocated memory. Deep copy creates new copies of dynamically allocated memory, ensuring complete independence. Shallow copy (default copy constructor) can cause double deletion, data corruption. Deep copy requires custom copy constructor and assignment operator (Rule of Three/Five)."
          marks={5}
          icon={<Copy className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <pre className="text-xs font-mono">
{`Shallow Copy (Problem):
┌─────────┐     ┌─────────┐
│ Object1 │────▶│  Heap   │
└─────────┘     │  Data   │
┌─────────┐     └─────────┘
│ Object2 │─────┘ (same address)
└─────────┘

Deep Copy (Solution):
┌─────────┐     ┌─────────┐
│ Object1 │────▶│  Heap   │
└─────────┘     │  Data1  │
                └─────────┘
┌─────────┐     ┌─────────┐
│ Object2 │────▶│  Heap   │
└─────────┘     │  Data2  │
                └─────────┘`}
            </pre>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
#include <cstring>
using namespace std;

// SHALLOW COPY - Problematic
class ShallowString {
private:
    char* data;
public:
    ShallowString(const char* str) {
        data = new char[strlen(str) + 1];
        strcpy(data, str);
        cout << "Constructor: " << data << " at " << (void*)data << endl;
    }
    
    // Default copy constructor does shallow copy!
    // Both objects share same memory
    
    ~ShallowString() {
        cout << "Destructor: deleting " << (void*)data << endl;
        delete[] data;  // Double delete problem!
    }
    
    void display() {
        cout << data << " (" << (void*)data << ")" << endl;
    }
};

// DEEP COPY - Correct implementation
class DeepString {
private:
    char* data;
public:
    DeepString(const char* str) {
        data = new char[strlen(str) + 1];
        strcpy(data, str);
        cout << "Constructor: " << data << " at " << (void*)data << endl;
    }
    
    // Deep copy constructor
    DeepString(const DeepString& other) {
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
        cout << "Deep copy: " << data << " at " << (void*)data << endl;
    }
    
    // Deep copy assignment operator
    DeepString& operator=(const DeepString& other) {
        if (this != &other) {
            delete[] data;
            data = new char[strlen(other.data) + 1];
            strcpy(data, other.data);
        }
        cout << "Assignment: " << data << " at " << (void*)data << endl;
        return *this;
    }
    
    ~DeepString() {
        cout << "Destructor: deleting " << (void*)data << endl;
        delete[] data;
    }
    
    void display() {
        cout << data << " (" << (void*)data << ")" << endl;
    }
};

// Rule of Three (C++03): Copy constructor, Copy assignment, Destructor
// Rule of Five (C++11): + Move constructor, Move assignment

class ResourceManager {
private:
    int* data;
    size_t size;
public:
    ResourceManager(size_t s) : size(s) {
        data = new int[s];
        cout << "Allocated " << size << " ints" << endl;
    }
    
    // Deep copy constructor
    ResourceManager(const ResourceManager& other) 
        : size(other.size), data(new int[other.size]) {
        copy(other.data, other.data + size, data);
        cout << "Deep copied" << endl;
    }
    
    // Deep copy assignment
    ResourceManager& operator=(const ResourceManager& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            copy(other.data, other.data + size, data);
            cout << "Deep assigned" << endl;
        }
        return *this;
    }
    
    ~ResourceManager() {
        delete[] data;
        cout << "Destructor" << endl;
    }
};

int main() {
    cout << "=== SHALLOW COPY PROBLEM ===" << endl;
    ShallowString s1("Hello");
    ShallowString s2 = s1;  // Shallow copy - both point to same memory
    
    s1.display();
    s2.display();
    // CRASH on destruction - double delete!
    
    cout << "\\n=== DEEP COPY SOLUTION ===" << endl;
    DeepString d1("World");
    DeepString d2 = d1;  // Deep copy - separate memory
    
    d1.display();
    d2.display();
    // Clean destruction - no double delete
    
    cout << "\\n=== When to use which ===" << endl;
    cout << "Shallow copy: No dynamic memory, trivial classes" << endl;
    cout << "Deep copy: Classes with dynamic memory, resources" << endl;
    
    return 0;
}

// Best Practices:
// 1. Follow Rule of Three/Five for resource management
// 2. Use smart pointers (unique_ptr, shared_ptr) to avoid manual memory
// 3. Disable copy if not needed (unique_ptr)
// 4. Prefer memberwise copy for simple types`}
          />
        </QuestionCard>
      </section>

      {/* Summary */}
      <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
        <div className="flex items-center gap-3 mb-4">
          <Target className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-bold text-primary mb-2">Java-Specific (67-75)</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li><strong>Q67:</strong> == (reference) vs .equals() (content)</li>
              <li><strong>Q68:</strong> String interning = string pool optimization</li>
              <li><strong>Q69:</strong> Strings immutable for security, caching, thread-safety</li>
              <li><strong>Q70:</strong> final = constant, prevents overriding/inheritance</li>
              <li><strong>Q71:</strong> Abstract class (state) vs Interface (contract)</li>
              <li><strong>Q72:</strong> Functional interface = single abstract method</li>
              <li>
  <strong>Q73:</strong> Lambda = anonymous function (params) {"->"} body
</li>
              <li><strong>Q74:</strong> Method reference :: = shorthand for lambda</li>
              <li><strong>Q75:</strong> instanceof = type checking before casting</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-primary mb-2">C++-Specific (76-82)</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li><strong>Q76:</strong> C++ vs Java: multiple inheritance, pointers, manual memory</li>
              <li><strong>Q77:</strong> Multiple inheritance + diamond problem + virtual inheritance</li>
              <li><strong>Q78:</strong> Virtual destructors = proper cleanup in polymorphism</li>
              <li><strong>Q79:</strong> Friend = non-member access to private members</li>
              <li><strong>Q80:</strong> this pointer = current object reference</li>
              <li><strong>Q81:</strong> Templates (compile-time) vs Generics (type erasure)</li>
              <li><strong>Q82:</strong> Deep copy vs Shallow copy + Rule of Three/Five</li>
            </ul>
          </div>
        </div>
      </div>

      <Quiz questions={quizQuestions} title="Java & C++ OOP Quiz" />
    </TopicContent>
  );
}