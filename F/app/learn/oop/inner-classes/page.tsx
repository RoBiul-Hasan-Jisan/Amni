"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
  FileQuestion,
  Layers,
  Box,
  Lock,
  GitBranch,
  Eye,
} from "lucide-react";

const javaInnerClassesCode = `// Inner/Nested Classes in Java

// Outer class
public class OuterClass {
    private String outerField = "Outer Field";
    private static String staticOuterField = "Static Outer Field";
    
    // 1. Non-static Inner Class (Member Inner Class)
    class InnerClass {
        void display() {
            // Can access all members of outer class
            System.out.println("Accessing: " + outerField);
            System.out.println("Accessing: " + staticOuterField);
        }
    }
    
    // 2. Static Nested Class
    static class StaticNestedClass {
        void display() {
            // Can only access static members of outer class
            System.out.println("Accessing: " + staticOuterField);
            // System.out.println(outerField); // ERROR!
        }
    }
    
    // 3. Local Inner Class (inside a method)
    void methodWithLocalClass() {
        int localVar = 100;
        
        class LocalClass {
            void display() {
                System.out.println("Local class accessing: " + outerField);
                System.out.println("Local var (effectively final): " + localVar);
            }
        }
        
        LocalClass local = new LocalClass();
        local.display();
    }
    
    // 4. Anonymous Inner Class
    void useAnonymousClass() {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                System.out.println("Anonymous class running");
                System.out.println("Accessing: " + outerField);
            }
        };
        runnable.run();
    }
    
    public static void main(String[] args) {
        // Creating non-static inner class object
        OuterClass outer = new OuterClass();
        InnerClass inner = outer.new InnerClass();
        inner.display();
        
        // Creating static nested class object (no outer instance needed)
        StaticNestedClass staticNested = new StaticNestedClass();
        staticNested.display();
        
        outer.methodWithLocalClass();
        outer.useAnonymousClass();
    }
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is an inner class in Java?",
    options: [
      "A class inside another class",
      "A class that cannot be instantiated",
      "A class with only static methods",
      "A class that inherits from multiple classes",
    ],
    correctAnswer: 0,
    explanation: "An inner class is a class defined within another class. It has access to all members (including private) of the outer class.",
  },
  {
    id: 2,
    question: "What is the difference between inner class and static nested class?",
    options: [
      "No difference, they are the same",
      "Inner class needs outer instance, static nested doesn't",
      "Static nested class can access non-static outer members",
      "Inner class can only have static methods",
    ],
    correctAnswer: 1,
    explanation: "Inner class (non-static) requires an instance of the outer class and can access all outer members. Static nested class doesn't need outer instance and can only access static outer members.",
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

export default function InnerClassesPage() {
  const result = getSubtopicBySlug("oop", "inner-classes");

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
            What are Inner/Nested Classes?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Inner classes</strong> (also called nested classes) 
            are classes defined within another class. They help in logical grouping of classes that 
            are only used in one place, increasing encapsulation and creating more readable code.
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
            Types of Inner Classes
          </h2>
        </div>
        <CodeBlock code={javaInnerClassesCode} language="java" />
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

        {/* Question 35: What is an inner class? */}
        <QuestionCard
          number={35}
          title="Inner Class in Java"
          question="What is an inner class in Java? Why use it?"
          answer="An inner class is a class defined inside another class. Inner classes are used to: (1) logically group classes that are only used in one place, (2) increase encapsulation by hiding the inner class from outside, (3) access outer class members (including private), (4) create more readable and maintainable code. Inner classes have access to all members of the outer class."
          marks={5}
          icon={<Layers className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Benefits of Inner Classes:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>Logical grouping of related classes</li>
              <li>Increased encapsulation</li>
              <li>Access to outer class private members</li>
              <li>Used in event handling (GUI programming)</li>
              <li>Implementing helper classes</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`// Example: Building a simple linked list with inner Node class
public class LinkedList {
    private Node head;
    
    // Inner class - only used by LinkedList
    private class Node {
        int data;
        Node next;
        
        Node(int data) {
            this.data = data;
        }
    }
    
    public void add(int data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
        } else {
            Node current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
    }
    
    public void display() {
        Node current = head;
        while (current != null) {
            System.out.print(current.data + " -> ");
            current = current.next;
        }
        System.out.println("null");
    }
    
    public static void main(String[] args) {
        LinkedList list = new LinkedList();
        list.add(10);
        list.add(20);
        list.add(30);
        list.display();  // 10 -> 20 -> 30 -> null
    }
}`}
          />
        </QuestionCard>

        {/* Question 36: Create non-static inner class object */}
        <QuestionCard
          number={36}
          title="Creating Non-static Inner Class Object"
          question="How do you create an object of a non-static inner class?"
          answer="To create an object of a non-static inner class, you first need an instance of the outer class. Syntax: OuterClass.InnerClass innerObj = outerObj.new InnerClass(); The inner class object is always associated with an outer class instance, which is why it can access the outer class's instance variables."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class Outer {
    private String message = "Hello from Outer";
    
    class Inner {
        void show() {
            System.out.println(message);  // Can access outer's private member
        }
    }
    
    public static void main(String[] args) {
        // Step 1: Create outer class object first
        Outer outer = new Outer();
        
        // Step 2: Create inner class object using outer instance
        Outer.Inner inner = outer.new Inner();
        
        // Alternative - one line
        Outer.Inner inner2 = new Outer().new Inner();
        
        inner.show();  // Hello from Outer
        inner2.show(); // Hello from Outer
    }
}

// Key Points:
// - Cannot create inner class object directly: new Outer.Inner() // ERROR
// - Inner class object cannot exist without outer class object
// - Inner class has implicit reference to outer class: Outer.this`}
          />
        </QuestionCard>

        {/* Question 37: Static Nested vs Inner Class */}
        <QuestionCard
          number={37}
          title="Static Nested vs Inner Class"
          question="What is a static nested class? How is it different from an inner class?"
          answer="A static nested class is a class defined with 'static' keyword inside another class. Differences: (1) Static nested class doesn't require outer class instance; inner class does. (2) Static nested class can only access static members of outer class; inner class can access all members. (3) Static nested class objects can be created independently; inner class objects need outer instance. (4) Static nested class is like a top-level class nested for packaging."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Static Nested Class vs Inner Class"
            headers={["Feature", "Inner Class", "Static Nested Class"]}
            rows={[
              ["Keyword", "None (non-static)", "static"],
              ["Outer instance needed", "Yes", "No"],
              ["Can access non-static outer", "Yes", "No"],
              ["Can access static outer", "Yes", "Yes"],
              ["Creation syntax", "outer.new Inner()", "new Outer.StaticNested()"],
              ["Has reference to outer", "Yes (Outer.this)", "No"],
            ]}
          />
          <CodeBlock
            language="java"
            code={`public class Outer {
    private int instanceVar = 10;
    private static int staticVar = 20;
    
    // Non-static Inner Class
    class Inner {
        void display() {
            System.out.println("Inner can access: " + instanceVar);  // OK
            System.out.println("Inner can access: " + staticVar);    // OK
        }
    }
    
    // Static Nested Class
    static class StaticNested {
        void display() {
            // System.out.println(instanceVar);  // ERROR!
            System.out.println("StaticNested can access: " + staticVar);  // OK only
        }
    }
    
    public static void main(String[] args) {
        // Inner class - needs outer instance
        Outer outer = new Outer();
        Inner inner = outer.new Inner();
        inner.display();
        
        // Static nested class - no outer instance needed
        StaticNested staticNested = new StaticNested();
        staticNested.display();
        
        // Also valid
        Outer.StaticNested nested2 = new Outer.StaticNested();
    }
}`}
          />
        </QuestionCard>

        {/* Question 38: Inner class accessing outer members */}
        <QuestionCard
          number={38}
          title="Inner Class Accessing Outer Members"
          question="Write code to demonstrate an inner class accessing outer class members."
          answer="Inner class has access to all members (including private) of the outer class. It can access them directly by name, or use 'OuterClass.this.member' for clarity when there are naming conflicts."
          marks={5}
          icon={<Eye className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class BankAccount {
    private double balance = 1000;
    private static String bankName = "MyBank";
    
    // Inner class representing a transaction
    class Transaction {
        private double amount;
        
        Transaction(double amount) {
            this.amount = amount;
        }
        
        void deposit() {
            // Direct access to outer class private member
            balance += amount;
            System.out.println("Deposited: $" + amount);
            System.out.println("New balance: $" + balance);
        }
        
        void withdraw() {
            if (amount <= balance) {
                balance -= amount;
                System.out.println("Withdrawn: $" + amount);
                System.out.println("New balance: $" + balance);
            } else {
                System.out.println("Insufficient balance!");
            }
        }
        
        void displayInfo() {
            // Access static member
            System.out.println("Bank: " + bankName);
            
            // Using OuterClass.this to explicitly access outer
            System.out.println("Current balance: $" + BankAccount.this.balance);
            
            // When there's no naming conflict, direct access works
            System.out.println("Transaction amount: $" + amount);
        }
    }
    
    void performTransaction(double amt) {
        Transaction t = new Transaction(amt);
        t.deposit();
        t.displayInfo();
    }
    
    public static void main(String[] args) {
        BankAccount account = new BankAccount();
        BankAccount.Transaction t1 = account.new Transaction(500);
        t1.deposit();
        
        BankAccount.Transaction t2 = account.new Transaction(200);
        t2.withdraw();
        
        t2.displayInfo();
    }
}

/* Output:
Deposited: $500.0
New balance: $1500.0
Withdrawn: $200.0
New balance: $1300.0
Bank: MyBank
Current balance: $1300.0
Transaction amount: $200.0
*/`}
          />
        </QuestionCard>

        {/* Question 39: Anonymous Inner Class */}
        <QuestionCard
          number={39}
          title="Anonymous Inner Class"
          question="What is an anonymous inner class? Give an example."
          answer="An anonymous inner class is an inner class without a name that is declared and instantiated in a single expression. It is typically used to override methods of a class or interface. Anonymous classes are useful when you need a one-time implementation of an interface or abstract class, especially in event handling and callback mechanisms."
          marks={5}
          icon={<Lock className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Characteristics of Anonymous Inner Classes:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>No name - declared and instantiated at the same time</li>
              <li>Must extend a class or implement an interface</li>
              <li>Cannot have explicit constructors</li>
              <li>Cannot define static members</li>
              <li>Common in GUI event handling and threading</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`// Example 1: Anonymous class extending Thread
public class AnonymousDemo {
    
    // Interface for callback
    interface Greeting {
        void greet();
    }
    
    // Abstract class
    abstract class Animal {
        abstract void sound();
    }
    
    public static void main(String[] args) {
        // Anonymous class implementing interface
        Greeting greeting = new Greeting() {
            @Override
            public void greet() {
                System.out.println("Hello from anonymous class!");
            }
        };
        greeting.greet();
        
        // Anonymous class extending abstract class
        Animal dog = new Animal() {
            @Override
            void sound() {
                System.out.println("Woof! Woof!");
            }
        };
        dog.sound();
        
        // Anonymous class with additional method
        Object obj = new Object() {
            void extraMethod() {
                System.out.println("Extra method in anonymous class");
            }
            
            @Override
            public String toString() {
                extraMethod();
                return "Custom toString";
            }
        };
        System.out.println(obj.toString());
        
        // Real-world usage: GUI event handling
        javax.swing.JButton button = new javax.swing.JButton("Click Me");
        button.addActionListener(new java.awt.event.ActionListener() {
            @Override
            public void actionPerformed(java.awt.event.ActionEvent e) {
                System.out.println("Button clicked!");
            }
        });
        
        // Using lambda expressions (Java 8+) - simpler alternative
        button.addActionListener(e -> System.out.println("Lambda click!"));
    }
}`}
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
              <span className="font-bold text-primary">Q35:</span>
              <span className="text-muted-foreground ml-1">Inner class = class inside another class - for encapsulation</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q36:</span>
              <span className="text-muted-foreground ml-1">Inner class object: outer.new Inner()</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q37:</span>
              <span className="text-muted-foreground ml-1">Static nested: no outer instance needed, can't access non-static</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q38:</span>
              <span className="text-muted-foreground ml-1">Inner class can access all outer members (including private)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q39:</span>
              <span className="text-muted-foreground ml-1">Anonymous inner class = no name, instantiated once</span>
            </div>
          </div>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="Inner Classes Quiz" />
    </TopicContent>
  );
}

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