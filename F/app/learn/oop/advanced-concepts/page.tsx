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
  GitBranch,
  Box,
  MessageSquare,
  Eye,
  Shield,
  AlertCircle,
  Zap,
} from "lucide-react";

const advancedConceptsCode = `// Advanced OOP Concepts Examples

// 1. Message Passing
class MessageSender {
    public void sendMessage(MessageReceiver receiver, String content) {
        // Message passing: sender calls receiver's method
        receiver.receive(content);
    }
}

class MessageReceiver {
    public void receive(String message) {
        System.out.println("Received: " + message);
    }
}

// 2. Composition over Inheritance (good example)
// Before - Bad: Deep inheritance
class Animal { }
class Bird extends Animal { }
class FlyingBird extends Bird { }
class Eagle extends FlyingBird { }  // Too deep!

// After - Good: Composition
class FlyBehavior {
    public void fly() { System.out.println("Flying..."); }
}

class BirdGood {
    private FlyBehavior flyBehavior;  // Composition
    
    public BirdGood(FlyBehavior behavior) {
        this.flyBehavior = behavior;
    }
    
    public void performFly() {
        flyBehavior.fly();
    }
}

// 3. Law of Demeter (violation vs compliance)
// Violation - chaining multiple method calls
class Address {
    private String city;
    public String getCity() { return city; }
}

class Employee {
    private Address address;
    public Address getAddress() { return address; }
}

class Company {
    private Employee employee;
    public Employee getEmployee() { return employee; }
    
    // Violation: Law of Demeter
    public void printCityBad() {
        String city = getEmployee().getAddress().getCity();
        System.out.println(city);
    }
    
    // Compliant: Don't traverse through objects
    public void printCityGood() {
        String city = getEmployeeCity();
        System.out.println(city);
    }
    
    public String getEmployeeCity() {
        return employee.getCity();
    }
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is message passing in OOP?",
    options: [
      "Sending emails between objects",
      "Objects communicating by invoking methods on each other",
      "Passing data between functions",
      "Network communication between systems",
    ],
    correctAnswer: 1,
    explanation: "Message passing is the mechanism where objects communicate by sending messages (invoking methods) to each other.",
  },
  {
    id: 2,
    question: "What is the difference between OOP and procedural programming?",
    options: [
      "No difference",
      "OOP uses objects, procedural uses functions/procedures",
      "Procedural is faster",
      "OOP doesn't support data hiding",
    ],
    correctAnswer: 1,
    explanation: "OOP organizes code around objects with data and methods; procedural organizes around functions and procedures.",
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

export default function AdvancedOOPConceptsPage() {
  const result = getSubtopicBySlug("oop", "advanced-concepts");

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
            Advanced OOP Concepts
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            These advanced OOP concepts explore deeper aspects of object-oriented programming including 
            paradigm comparisons, runtime features, design principles, and best practices.
          </p>
        </div>
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

        {/* Question 83: OOP vs Procedural Programming */}
        <QuestionCard
          number={83}
          title="OOP vs Procedural Programming"
          question="Explain the difference between OOP and procedural programming."
          answer="OOP organizes code around objects that contain both data and methods, supporting encapsulation, inheritance, and polymorphism. Procedural programming organizes code around functions/procedures that operate on separate data structures. OOP promotes code reuse and modularity; procedural is simpler for small programs but harder to maintain at scale."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <DifferenceTable
            title="OOP vs Procedural Programming"
            headers={["Feature", "OOP", "Procedural"]}
            rows={[
              ["Organization", "Objects (data + methods)", "Functions + separate data"],
              ["Data Security", "Encapsulation (private data)", "Global data accessible everywhere"],
              ["Code Reuse", "Inheritance & Composition", "Function libraries"],
              ["Modularity", "High (classes as modules)", "Low (functions are loosely coupled)"],
              ["Maintenance", "Easier for large systems", "Harder for large systems"],
              ["Performance", "Slight overhead", "Faster (less abstraction)"],
              ["Languages", "Java, C++, Python, C#", "C, Fortran, Pascal"],
              ["Best for", "Large, complex systems", "Small, simple programs"],
            ]}
          />
          <CodeBlock
            language="java"
            code={`// PROCEDURAL APPROACH (C-style in Java)
class ProceduralStyle {
    static String name;
    static int age;
    
    static void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }
    
    public static void main(String[] args) {
        name = "John";
        age = 25;
        display();  // Data and functions are separate
    }
}

// OOP APPROACH
class Person {
    private String name;  // Data encapsulated
    private int age;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public void display() {  // Method with data
        System.out.println("Name: " + name + ", Age: " + age);
    }
}

// OOP allows polymorphism and inheritance
class Employee extends Person {
    private double salary;
    
    public Employee(String name, int age, double salary) {
        super(name, age);
        this.salary = salary;
    }
    
    @Override
    public void display() {
        super.display();
        System.out.println("Salary: " + salary);
    }
}

// Procedural vs OOP Code Comparison
// Procedural: Separate data and functions
struct PersonData { char name[50]; int age; };  // C-style
void displayPerson(struct PersonData p);

// OOP: Encapsulated data and behavior
class PersonOOP {
    private String name;
    private int age;
    public void display() { /* ... */ }
}`}
          />
        </QuestionCard>

        {/* Question 84: Message Passing */}
        <QuestionCard
          number={84}
          title="Message Passing in OOP"
          question="What is message passing in OOP?"
          answer="Message passing is the mechanism by which objects communicate with each other. When one object wants another object to perform an action, it sends a 'message' by invoking a method on that object. This fundamental OOP concept enables collaboration between objects and supports encapsulation (objects only expose what's necessary)."
          marks={5}
          icon={<MessageSquare className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Message Passing Example
class BankAccount {
    private double balance;
    private String accountNumber;
    
    public BankAccount(String accNo, double initialBalance) {
        this.accountNumber = accNo;
        this.balance = initialBalance;
    }
    
    // These methods are the "messages" that can be sent
    public void deposit(double amount) {
        balance += amount;
        System.out.println("Deposited: " + amount);
    }
    
    public boolean withdraw(double amount) {
        if (amount <= balance) {
            balance -= amount;
            System.out.println("Withdrawn: " + amount);
            return true;
        }
        System.out.println("Insufficient funds!");
        return false;
    }
    
    public double getBalance() {
        return balance;
    }
}

class Customer {
    private String name;
    private BankAccount account;
    
    public Customer(String name, BankAccount account) {
        this.name = name;
        this.account = account;
    }
    
    // Customer sends messages to BankAccount
    public void makeDeposit(double amount) {
        System.out.println(name + " is depositing...");
        account.deposit(amount);  // Message passing!
    }
    
    public void makeWithdrawal(double amount) {
        System.out.println(name + " is withdrawing...");
        account.withdraw(amount);  // Message passing!
    }
    
    public void checkBalance() {
        System.out.println(name + "'s balance: " + account.getBalance());  // Message passing!
    }
}

// Multiple objects communicating
class ATM {
    private BankAccount currentAccount;
    
    public void insertCard(BankAccount account) {
        this.currentAccount = account;
        System.out.println("Card inserted");
    }
    
    public void withdraw(double amount) {
        if (currentAccount != null) {
            currentAccount.withdraw(amount);  // ATM sends message to account
        }
    }
}

// Real-world message passing example
class OrderProcessor {
    private PaymentGateway paymentGateway;
    private InventorySystem inventory;
    private EmailService emailService;
    
    public OrderProcessor(PaymentGateway pg, InventorySystem inv, EmailService es) {
        this.paymentGateway = pg;
        this.inventory = inv;
        this.emailService = es;
    }
    
    public void processOrder(Order order) {
        // Multiple message passing examples
        boolean paymentSuccess = paymentGateway.charge(order.getAmount());
        
        if (paymentSuccess) {
            inventory.reserveItems(order.getItems());
            emailService.sendConfirmation(order.getCustomerEmail());
            order.markAsProcessed();
        }
    }
}

// Usage
public class MessagePassingDemo {
    public static void main(String[] args) {
        BankAccount account = new BankAccount("12345", 1000);
        Customer customer = new Customer("Alice", account);
        
        // Message passing in action
        customer.checkBalance();      // Send message to get balance
        customer.makeDeposit(500);    // Send message to deposit
        customer.makeWithdrawal(200); // Send message to withdraw
        customer.checkBalance();      // Send message again
    }
}

/* Output:
Alice's balance: 1000.0
Alice is depositing...
Deposited: 500.0
Alice is withdrawing...
Withdrawn: 200.0
Alice's balance: 1300.0
*/`}
          />
        </QuestionCard>

        {/* Question 85: Object-Oriented vs Object-Based */}
        <QuestionCard
          number={85}
          title="Object-Oriented vs Object-Based Languages"
          question="What is the difference between object-oriented and object-based languages?"
          answer="Object-oriented languages support all four pillars: encapsulation, inheritance, polymorphism, and abstraction. Object-based languages support objects and encapsulation but NOT inheritance and polymorphism. Examples: OOP - Java, C++, Python, C#. Object-based - JavaScript (before ES6), Visual Basic (pre-.NET), Ada. Object-based languages can have objects but lack the hierarchical relationships."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Object-Oriented vs Object-Based Languages"
            headers={["Feature", "Object-Oriented", "Object-Based"]}
            rows={[
              ["Encapsulation", "✓ Yes", "✓ Yes"],
              ["Inheritance", "✓ Yes", "✗ No"],
              ["Polymorphism", "✓ Yes", "✗ No"],
              ["Abstraction", "✓ Yes", "Partial"],
              ["Examples", "Java, C++, Python, C#", "JavaScript (pre-ES6), VB6, Ada"],
              ["Code Reuse", "High (via inheritance)", "Low (no inheritance)"],
              ["Dynamic Binding", "✓ Yes", "Limited"],
            ]}
          />
          <CodeBlock
            language="java"
            code={`// OBJECT-ORIENTED LANGUAGE (Java)
// Supports inheritance and polymorphism
class Animal {
    public void sound() {
        System.out.println("Animal makes sound");
    }
}

class Dog extends Animal {  // Inheritance
    @Override
    public void sound() {   // Polymorphism
        System.out.println("Dog barks");
    }
}

class Cat extends Animal {
    @Override
    public void sound() {
        System.out.println("Cat meows");
    }
}

// OBJECT-BASED (JavaScript before ES6 - no classical inheritance)
// JavaScript object example (object-based)
var person = {
    name: "John",
    age: 25,
    greet: function() {
        console.log("Hello, I'm " + this.name);
    }
};

// Prototype-based (not classical inheritance)
function Animal(name) {
    this.name = name;
}

Animal.prototype.sound = function() {
    console.log("Some sound");
};

// This is prototype chaining, not classical inheritance
function Dog(name) {
    Animal.call(this, name);
}

Dog.prototype = Object.create(Animal.prototype);

// OBJECT-BASED (Visual Basic 6 - no inheritance)
' VB6 Code (object-based)
' Class Module: clsPerson
Private m_Name As String
Public Property Get Name() As String
    Name = m_Name
End Property
Public Property Let Name(ByVal vNewValue As String)
    m_Name = vNewValue
End Property
' No inheritance support!

// Modern JavaScript (ES6+) is fully object-oriented
class ModernAnimal {
    sound() {
        console.log("Some sound");
    }
}

class ModernDog extends ModernAnimal {  // Now supports inheritance
    sound() {
        console.log("Woof!");
    }
}`}
          />
        </QuestionCard>

        {/* Question 86: RTTI (Run-Time Type Identification) */}
        <QuestionCard
          number={86}
          title="RTTI (Run-Time Type Identification)"
          question="What is RTTI (Run-Time Type Identification)?"
          answer="RTTI is a mechanism that allows determining the actual type of an object at runtime. In C++, RTTI is provided via typeid operator and dynamic_cast. In Java, instanceof operator and getClass() method provide similar functionality. RTTI enables safe downcasting and type checking, but should be used sparingly as it may indicate poor design (violating polymorphism)."
          marks={5}
          icon={<Eye className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Java RTTI Examples
class Animal { }
class Dog extends Animal { 
    void bark() { System.out.println("Woof!"); }
}
class Cat extends Animal { 
    void meow() { System.out.println("Meow!"); }
}

public class RTTIDemo {
    
    // instanceof operator (Java RTTI)
    public static void identifyAnimal(Animal animal) {
        if (animal instanceof Dog) {
            System.out.println("This is a Dog");
            Dog d = (Dog) animal;  // Downcasting
            d.bark();
        } 
        else if (animal instanceof Cat) {
            System.out.println("This is a Cat");
            Cat c = (Cat) animal;
            c.meow();
        }
        else {
            System.out.println("Unknown animal");
        }
    }
    
    // getClass() method
    public static void printType(Object obj) {
        System.out.println("Class: " + obj.getClass().getName());
        System.out.println("Simple name: " + obj.getClass().getSimpleName());
    }
    
    public static void main(String[] args) {
        Animal a1 = new Dog();
        Animal a2 = new Cat();
        
        identifyAnimal(a1);  // Output: This is a Dog, Woof!
        identifyAnimal(a2);  // Output: This is a Cat, Meow!
        
        printType(a1);  // Output: Class: Dog, Simple name: Dog
        printType("Hello");  // Output: Class: java.lang.String
        
        // Using getClass for type comparison
        if (a1.getClass() == Dog.class) {
            System.out.println("a1 is exactly a Dog");
        }
    }
}

// C++ RTTI Example (commented)
/*
#include <iostream>
#include <typeinfo>
using namespace std;

class Animal {
public:
    virtual ~Animal() {}  // Need virtual for RTTI
};

class Dog : public Animal {
public:
    void bark() { cout << "Woof!" << endl; }
};

class Cat : public Animal {
public:
    void meow() { cout << "Meow!" << endl; }
};

int main() {
    Animal* a1 = new Dog();
    Animal* a2 = new Cat();
    
    // typeid operator
    cout << "Type: " << typeid(*a1).name() << endl;
    
    // dynamic_cast for safe downcasting
    Dog* d = dynamic_cast<Dog*>(a1);
    if (d) {
        d->bark();  // Safe
    }
    
    Cat* c = dynamic_cast<Cat*>(a1);
    if (c) {  // Will be null
        c->meow();
    }
    
    return 0;
}
*/

// When to use (and NOT use) RTTI
class Shape {
    public double area() { return 0; }
}

class Circle extends Shape {
    private double radius;
    public Circle(double r) { radius = r; }
    public double area() { return Math.PI * radius * radius; }
    public double getRadius() { return radius; }
}

class Rectangle extends Shape {
    private double width, height;
    public Rectangle(double w, double h) { width = w; height = h; }
    public double area() { return width * height; }
}

// BAD: Using RTTI to determine behavior (violates polymorphism)
void processBad(Shape shape) {
    if (shape instanceof Circle) {
        Circle c = (Circle) shape;
        System.out.println("Circle radius: " + c.getRadius());
    } else if (shape instanceof Rectangle) {
        // Rectangle-specific logic
    }
}

// GOOD: Using polymorphism instead
void processGood(Shape shape) {
    System.out.println("Area: " + shape.area());  // No RTTI needed
}`}
          />
        </QuestionCard>

        {/* Question 87: Reflection in Java */}
        <QuestionCard
          number={87}
          title="Reflection in Java"
          question="What is reflection in Java? How can it break encapsulation?"
          answer="Reflection is a feature that allows inspecting and modifying classes, methods, fields at runtime, even private ones. It breaks encapsulation because private fields/methods can be accessed and modified from outside the class using setAccessible(true). This violates the principle of data hiding. Reflection should be used sparingly - for frameworks, debugging, or serialization, not normal application code."
          marks={5}
          icon={<AlertCircle className="h-3 w-3" />}
        >
          <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg mb-4">
            <h4 className="font-medium text-destructive mb-1">⚠️ Security Warning</h4>
            <p className="text-sm text-muted-foreground">
              Reflection can break encapsulation and bypass access controls. Use only when necessary 
              (frameworks, testing, debugging) and never in production business logic unless absolutely required.
            </p>
          </div>
          <CodeBlock
            language="java"
            code={`import java.lang.reflect.*;

class SecureBankAccount {
    private double balance;
    private String accountNumber;
    private static final String BANK_SECRET = "TOP_SECRET_KEY";
    
    public SecureBankAccount(String accNo, double initialBalance) {
        this.accountNumber = accNo;
        this.balance = initialBalance;
    }
    
    private void validateTransaction(double amount) {
        if (amount < 0) {
            throw new IllegalArgumentException("Negative amount!");
        }
    }
    
    private void auditLog(String action) {
        System.out.println("AUDIT: " + action + " on account " + accountNumber);
    }
    
    public double getBalance() {
        return balance;
    }
}

public class ReflectionDemo {
    
    // Reflection breaking encapsulation
    public static void breakEncapsulation() throws Exception {
        SecureBankAccount account = new SecureBankAccount("12345", 1000);
        
        System.out.println("Original balance: " + account.getBalance());
        
        // Get private field
        Field balanceField = SecureBankAccount.class.getDeclaredField("balance");
        balanceField.setAccessible(true);  // BREAKS ENCAPSULATION!
        
        // Modify private field directly
        balanceField.setDouble(account, 999999.99);
        System.out.println("Modified balance: " + account.getBalance());
        
        // Access private static final field
        Field secretField = SecureBankAccount.class.getDeclaredField("BANK_SECRET");
        secretField.setAccessible(true);
        System.out.println("Bank secret: " + secretField.get(null));
        
        // Invoke private method
        Method auditMethod = SecureBankAccount.class.getDeclaredMethod("auditLog", String.class);
        auditMethod.setAccessible(true);
        auditMethod.invoke(account, "UNAUTHORIZED ACCESS");
        
        // Get all methods (including private)
        System.out.println("\\nAll methods:");
        for (Method m : SecureBankAccount.class.getDeclaredMethods()) {
            System.out.println("  " + m.getName() + " (private: " + 
                java.lang.reflect.Modifier.isPrivate(m.getModifiers()) + ")");
        }
    }
    
    // Legitimate uses of reflection
    public static void legitimateUses() throws Exception {
        // 1. Dynamic object creation (Factory patterns)
        Class<?> clazz = Class.forName("java.util.ArrayList");
        Object list = clazz.getDeclaredConstructor().newInstance();
        System.out.println("Created: " + list.getClass().getSimpleName());
        
        // 2. Inspecting class structure (IDE autocomplete)
        Method[] methods = String.class.getMethods();
        System.out.println("String has " + methods.length + " public methods");
        
        // 3. Annotation processing
        Class<MyClass> objClass = MyClass.class;
        for (Annotation anno : objClass.getAnnotations()) {
            System.out.println("Annotation: " + anno);
        }
    }
    
    @Deprecated
    class MyClass { }
    
    public static void main(String[] args) throws Exception {
        System.out.println("=== BREAKING ENCAPSULATION ===\n");
        breakEncapsulation();
        
        System.out.println("\n=== LEGITIMATE USES ===\n");
        legitimateUses();
    }
}

// How to protect against reflection?
// 1. SecurityManager (deprecated in newer Java)
// 2. Use final fields (still accessible via reflection)
// 3. Use Sealed classes (Java 17+)
// 4. Use modules with limited exports (Java 9+)

// Modern protection: Module system (module-info.java)
/*
module my.module {
    exports com.mypackage.api;
    // Reflection not allowed on non-exported packages
}
*/`}
          />
        </QuestionCard>

        {/* Question 88: Composition over Inheritance */}
        <QuestionCard
          number={88}
          title="Composition over Inheritance"
          question="Explain composition over inheritance with an example."
          answer="'Composition over inheritance' is a design principle preferring object composition (has-a) over class inheritance (is-a). Composition provides more flexibility, reduces tight coupling, and avoids deep inheritance hierarchies. Use inheritance for true 'is-a' relationships; use composition for 'has-a' or when behavior needs to change dynamically."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Problems with Deep Inheritance:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>Fragile base class problem - changes to base class break all derived classes</li>
              <li>Inflexible - behavior fixed at compile time</li>
              <li>Can't change behavior at runtime</li>
              <li>Deep hierarchies are hard to understand</li>
              <li>Multiple inheritance causes diamond problem</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`// ❌ BAD: Deep inheritance hierarchy
class Animal {
    void eat() { System.out.println("Eating"); }
}

class Bird extends Animal {
    void fly() { System.out.println("Flying"); }
}

class Sparrow extends Bird { }
class Eagle extends Bird { }
class Penguin extends Bird { 
    @Override
    void fly() { 
        throw new UnsupportedOperationException("Can't fly!"); 
    }
}

// ✅ GOOD: Composition approach
// Define behaviors as separate interfaces/classes
interface FlyBehavior {
    void fly();
}

class FlyWithWings implements FlyBehavior {
    public void fly() { System.out.println("Flying with wings!"); }
}

class FlyNoWay implements FlyBehavior {
    public void fly() { System.out.println("Cannot fly!"); }
}

class FlyWithRocket implements FlyBehavior {
    public void fly() { System.out.println("Flying with rocket power!"); }
}

interface SoundBehavior {
    void makeSound();
}

class BarkSound implements SoundBehavior {
    public void makeSound() { System.out.println("Woof! Woof!"); }
}

class MeowSound implements SoundBehavior {
    public void makeSound() { System.out.println("Meow! Meow!"); }
}

class QuackSound implements SoundBehavior {
    public void makeSound() { System.out.println("Quack! Quack!"); }
}

// Animal class using composition
class AnimalGood {
    private String name;
    private FlyBehavior flyBehavior;
    private SoundBehavior soundBehavior;
    
    public AnimalGood(String name, FlyBehavior fly, SoundBehavior sound) {
        this.name = name;
        this.flyBehavior = fly;
        this.soundBehavior = sound;
    }
    
    public void performFly() {
        System.out.print(name + ": ");
        flyBehavior.fly();
    }
    
    public void performSound() {
        System.out.print(name + ": ");
        soundBehavior.makeSound();
    }
    
    // Dynamic behavior change at runtime!
    public void setFlyBehavior(FlyBehavior behavior) {
        this.flyBehavior = behavior;
    }
    
    public void setSoundBehavior(SoundBehavior behavior) {
        this.soundBehavior = behavior;
    }
}

// Usage - flexible composition
public class CompositionDemo {
    public static void main(String[] args) {
        // Create animals with different behaviors
        AnimalGood dog = new AnimalGood("Rex", new FlyNoWay(), new BarkSound());
        AnimalGood cat = new AnimalGood("Whiskers", new FlyNoWay(), new MeowSound());
        AnimalGood duck = new AnimalGood("Donald", new FlyWithWings(), new QuackSound());
        
        dog.performSound();  // Rex: Woof! Woof!
        dog.performFly();    // Rex: Cannot fly!
        
        duck.performSound(); // Donald: Quack! Quack!
        duck.performFly();   // Donald: Flying with wings!
        
        // Dynamic behavior change at runtime!
        System.out.println("\\n=== Rocket Duck! ===");
        duck.setFlyBehavior(new FlyWithRocket());
        duck.performFly();    // Donald: Flying with rocket power!
        
        // Benefits of composition:
        // 1. Flexible - change behavior at runtime
        // 2. Reusable - behaviors can be shared
        // 3. Testable - mock behaviors easily
        // 4. Avoids deep inheritance
    }
}

// Real-world example: UI Components
// Bad inheritance approach
class UIComponent { }
class ClickableComponent extends UIComponent { }
class DraggableComponent extends UIComponent { }
class ClickableDraggableComponent extends ClickableComponent { } // Multiple inheritance issue!

// Good composition approach
interface ClickHandler {
    void onClick();
}

interface DragHandler {
    void onDrag();
}

class ClickBehavior implements ClickHandler {
    public void onClick() { System.out.println("Clicked!"); }
}

class DragBehavior implements DragHandler {
    public void onDrag() { System.out.println("Dragged!"); }
}

class Component {
    private ClickHandler clickHandler;
    private DragHandler dragHandler;
    
    public Component(ClickHandler click, DragHandler drag) {
        this.clickHandler = click;
        this.dragHandler = drag;
    }
    
    public void handleClick() {
        if (clickHandler != null) clickHandler.onClick();
    }
    
    public void handleDrag() {
        if (dragHandler != null) dragHandler.onDrag();
    }
}

// Create any combination easily
Component clickableOnly = new Component(new ClickBehavior(), null);
Component draggableOnly = new Component(null, new DragBehavior());
Component both = new Component(new ClickBehavior(), new DragBehavior());`}
          />
        </QuestionCard>

        {/* Question 89: Law of Demeter */}
        <QuestionCard
          number={89}
          title="Law of Demeter"
          question="What is the Law of Demeter?"
          answer="The Law of Demeter (LoD) or 'Principle of Least Knowledge' states that an object should only communicate with its immediate collaborators, not with strangers. It suggests that a method should only call methods on: itself, its parameters, objects it creates, its direct fields, or constants. Violations appear as long chains like 'a.getB().getC().doSomething()'. Following LoD reduces coupling and improves maintainability."
          marks={5}
          icon={<Shield className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Law of Demeter - Allowed Method Calls:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>On the object itself (this.method())</li>
              <li>On method parameters (param.method())</li>
              <li>On objects created within the method (new Object().method())</li>
              <li>On directly held fields (field.method())</li>
              <li>On constants/static methods (Class.staticMethod())</li>
            </ul>
            <p className="text-sm text-muted-foreground mt-2">
              <strong className="text-destructive">NOT allowed:</strong> a.getB().getC().getD().doSomething()
            </p>
          </div>
          <CodeBlock
            language="java"
            code={`// Violation of Law of Demeter
class Address {
    private String city;
    private String street;
    private String zipCode;
    
    public String getCity() { return city; }
    public String getStreet() { return street; }
    public String getZipCode() { return zipCode; }
}

class Employee {
    private Address address;
    private String name;
    
    public Address getAddress() { return address; }
    public String getName() { return name; }
}

class Department {
    private Employee manager;
    
    public Employee getManager() { return manager; }
}

class Company {
    private Department department;
    
    public Department getDepartment() { return department; }
    
    // ❌ VIOLATION: Multiple dots chaining - Law of Demeter violation
    public void printManagerCityBad() {
        // Company → Department → Employee → Address → getCity()
        String city = getDepartment().getManager().getAddress().getCity();
        System.out.println("Manager's city: " + city);
    }
}

// ✅ COMPLIANT: Law of Demeter followed
class EmployeeGood {
    private AddressGood address;
    private String name;
    
    // Provide direct access to needed information
    public String getCity() {
        return address != null ? address.getCity() : null;
    }
    
    public String getName() { return name; }
}

class AddressGood {
    private String city;
    private String street;
    
    public String getCity() { return city; }
}

class DepartmentGood {
    private EmployeeGood manager;
    
    public String getManagerCity() {
        return manager != null ? manager.getCity() : null;
    }
}

class CompanyGood {
    private DepartmentGood department;
    
    // Compliant: Only one dot
    public void printManagerCityGood() {
        String city = getManagerCity();
        System.out.println("Manager's city: " + city);
    }
    
    public String getManagerCity() {
        return department != null ? department.getManagerCity() : null;
    }
}

// Complete example showing refactoring
class OrderViolation {
    private Customer customer;
    
    public void processOrder() {
        // VIOLATION: Chaining through multiple objects
        String city = customer.getAddress().getCity();
        String phone = customer.getAddress().getPhoneNumber();
        boolean isValid = customer.getAddress().getZipCode().isValid();
        
        // VIOLATION: Accessing internal objects directly
        customer.getCart().getItems().forEach(item -> {
            item.getProduct().applyDiscount();
        });
    }
}

class OrderCompliant {
    private CustomerCompliant customer;
    
    public void processOrder() {
        // COMPLIANT: Ask the customer directly
        String city = customer.getCity();
        String phone = customer.getPhoneNumber();
        boolean isValid = customer.isZipCodeValid();
        
        // COMPLIANT: Delegate to customer
        customer.applyDiscountsToCart();
    }
}

class CustomerCompliant {
    private Address address;
    private ShoppingCart cart;
    
    public String getCity() {
        return address != null ? address.getCity() : null;
    }
    
    public String getPhoneNumber() {
        return address != null ? address.getPhoneNumber() : null;
    }
    
    public boolean isZipCodeValid() {
        return address != null && address.getZipCode().isValid();
    }
    
    public void applyDiscountsToCart() {
        cart.applyDiscounts();
    }
}

// Real-world example: UI Component violation
class UIViolation {
    private JPanel panel;
    
    public void doSomething() {
        // VIOLATION: Chaining through UI hierarchy
        Color color = panel.getBorder().getBorderColor();
        Font font = panel.getLabel().getFont();
        String text = panel.getTextField().getText();
    }
}

class UICompliant {
    private CustomPanel panel;
    
    public void doSomething() {
        // COMPLIANT: Panel provides needed info
        Color color = panel.getBorderColor();
        Font font = panel.getLabelFont();
        String text = panel.getTextFieldText();
    }
}

class CustomPanel {
    private Border border;
    private Label label;
    private TextField textField;
    
    public Color getBorderColor() {
        return border != null ? border.getColor() : null;
    }
    
    public Font getLabelFont() {
        return label != null ? label.getFont() : null;
    }
    
    public String getTextFieldText() {
        return textField != null ? textField.getText() : "";
    }
}

// Benefits of following Law of Demeter:
// 1. Reduced coupling between classes
// 2. Easier to test (mock fewer dependencies)
// 3. More maintainable (changes localized)
// 4. Better encapsulation`}
          />
        </QuestionCard>

        {/* Summary Section */}
        <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q83:</span>
              <span className="text-muted-foreground ml-1">OOP = objects + data + methods; Procedural = functions + data</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q84:</span>
              <span className="text-muted-foreground ml-1">Message passing = objects communicating via method calls</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q85:</span>
              <span className="text-muted-foreground ml-1">OOP supports inheritance/polymorphism; Object-based does not</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q86:</span>
              <span className="text-muted-foreground ml-1">RTTI = instanceof, getClass() (Java); typeid, dynamic_cast (C++)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q87:</span>
              <span className="text-muted-foreground ml-1">Reflection = inspect/modify classes at runtime (can break encapsulation)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q88:</span>
              <span className="text-muted-foreground ml-1">Composition (has-a) over inheritance (is-a) for flexibility</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q89:</span>
              <span className="text-muted-foreground ml-1">Law of Demeter = only talk to immediate friends (no chaining)</span>
            </div>
          </div>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="Advanced OOP Concepts Quiz" />
    </TopicContent>
  );
}