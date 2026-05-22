"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock } from "@/components/code-block";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Code,
  Target,
  GitBranch,
  Box,
  Eye,
  MessageSquare,
  Zap,
  Shield,
  Link,
  Users,
  Bell,
  Coffee,
  AlertCircle,
} from "lucide-react";

// Quick Reference Card Component
interface QuickRefCardProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  bgColor?: string;
}

const QuickRefCard: React.FC<QuickRefCardProps> = ({ title, icon, children, bgColor = "bg-primary/5" }) => {
  return (
    <div className={`mb-6 rounded-lg border border-border overflow-hidden ${bgColor}`}>
      <div className="p-4 border-b border-border bg-card/50">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-primary/10">
            {icon}
          </div>
          <h3 className="text-lg font-semibold text-foreground">{title}</h3>
        </div>
      </div>
      <div className="p-4">
        {children}
      </div>
    </div>
  );
};

// Table Component
const SimpleTable: React.FC<{
  headers: string[];
  rows: Array<Array<string | React.ReactNode>>;
}> = ({ headers, rows }) => {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-muted">
            {headers.map((header, idx) => (
              <th key={idx} className="border border-border p-2 text-left font-semibold text-foreground">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} className="even:bg-muted/30">
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} className="border border-border p-2 text-muted-foreground">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default function QuickRevisionGuidePage() {
  const result = getSubtopicBySlug("oop", "quick-reference");

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
      <section className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-foreground mb-3">
          OOP Quick Revision Guide
        </h1>
        <p className="text-muted-foreground">
          Essential concepts for exams and interviews - 5-minute refresher
        </p>
        <div className="mt-4 flex justify-center gap-2 flex-wrap">
          <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm">16 Topics</span>
          <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm">Quick Reference</span>
          <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm">Code Examples</span>
        </div>
      </section>

      {/* 1. instanceof Operator */}
      <QuickRefCard title="instanceof Operator (Java)" icon={<Eye className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Purpose:</strong> Checks whether an object is an instance of a specific class or interface.
          Returns <code className="bg-muted px-1 rounded">true</code> if the object is an instance, <code className="bg-muted px-1 rounded">false</code> otherwise.
        </p>
        <CodeBlock
          language="java"
          code={`// instanceof operator examples
class Animal { }
class Dog extends Animal { }
class Cat extends Animal { }

public class InstanceofDemo {
    public static void main(String[] args) {
        Animal a = new Dog();
        
        // Basic usage
        System.out.println(a instanceof Dog);     // true
        System.out.println(a instanceof Animal);  // true
        System.out.println(a instanceof Cat);     // false
        
        // null check
        Animal b = null;
        System.out.println(b instanceof Animal);  // false (null is not instance of anything)
        
        // Practical use - safe downcasting
        if (a instanceof Dog) {
            Dog d = (Dog) a;  // Safe cast
            System.out.println("It's a Dog!");
        }
        
        // Pattern matching (Java 16+)
        if (a instanceof Dog d) {
            d.bark();  // Directly use d without explicit cast
        }
    }
}

// instanceof with interfaces
interface Drawable { }
class Circle implements Drawable { }

Drawable shape = new Circle();
System.out.println(shape instanceof Circle);    // true
System.out.println(shape instanceof Drawable); // true`}
        />
        <div className="mt-3 p-2 bg-muted/30 rounded text-sm">
          <span className="font-bold text-primary">Quick Note:</span> Use instanceof before downcasting to avoid ClassCastException.
          <span className="block text-xs text-muted-foreground mt-1">⚠️ Overuse may indicate poor design - prefer polymorphism when possible.</span>
        </div>
      </QuickRefCard>

      {/* 2. Lambda Expressions */}
      <QuickRefCard title="Lambda Expressions (Basic Idea)" icon={<Zap className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Purpose:</strong> Lambda expressions provide a concise way to implement functional interfaces
          (interfaces with a single abstract method). They enable functional programming in Java.
        </p>
        <CodeBlock
          language="java"
          code={`// Lambda expressions - concise anonymous functions

// Before Lambda (Anonymous class)
Runnable oldWay = new Runnable() {
    @Override
    public void run() {
        System.out.println("Hello");
    }
};

// After Lambda (Java 8+)
Runnable newWay = () -> System.out.println("Hello");

// Syntax: (parameters) -> expression or { statements }

// Examples with different parameter types
interface Calculator {
    int calculate(int a, int b);
}

// No parameters
interface Greeting {
    void sayHello();
}
Greeting greet = () -> System.out.println("Hello!");

// One parameter
interface Square {
    int calculate(int x);
}
Square sq = x -> x * x;

// Multiple parameters
Calculator add = (a, b) -> a + b;
Calculator multiply = (a, b) -> a * b;

// Multiple statements - use braces
Calculator complex = (a, b) -> {
    int result = a * a + b * b;
    return result;
};

// Using with collections
List<String> names = Arrays.asList("John", "Jane", "Bob");
names.forEach(name -> System.out.println(name));
names.sort((a, b) -> a.compareTo(b));

// Method references (shorthand)
names.forEach(System.out::println);  // Same as above

// Stream API with lambdas
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
int sum = numbers.stream()
    .filter(n -> n % 2 == 0)      // Keep even numbers
    .map(n -> n * n)               // Square them
    .reduce(0, (a, b) -> a + b);   // Sum them
// Result: 4 + 16 = 20`}
        />
        <div className="mt-3 p-2 bg-muted/30 rounded text-sm">
          <span className="font-bold text-primary">Key Points:</span>
          <ul className="list-disc list-inside text-xs text-muted-foreground mt-1">
            <li>Works only with functional interfaces (@FunctionalInterface)</li>
            <li>Can access effectively final local variables</li>
            <li>Enables parallel processing with Stream API</li>
            <li>More readable than anonymous classes</li>
          </ul>
        </div>
      </QuickRefCard>

      {/* 3. Templates vs Generics */}
      <QuickRefCard title="Templates (C++) vs Generics (Java)" icon={<GitBranch className="h-4 w-4 text-primary" />}>
        <SimpleTable
          headers={["Feature", "C++ Templates", "Java Generics"]}
          rows={[
            ["Implementation", "Compile-time (code generation)", "Type erasure (runtime)"],
            ["Type checking", "At compile time (early)", "At compile time"],
            ["Specialization", "✓ Supported", "✗ Not supported"],
            ["Primitive types", "✓ Works with int, char, etc.", "✗ Uses wrapper classes (Integer, Character)"],
            ["Instantiation", "Creates separate code per type", "Single bytecode version"],
            ["Performance", "Faster (no type erasure overhead)", "Slight overhead from boxing"],
            ["Debugging", "Harder (error messages complex)", "Easier"],
            ["Code size", "Larger (bloat)", "Smaller"],
          ]}
        />
        <CodeBlock
          language="java"
          code={`// C++ Templates (compile-time code generation)
/*
template<typename T>
class Box {
    T value;
public:
    Box(T v) : value(v) {}
    T getValue() { return value; }
};

Box<int> intBox(10);      // Creates code for int
Box<double> doubleBox(3.14); // Creates code for double
Box<string> stringBox("Hi"); // Creates code for string
*/

// Java Generics (type erasure - single bytecode)
class Box<T> {
    private T value;
    
    public Box(T value) {
        this.value = value;
    }
    
    public T getValue() {
        return value;
    }
}

Box<Integer> intBox = new Box<>(10);
Box<Double> doubleBox = new Box<>(3.14);
Box<String> stringBox = new Box<>("Hi");

// Bounded type parameters
class NumberBox<T extends Number> {
    private T value;
    public NumberBox(T value) { this.value = value; }
    public double doubleValue() { return value.doubleValue(); }
}

// Wildcards
void printList(List<?> list) {  // Unknown type
    for (Object obj : list) {
        System.out.println(obj);
    }
}

void processNumbers(List<? extends Number> numbers) {  // Upper bound
    // Can read as Number, cannot write
}

void addNumbers(List<? super Integer> list) {  // Lower bound
    list.add(10);  // Can add Integer
}

// Type erasure example (what happens at runtime)
// List<String> and List<Integer> become List at runtime
// Compiler adds casts where needed`}
        />
      </QuickRefCard>

      {/* 4. Use Case Diagram */}
      <QuickRefCard title="Use Case Diagram (Basics)" icon={<Users className="h-4 w-4 text-primary" />}>
        <div className="space-y-3">
          <p className="text-muted-foreground">
            <strong>Purpose:</strong> Shows system functionality from user perspective. Displays actors, use cases, and relationships.
          </p>
          <div className="bg-muted/30 p-3 rounded font-mono text-xs">
            <pre className="whitespace-pre-wrap">
{`┌─────────────────────────────────────────────────────────┐
│                   ATM System                            │
│                                                         │
│  ┌─────────┐      ┌──────────────┐                      │
│  │         │      │   Withdraw   │                      │
│  │ Customer│─────▶│   Cash       │                      │
│  │         │      └──────────────┘                      │
│  └─────────┘             │                              │
│       │                  │ <<include>>                  │
│       │           ┌──────▼──────┐                       │
│       │           │  Validate   │                       │
│       │           │  PIN        │                       │
│       │           └─────────────┘                       │
│       │                                                 │
│       │      ┌──────────────┐      ┌──────────────┐    │
│       └─────▶│  Check       │      │  Deposit     │    │
│              │  Balance     │      │  Funds       │    │
│              └──────────────┘      └──────────────┘    │
│                                                         │
│  ┌─────────┐      ┌──────────────┐                      │
│  │  Admin  │─────▶│   Refill     │                      │
│  │         │      │   Cash       │                      │
│  └─────────┘      └──────────────┘                      │
└─────────────────────────────────────────────────────────┘

Components:
• Actor: Stick figure (Customer, Admin) - external entity
• Use Case: Oval (Withdraw Cash, Check Balance) - system function
• System Boundary: Rectangle - system scope
• Relationships:
  - Association: Line between actor and use case
  - <<include>>: One use case always calls another
  - <<extend>>: Optional behavior (dashed arrow)
  - Generalization: Arrow from child to parent actor`}
            </pre>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div><span className="font-bold text-primary">Actor:</span> User/System interacting with system</div>
            <div><span className="font-bold text-primary">Use Case:</span> Specific functionality</div>
            <div><span className="font-bold text-primary">Include:</span> Always executes</div>
            <div><span className="font-bold text-primary">Extend:</span> Optional extension</div>
          </div>
        </div>
      </QuickRefCard>

      {/* 5. Sequence Diagram (Login Process) */}
      <QuickRefCard title="Sequence Diagram (Login Process)" icon={<MessageSquare className="h-4 w-4 text-primary" />}>
        <div className="bg-muted/30 p-3 rounded font-mono text-xs overflow-x-auto">
          <pre className="whitespace-pre">
{`User    LoginView    LoginController    UserService    Database
  │          │              │                │            │
  │─Login────▶│              │                │            │
  │(cred)     │              │                │            │
  │           │─validate────▶│                │            │
  │           │ (cred)       │                │            │
  │           │              │─authenticate──▶│            │
  │           │              │ (user,pass)    │            │
  │           │              │                │─queryUser─▶│
  │           │              │                │            │
  │           │              │                │─userData──▶│
  │           │              │                │            │
  │           │              │─result────────▶│            │
  │           │              │ (success/fail) │            │
  │           │─response────▶│                │            │
  │           │ (success)    │                │            │
  │─dashboard◀│              │                │            │
  │           │              │                │            │

Alt - Alternative Flow (Invalid Credentials):
  │           │              │                │            │
  │           │              │─result────────▶│            │
  │           │              │ (invalid)      │            │
  │           │─error───────▶│                │            │
  │           │ (invalid)    │                │            │
  │─showError◀│              │                │            │

Legend:
▶─ Solid arrow = Message
◀─ Dashed arrow = Return
│ Vertical line = Lifeline
┌┐ Rectangle = Activation bar
[ ] = Condition/Alternative`}
          </pre>
        </div>
        <div className="mt-3 text-sm text-muted-foreground">
          <span className="font-bold text-primary">Key Elements:</span> Lifeline (vertical dashed line), Activation bar (rectangle), Message (arrow), Return (dashed arrow), Alt (alternative flow)
        </div>
      </QuickRefCard>

      {/* 6. Generalization vs Specialization */}
      <QuickRefCard title="Generalization vs Specialization" icon={<GitBranch className="h-4 w-4 text-primary" />}>
        <SimpleTable
          headers={["Feature", "Generalization", "Specialization"]}
          rows={[
            ["Direction", "Bottom-up", "Top-down"],
            ["Process", "Extract common features", "Add specific features"],
            ["Abstraction Level", "Increases", "Decreases"],
            ["Detail", "Less detail", "More detail"],
            ["Example", "Dog, Cat → Animal", "Animal → Dog, Cat"],
            ["UML Symbol", "Hollow triangle up", "Hollow triangle up"],
          ]}
        />
        <div className="mt-3 bg-muted/30 p-3 rounded font-mono text-xs">
          <pre>
{`                    ┌─────────────────┐
                    │     Animal      │  ← Superclass
                    │  (General)      │    (Generalization)
                    └────────┬────────┘
                             ▲
                             │
            ┌────────────────┼────────────────┐
            │                │                │
      ┌─────▼─────┐    ┌──────▼──────┐   ┌─────▼─────┐
      │   Dog     │    │    Cat      │   │   Bird    │
      │ (Special) │    │  (Special)  │   │ (Special) │
      └───────────┘    └─────────────┘   └───────────┘
                    (Specialization)

Example Code:
// Generalization (extracting common features)
class Animal {           // Created by generalizing Dog, Cat, Bird
    void breathe() { }
    void eat() { }
}

// Specialization (adding specific features)
class Dog extends Animal {
    void bark() { }      // Specific to Dog
}

class Bird extends Animal {
    void fly() { }       // Specific to Bird
}`}
          </pre>
        </div>
      </QuickRefCard>

      {/* 7. Factory Pattern */}
      <QuickRefCard title="Factory Pattern" icon={<Box className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Purpose:</strong> Creates objects without exposing instantiation logic to client.
          Promotes loose coupling and makes code easier to extend.
        </p>
        <CodeBlock
          language="java"
          code={`// Factory Pattern Example
interface Vehicle {
    void drive();
}

class Car implements Vehicle {
    public void drive() { System.out.println("Driving car"); }
}

class Bike implements Vehicle {
    public void drive() { System.out.println("Riding bike"); }
}

class Truck implements Vehicle {
    public void drive() { System.out.println("Driving truck"); }
}

// Simple Factory
class VehicleFactory {
    public static Vehicle createVehicle(String type) {
        switch (type.toLowerCase()) {
            case "car": return new Car();
            case "bike": return new Bike();
            case "truck": return new Truck();
            default: throw new IllegalArgumentException("Unknown type");
        }
    }
}

// Abstract Factory
interface GUIFactory {
    Button createButton();
    TextBox createTextBox();
}

class WindowsFactory implements GUIFactory {
    public Button createButton() { return new WindowsButton(); }
    public TextBox createTextBox() { return new WindowsTextBox(); }
}

class MacFactory implements GUIFactory {
    public Button createButton() { return new MacButton(); }
    public TextBox createTextBox() { return new MacTextBox(); }
}

// Usage
public class FactoryDemo {
    public static void main(String[] args) {
        Vehicle car = VehicleFactory.createVehicle("car");
        car.drive();  // Driving car
        
        Vehicle bike = VehicleFactory.createVehicle("bike");
        bike.drive();  // Riding bike
        
        // Abstract factory
        GUIFactory factory = new WindowsFactory();
        Button btn = factory.createButton();
        btn.render();
    }
}

// Benefits:
// 1. Single responsibility - creation logic in one place
// 2. Open/Closed principle - add new types without modifying client
// 3. Loose coupling - client depends on interface, not concrete class`}
        />
      </QuickRefCard>

      {/* 8. Observer Pattern */}
      <QuickRefCard title="Observer Pattern" icon={<Bell className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Purpose:</strong> Defines one-to-many dependency where when subject changes state,
          all observers are notified automatically. Used in event handling, pub-sub systems.
        </p>
        <CodeBlock
          language="java"
          code={`// Observer Pattern Example
import java.util.*;

// Observer interface
interface Observer {
    void update(String message);
}

// Subject interface
interface Subject {
    void attach(Observer o);
    void detach(Observer o);
    void notifyObservers();
}

// Concrete Subject
class NewsAgency implements Subject {
    private List<Observer> observers = new ArrayList<>();
    private String news;
    
    public void attach(Observer o) { observers.add(o); }
    public void detach(Observer o) { observers.remove(o); }
    
    public void notifyObservers() {
        for (Observer o : observers) {
            o.update(news);
        }
    }
    
    public void publishNews(String news) {
        this.news = news;
        notifyObservers();
    }
}

// Concrete Observers
class EmailSubscriber implements Observer {
    private String email;
    
    public EmailSubscriber(String email) { this.email = email; }
    
    public void update(String message) {
        System.out.println("Email to " + email + ": " + message);
    }
}

class SMSSubscriber implements Observer {
    private String phone;
    
    public SMSSubscriber(String phone) { this.phone = phone; }
    
    public void update(String message) {
        System.out.println("SMS to " + phone + ": " + message);
    }
}

// Usage
public class ObserverDemo {
    public static void main(String[] args) {
        NewsAgency agency = new NewsAgency();
        
        Observer emailSub = new EmailSubscriber("user@example.com");
        Observer smsSub = new SMSSubscriber("+1234567890");
        
        agency.attach(emailSub);
        agency.attach(smsSub);
        
        agency.publishNews("Breaking News!");
        // Email to user@example.com: Breaking News!
        // SMS to +1234567890: Breaking News!
        
        agency.detach(smsSub);
        agency.publishNews("Another update");
        // Only email receives this
    }
}

// Java built-in Observer (deprecated in Java 9)
// Alternative: PropertyChangeListener, EventListener`}
        />
        <div className="mt-3 p-2 bg-muted/30 rounded text-sm">
          <span className="font-bold text-primary">Real-world uses:</span>
          <ul className="list-disc list-inside text-xs text-muted-foreground mt-1">
            <li>GUI event listeners (button clicks)</li>
            <li>Model-View-Controller (MVC) architecture</li>
            <li>Publish-subscribe systems</li>
            <li>Stock market price updates</li>
            <li>Social media notifications</li>
          </ul>
        </div>
      </QuickRefCard>

     {/* 9. Coupling vs Cohesion */}
<QuickRefCard
  title="Coupling vs Cohesion"
  icon={<Link className="h-4 w-4 text-primary" />}
>
  <SimpleTable
    headers={["Aspect", "Coupling", "Cohesion"]}
    rows={[
      ["Definition", "Inter-module dependency", "Intra-module focus"],
      ["Desired Level", "Low (loose coupling)", "High (strong cohesion)"],
      ["Good/Bad", "Good: Low coupling", "Good: High cohesion"],
      ["Impact", "Changes propagate", "Changes localized"],
      ["Testing", "Harder with high coupling", "Easier with high cohesion"],
      ["Reusability", "Reduced with high coupling", "Increased with high cohesion"],
    ]}
  />

  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
    <div className="p-2 bg-destructive/10 rounded">
      <span className="font-bold text-destructive">❌ High Coupling (Bad)</span>
      <pre className="text-xs mt-1">
{`class A {
    B b = new B();  // Direct dependency
    b.doSomething();
}`}
      </pre>
    </div>

    <div className="p-2 bg-success/10 rounded">
      <span className="font-bold text-success">✅ Low Coupling (Good)</span>
      <pre className="text-xs mt-1">
{`class A {
    InterfaceB b;  // Depends on abstraction
    // Dependency injection
}`}
      </pre>
    </div>

    <div className="p-2 bg-destructive/10 rounded">
      <span className="font-bold text-destructive">❌ Low Cohesion (Bad)</span>
      <pre className="text-xs mt-1">
{`class Utils {
    sendEmail()
    calculateTax()
    compressFile()
}`}
      </pre>
    </div>

    <div className="p-2 bg-success/10 rounded">
      <span className="font-bold text-success">✅ High Cohesion (Good)</span>
      <pre className="text-xs mt-1">
{`class EmailService {
    sendEmail()
    validateEmail()
    formatEmail()
}`}
      </pre>
    </div>
  </div>
</QuickRefCard>
      {/* 10. SOLID Principles */}
      <QuickRefCard title="SOLID Principles (Basic Idea)" icon={<Shield className="h-4 w-4 text-primary" />}>
        <div className="space-y-2 text-sm">
          <div className="p-2 border-l-4 border-primary bg-primary/5">
            <span className="font-bold text-primary">S</span> - <strong>Single Responsibility</strong>
            <p className="text-xs text-muted-foreground mt-1">A class should have only one reason to change. One job per class.</p>
          </div>
          <div className="p-2 border-l-4 border-primary bg-primary/5">
            <span className="font-bold text-primary">O</span> - <strong>Open/Closed</strong>
            <p className="text-xs text-muted-foreground mt-1">Open for extension, closed for modification. Add features without changing existing code.</p>
          </div>
          <div className="p-2 border-l-4 border-primary bg-primary/5">
            <span className="font-bold text-primary">L</span> - <strong>Liskov Substitution</strong>
            <p className="text-xs text-muted-foreground mt-1">Derived classes must be substitutable for base classes. Child can replace parent without breaking.</p>
          </div>
          <div className="p-2 border-l-4 border-primary bg-primary/5">
            <span className="font-bold text-primary">I</span> - <strong>Interface Segregation</strong>
            <p className="text-xs text-muted-foreground mt-1">Don't force clients to depend on interfaces they don't use. Split fat interfaces.</p>
          </div>
          <div className="p-2 border-l-4 border-primary bg-primary/5">
            <span className="font-bold text-primary">D</span> - <strong>Dependency Inversion</strong>
            <p className="text-xs text-muted-foreground mt-1">Depend on abstractions, not concretions. High-level modules shouldn't depend on low-level modules.</p>
          </div>
        </div>
        <div className="mt-3 p-2 bg-primary/10 rounded text-xs text-center">
          <span className="font-bold">Mnemonic:</span> "Single Open Liskov Interface Dependency" or "SOLID"
        </div>
      </QuickRefCard>

      {/* 11. Message Passing */}
      <QuickRefCard title="Message Passing in OOP" icon={<MessageSquare className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Definition:</strong> Mechanism where objects communicate by invoking methods on each other.
          The sender sends a message, the receiver executes the method.
        </p>
        <CodeBlock
          language="java"
          code={`// Message Passing Example
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}

class Client {
    public void compute() {
        Calculator calc = new Calculator();
        // Message passing: client sends "add" message to calculator
        int result = calc.add(5, 3);  // ← This is message passing
        System.out.println(result);
    }
}

// Real-world example
class Order {
    private double amount;
    
    public void process(PaymentGateway gateway) {
        // Message passing: Order sends "charge" message to PaymentGateway
        boolean success = gateway.charge(amount);
        
        if (success) {
            // Message passing: Order sends "sendConfirmation" message to EmailService
            emailService.sendConfirmation("order@example.com");
        }
    }
}

// Characteristics:
// 1. Objects only communicate through method calls
// 2. Sender doesn't need to know implementation details
// 3. Supports encapsulation
// 4. Enables polymorphism (same message, different behavior)`}
        />
        <div className="mt-3 p-2 bg-muted/30 rounded text-sm">
          <span className="font-bold text-primary">Analogy:</span> Like ordering coffee - you send a "message" (order) to the barista,
          who knows how to execute it. You don't need to know how the coffee machine works.
        </div>
      </QuickRefCard>

      {/* 12. RTTI */}
      <QuickRefCard title="RTTI (Run-Time Type Identification)" icon={<Eye className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Definition:</strong> Mechanism to determine the actual type of an object at runtime.
          Allows safe downcasting and type checking.
        </p>
        <CodeBlock
          language="java"
          code={`// Java RTTI
class Animal { }
class Dog extends Animal { 
    void bark() { System.out.println("Woof!"); }
}

public class RTTIDemo {
    public static void main(String[] args) {
        Animal a = new Dog();
        
        // 1. instanceof operator
        if (a instanceof Dog) {
            Dog d = (Dog) a;  // Safe downcast
            d.bark();
        }
        
        // 2. getClass() method
        System.out.println(a.getClass().getName());     // Dog
        System.out.println(a.getClass().getSimpleName()); // Dog
        
        // 3. Class comparison
        if (a.getClass() == Dog.class) {
            System.out.println("Exactly a Dog object");
        }
        
        // 4. isInstance() method
        boolean isDog = Dog.class.isInstance(a);  // true
    }
}

// C++ RTTI (conceptual)
/*
#include <typeinfo>
Animal* a = new Dog();
if (typeid(*a) == typeid(Dog)) {
    Dog* d = dynamic_cast<Dog*>(a);
    d->bark();
}
*/

// When to use RTTI:
// ✅ Serialization/deserialization
// ✅ Debugging/logging
// ✅ Framework code (Spring, Hibernate)
// ❌ Business logic (use polymorphism instead)`}
        />
        <div className="mt-3 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-sm">
          <span className="font-bold text-yellow-500">⚠️ Caution:</span> Overusing RTTI indicates poor design.
          Consider using polymorphism (method overriding) instead.
        </div>
      </QuickRefCard>

      {/* 13. Reflection (Java) */}
      <QuickRefCard title="Reflection in Java" icon={<Coffee className="h-4 w-4 text-primary" />}>
        <p className="text-muted-foreground mb-3">
          <strong>Definition:</strong> Feature that allows inspecting and modifying classes, methods, fields at runtime,
          even private ones. Powerful but can break encapsulation.
        </p>
        <CodeBlock
          language="java"
          code={`import java.lang.reflect.*;

class Person {
    private String name = "Secret";
    private int age = 30;
    
    private void secretMethod() {
        System.out.println("This is private!");
    }
}

public class ReflectionDemo {
    public static void main(String[] args) throws Exception {
        Person p = new Person();
        
        // 1. Get class information
        Class<?> clazz = p.getClass();
        System.out.println("Class: " + clazz.getName());
        
        // 2. Access private field
        Field nameField = Person.class.getDeclaredField("name");
        nameField.setAccessible(true);  // Break encapsulation!
        String name = (String) nameField.get(p);
        System.out.println("Name: " + name);
        
        // 3. Modify private field
        nameField.set(p, "Hacked!");
        
        // 4. Invoke private method
        Method secretMethod = Person.class.getDeclaredMethod("secretMethod");
        secretMethod.setAccessible(true);
        secretMethod.invoke(p);
        
        // 5. Get all methods
        Method[] methods = Person.class.getDeclaredMethods();
        for (Method m : methods) {
            System.out.println(m.getName() + " - private: " + 
                Modifier.isPrivate(m.getModifiers()));
        }
        
        // 6. Create instance dynamically
        Class<?> stringClass = Class.forName("java.lang.String");
        Object str = stringClass.getConstructor(String.class)
            .newInstance("Hello Reflection");
    }
}

// Legitimate uses:
// - Frameworks (Spring, Hibernate)
// - Debugging tools
// - Serialization libraries
// - Testing frameworks (JUnit)
// - Code analysis tools`}
        />
        <div className="mt-3 p-2 bg-destructive/10 border border-destructive/20 rounded text-sm">
          <span className="font-bold text-destructive">⚠️ Security Warning:</span> Reflection can break encapsulation.
          Use only when necessary (frameworks, debugging). Never use in business logic without good reason.
        </div>
      </QuickRefCard>

      {/* Quick Comparison Table */}
      <div className="mt-8 p-4 bg-primary/5 rounded-lg border border-primary/20">
        <div className="flex items-center gap-2 mb-3">
          <Target className="h-5 w-5 text-primary" />
          <h3 className="font-bold text-foreground">Quick Comparison Chart</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-muted">
                <th className="border border-border p-2 text-left">Concept</th>
                <th className="border border-border p-2 text-left">Purpose</th>
                <th className="border border-border p-2 text-left">Key Keyword/Operator</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-border p-2">instanceof</td>
                <td className="border border-border p-2">Type checking at runtime</td>
                <td className="border border-border p-2"><code>instanceof</code></td>
              </tr>
              <tr>
                <td className="border border-border p-2">Lambda</td>
                <td className="border border-border p-2">Functional programming</td>
                <td className="border border-border p-2"><code>-&gt;</code></td>
              </tr>
              <tr>
                <td className="border border-border p-2">Generics</td>
                <td className="border border-border p-2">Type-safe collections</td>
                <td className="border border-border p-2"><code>&lt;T&gt;</code></td>
              </tr>
              <tr>
                <td className="border border-border p-2">RTTI</td>
                <td className="border border-border p-2">Runtime type identification</td>
                <td className="border border-border p-2"><code>instanceof, getClass()</code></td>
              </tr>
              <tr>
                <td className="border border-border p-2">Reflection</td>
                <td className="border border-border p-2">Runtime class inspection</td>
                <td className="border border-border p-2"><code>Class, Field, Method</code></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Interview Tips */}
      <div className="mt-6 p-4 bg-muted/30 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <AlertCircle className="h-5 w-5 text-primary" />
          <h3 className="font-bold text-foreground">Interview Quick Tips</h3>
        </div>
        <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
          <li><strong>instanceof</strong> - Always check before downcasting to avoid ClassCastException</li>
          <li><strong>Lambda</strong> - Only works with functional interfaces (@FunctionalInterface)</li>
          <li><strong>Generics vs Templates</strong> - Java uses type erasure, C++ generates separate code</li>
          <li><strong>SOLID</strong> - Remember "S.O.L.I.D" or use a mnemonic</li>
          <li><strong>Coupling vs Cohesion</strong> - Low coupling + High cohesion = Good design</li>
          <li><strong>Reflection</strong> - Powerful but breaks encapsulation - use sparingly</li>
        </ul>
      </div>
    </TopicContent>
  );
}