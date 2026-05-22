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
  Shield,
  GitBranch,
  Layers,
  Box,
  Link,
  Zap,
  Eye,
} from "lucide-react";

const solidExamplesCode = `// SOLID Principles Examples

// 1. Single Responsibility Principle (SRP)
// Before - Bad: One class handles multiple responsibilities
class UserBad {
    private String name;
    private String email;
    
    // User data management
    public void saveToDatabase() { /* DB code */ }
    public void sendEmail() { /* Email code */ }
    public void generateReport() { /* Report code */ }
}

// After - Good: Separate classes for separate responsibilities
class User {
    private String name;
    private String email;
    // Getters/setters only
}

class UserRepository {
    public void save(User user) { /* DB code */ }
}

class EmailService {
    public void sendEmail(User user, String message) { /* Email code */ }
}

class ReportGenerator {
    public void generateReport(User user) { /* Report code */ }
}


// 2. Open/Closed Principle (OCP)
// Before - Bad: Modifying existing code for new features
class DiscountCalculatorBad {
    public double calculate(String type, double price) {
        if (type.equals("Regular")) return price;
        else if (type.equals("Premium")) return price * 0.8;
        else if (type.equals("VIP")) return price * 0.7;
        // Need to modify this method for new discount types
        return price;
    }
}

// After - Good: Open for extension, closed for modification
interface DiscountStrategy {
    double calculate(double price);
}

class RegularDiscount implements DiscountStrategy {
    public double calculate(double price) { return price; }
}

class PremiumDiscount implements DiscountStrategy {
    public double calculate(double price) { return price * 0.8; }
}

class VIPDiscount implements DiscountStrategy {
    public double calculate(double price) { return price * 0.7; }
}

class DiscountCalculator {
    public double calculate(DiscountStrategy strategy, double price) {
        return strategy.calculate(price);
    }
}


// 3. Liskov Substitution Principle (LSP)
// Before - Bad: Derived class changes parent behavior
class Rectangle {
    protected int width;
    protected int height;
    
    public void setWidth(int w) { width = w; }
    public void setHeight(int h) { height = h; }
    public int getArea() { return width * height; }
}

class Square extends Rectangle {
    @Override
    public void setWidth(int w) { 
        super.setWidth(w);
        super.setHeight(w);  // Violates LSP - changes expected behavior
    }
    
    @Override
    public void setHeight(int h) {
        super.setWidth(h);
        super.setHeight(h);
    }
}

// After - Good: Proper abstraction
interface Shape {
    int getArea();
}

class RectangleGood implements Shape {
    private int width;
    private int height;
    
    public RectangleGood(int w, int h) { width = w; height = h; }
    public int getArea() { return width * height; }
}

class SquareGood implements Shape {
    private int side;
    
    public SquareGood(int s) { side = s; }
    public int getArea() { return side * side; }
}


// 4. Interface Segregation Principle (ISP)
// Before - Bad: Fat interface
interface Worker {
    void work();
    void eat();
    void sleep();
}

class Robot implements Worker {
    public void work() { /* works */ }
    public void eat() { /* Robot doesn't eat! */ }
    public void sleep() { /* Robot doesn't sleep! */ }
}

// After - Good: Segregated interfaces
interface Workable {
    void work();
}

interface Eatable {
    void eat();
}

interface Sleepable {
    void sleep();
}

class Human implements Workable, Eatable, Sleepable {
    public void work() { /* work */ }
    public void eat() { /* eat */ }
    public void sleep() { /* sleep */ }
}

class RobotGood implements Workable {
    public void work() { /* work only */ }
}


// 5. Dependency Inversion Principle (DIP)
// Before - Bad: High-level depends on low-level
class EmailSender {
    public void send(String message) { /* send email */ }
}

class NotificationServiceBad {
    private EmailSender emailSender = new EmailSender(); // Direct dependency
    
    public void notify(String message) {
        emailSender.send(message);
    }
}

// After - Good: Depend on abstractions
interface MessageSender {
    void send(String message);
}

class EmailSenderGood implements MessageSender {
    public void send(String message) { /* send email */ }
}

class SMSSender implements MessageSender {
    public void send(String message) { /* send SMS */ }
}

class NotificationService {
    private MessageSender sender;
    
    public NotificationService(MessageSender sender) {  // Dependency injection
        this.sender = sender;
    }
    
    public void notify(String message) {
        sender.send(message);
    }
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What does SOLID stand for?",
    options: [
      "Single, Object, Linked, Interface, Data",
      "Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion",
      "Simple, Observable, Linear, Integrated, Dynamic",
      "Structure, Object, Logic, Input, Data",
    ],
    correctAnswer: 1,
    explanation: "SOLID is an acronym for five design principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.",
  },
  {
    id: 2,
    question: "What is the difference between coupling and cohesion?",
    options: [
      "They are the same thing",
      "Coupling is inter-module dependency, cohesion is intra-module focus",
      "Coupling is about classes, cohesion is about methods",
      "Coupling is good, cohesion is bad",
    ],
    correctAnswer: 1,
    explanation: "Coupling measures how dependent modules are on each other (low coupling is good). Cohesion measures how focused a module's responsibilities are (high cohesion is good).",
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

export default function SOLIDPrinciplesPage() {
  const result = getSubtopicBySlug("oop", "solid-principles");

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
      {/* Introduction */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <BookOpen className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            OOP Design Principles (SOLID)
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">SOLID principles</strong> are five design principles 
            introduced by Robert C. Martin that help create maintainable, scalable, and robust 
            object-oriented software.
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
            SOLID Examples
          </h2>
        </div>
        <CodeBlock code={solidExamplesCode} language="java" />
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

        {/* Question 46: SOLID Principles Overview */}
        <QuestionCard
          number={46}
          title="SOLID Principles"
          question="What are SOLID principles in OOP? Explain each briefly."
          answer="SOLID is an acronym for five design principles: S - Single Responsibility: A class should have only one reason to change. O - Open/Closed: Classes open for extension, closed for modification. L - Liskov Substitution: Derived classes must be substitutable for base classes. I - Interface Segregation: Clients shouldn't depend on interfaces they don't use. D - Dependency Inversion: Depend on abstractions, not concretions."
          marks={5}
          icon={<Shield className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">SOLID Quick Reference:</h4>
            <div className="grid grid-cols-1 gap-2 text-sm">
              <div><span className="font-bold text-primary">S</span> - Single Responsibility (One job per class)</div>
              <div><span className="font-bold text-primary">O</span> - Open/Closed (Extend, don't modify)</div>
              <div><span className="font-bold text-primary">L</span> - Liskov Substitution (Subtypes replace parents)</div>
              <div><span className="font-bold text-primary">I</span> - Interface Segregation (Split fat interfaces)</div>
              <div><span className="font-bold text-primary">D</span> - Dependency Inversion (Abstractions over details)</div>
            </div>
          </div>
          <CodeBlock
            language="java"
            code={`// Quick Example of all SOLID principles

// S: Single Responsibility
class Invoice { /* Invoice data only */ }
class InvoiceRepository { /* Save to DB only */ }
class InvoicePrinter { /* Print only */ }

// O: Open/Closed
interface PaymentMethod { void pay(double amount); }
class CreditCard implements PaymentMethod { public void pay(double amt) {} }
class PayPal implements PaymentMethod { public void pay(double amt) {} }

// L: Liskov Substitution
interface Bird { void move(); }
class Sparrow implements Bird { public void move() { fly(); } }
class Penguin implements Bird { public void move() { walk(); } }

// I: Interface Segregation
interface Flyable { void fly(); }
interface Swimmable { void swim(); }
class Duck implements Flyable, Swimmable { 
    public void fly() {} 
    public void swim() {} 
}

// D: Dependency Inversion
interface Database { void save(); }
class MySQL implements Database { public void save() {} }
class App { 
    private Database db;
    public App(Database db) { this.db = db; }
}`}
          />
        </QuestionCard>

        {/* Question 47: Single Responsibility Principle */}
        <QuestionCard
          number={47}
          title="Single Responsibility Principle (SRP)"
          question="What is the Single Responsibility Principle?"
          answer="The Single Responsibility Principle states that a class should have only one reason to change, meaning it should have only one job or responsibility. This makes classes easier to understand, maintain, and test. When a class has multiple responsibilities, changes to one responsibility may affect others, increasing bug risk and reducing reusability."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// ❌ BAD: Multiple responsibilities
class Employee {
    private String name;
    private double salary;
    
    // Responsibility 1: Employee data management
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    // Responsibility 2: Salary calculation
    public double calculateSalary() { return salary * 1.1; }
    
    // Responsibility 3: Database operations
    public void saveToDatabase() { /* DB code */ }
    
    // Responsibility 4: Report generation
    public void generateReport() { /* Report code */ }
    
    // Responsibility 5: Email sending
    public void sendEmail() { /* Email code */ }
}

// ✅ GOOD: Separate classes for separate responsibilities
class Employee {
    private String name;
    private double salary;
    // Getters and setters only
}

class SalaryCalculator {
    public double calculate(Employee emp) {
        return emp.getSalary() * 1.1;
    }
}

class EmployeeRepository {
    public void save(Employee emp) { /* DB code */ }
}

class EmployeeReportGenerator {
    public void generateReport(Employee emp) { /* Report code */ }
}

class EmailService {
    public void sendEmail(Employee emp, String message) { /* Email code */ }
}

// Benefits:
// - Each class has one reason to change
// - Easier to test individually
// - Changes to one feature don't affect others
// - Classes are more reusable`}
          />
        </QuestionCard>

        {/* Question 48: Open/Closed Principle */}
        <QuestionCard
          number={48}
          title="Open/Closed Principle (OCP)"
          question="What is Open/Closed Principle?"
          answer="The Open/Closed Principle states that software entities (classes, modules, functions) should be open for extension but closed for modification. This means you should be able to add new functionality without changing existing code, typically achieved through inheritance, interfaces, and polymorphism."
          marks={5}
          icon={<Layers className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// ❌ BAD: Must modify existing code for new shapes
class AreaCalculatorBad {
    public double calculate(Object shape) {
        if (shape instanceof Rectangle) {
            Rectangle r = (Rectangle) shape;
            return r.width * r.height;
        } else if (shape instanceof Circle) {
            Circle c = (Circle) shape;
            return Math.PI * c.radius * c.radius;
        }
        // Need to add more if-else for Triangle, Square, etc.
        return 0;
    }
}

// ✅ GOOD: Open for extension, closed for modification
interface Shape {
    double area();
}

class Rectangle implements Shape {
    double width, height;
    public Rectangle(double w, double h) { width = w; height = h; }
    public double area() { return width * height; }
}

class Circle implements Shape {
    double radius;
    public Circle(double r) { radius = r; }
    public double area() { return Math.PI * radius * radius; }
}

class Triangle implements Shape {
    double base, height;
    public Triangle(double b, double h) { base = b; height = h; }
    public double area() { return 0.5 * base * height; }
}

// New shapes can be added without modifying this class!
class AreaCalculator {
    public double calculate(Shape shape) {
        return shape.area();  // Closed for modification, open for extension
    }
}

// Usage
public class OCPDemo {
    public static void main(String[] args) {
        AreaCalculator calculator = new AreaCalculator();
        
        List<Shape> shapes = Arrays.asList(
            new Rectangle(5, 10),
            new Circle(7),
            new Triangle(4, 6)
        );
        
        for (Shape shape : shapes) {
            System.out.println("Area: " + calculator.calculate(shape));
        }
    }
}`}
          />
        </QuestionCard>

        {/* Question 49: Liskov Substitution Principle */}
        <QuestionCard
          number={49}
          title="Liskov Substitution Principle (LSP)"
          question="What is Liskov Substitution Principle?"
          answer="The Liskov Substitution Principle states that objects of a derived class should be substitutable for objects of the base class without affecting program correctness. If class B extends class A, then we should be able to replace A with B without breaking the program. This means derived classes must honor the contract established by the base class."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">LSP Violation Signs:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>Derived class throws exceptions not thrown by base</li>
              <li>Derived class changes base class behavior unexpectedly</li>
              <li>Derived class has methods that do nothing (empty implementations)</li>
              <li>Type checking with instanceof before using derived class</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`// ❌ BAD: Violates LSP
class Bird {
    public void fly() {
        System.out.println("Flying...");
    }
}

class Penguin extends Bird {
    @Override
    public void fly() {
        // Penguin can't fly! Violates LSP
        throw new UnsupportedOperationException("Penguins can't fly!");
    }
}

// Usage that breaks
public void makeBirdFly(Bird bird) {
    bird.fly();  // Works for Sparrow but crashes for Penguin!
}

// ✅ GOOD: Proper abstraction
interface Bird {
    void move();
}

interface Flyable extends Bird {
    void fly();
}

interface Walkable extends Bird {
    void walk();
}

class Sparrow implements Flyable, Walkable {
    public void move() { fly(); }
    public void fly() { System.out.println("Sparrow flying"); }
    public void walk() { System.out.println("Sparrow walking"); }
}

class Penguin implements Walkable {
    public void move() { walk(); }
    public void walk() { System.out.println("Penguin walking"); }
}

// Now each bird type can be substituted correctly
public class LSPDemo {
    public static void makeBirdMove(Bird bird) {
        bird.move();  // Works for any Bird
    }
    
    public static void makeBirdFly(Flyable bird) {
        bird.fly();  // Only for birds that can fly
    }
}`}
          />
        </QuestionCard>

        {/* Question 50: Interface Segregation Principle */}
        <QuestionCard
          number={50}
          title="Interface Segregation Principle (ISP)"
          question="What is Interface Segregation Principle?"
          answer="The Interface Segregation Principle states that clients should not be forced to depend on interfaces they do not use. Instead of having one large 'fat' interface, it's better to have multiple smaller, focused interfaces. This prevents classes from having to implement methods they don't need."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// ❌ BAD: Fat interface
interface Worker {
    void work();
    void eat();
    void sleep();
    void attendMeeting();
    void submitReport();
}

class Robot implements Worker {
    public void work() { /* works */ }
    // Robot doesn't eat, sleep, attend meetings, or submit reports!
    public void eat() { throw new UnsupportedOperationException(); }
    public void sleep() { throw new UnsupportedOperationException(); }
    public void attendMeeting() { throw new UnsupportedOperationException(); }
    public void submitReport() { throw new UnsupportedOperationException(); }
}

// ✅ GOOD: Segregated interfaces
interface Workable {
    void work();
}

interface Eatable {
    void eat();
}

interface Sleepable {
    void sleep();
}

interface MeetingAttendable {
    void attendMeeting();
}

interface Reportable {
    void submitReport();
}

// Human implements only what humans need
class Human implements Workable, Eatable, Sleepable, MeetingAttendable, Reportable {
    public void work() { System.out.println("Human working"); }
    public void eat() { System.out.println("Human eating"); }
    public void sleep() { System.out.println("Human sleeping"); }
    public void attendMeeting() { System.out.println("Human in meeting"); }
    public void submitReport() { System.out.println("Human submitting report"); }
}

// Robot implements only what robots need
class RobotGood implements Workable {
    public void work() { System.out.println("Robot working 24/7"); }
    // No empty implementations needed!
}

// Another example: Multi-function printer
interface Printer {
    void print(String document);
}

interface Scanner {
    void scan(String document);
}

interface Fax {
    void fax(String document);
}

// Simple printer only needs Printer interface
class SimplePrinter implements Printer {
    public void print(String doc) { System.out.println("Printing: " + doc); }
}

// Multi-function printer implements all
class MultiFunctionPrinter implements Printer, Scanner, Fax {
    public void print(String doc) { System.out.println("Printing: " + doc); }
    public void scan(String doc) { System.out.println("Scanning: " + doc); }
    public void fax(String doc) { System.out.println("Faxing: " + doc); }
}`}
          />
        </QuestionCard>

        {/* Question 51: Dependency Inversion Principle */}
        <QuestionCard
          number={51}
          title="Dependency Inversion Principle (DIP)"
          question="What is Dependency Inversion Principle?"
          answer="The Dependency Inversion Principle states that: (1) High-level modules should not depend on low-level modules. Both should depend on abstractions. (2) Abstractions should not depend on details. Details should depend on abstractions. This promotes loose coupling and makes the system more flexible and testable through dependency injection."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// ❌ BAD: High-level depends directly on low-level
class EmailSender {
    public void send(String message) {
        System.out.println("Sending email: " + message);
    }
}

class SMSSender {
    public void send(String message) {
        System.out.println("Sending SMS: " + message);
    }
}

class NotificationServiceBad {
    private EmailSender emailSender = new EmailSender();
    // To change to SMS, we must modify this class!
    
    public void notify(String message) {
        emailSender.send(message);
    }
}

// ✅ GOOD: Depend on abstractions
interface MessageSender {
    void send(String message);
}

class EmailSenderGood implements MessageSender {
    public void send(String message) {
        System.out.println("Sending email: " + message);
    }
}

class SMSSenderGood implements MessageSender {
    public void send(String message) {
        System.out.println("Sending SMS: " + message);
    }
}

class PushNotificationSender implements MessageSender {
    public void send(String message) {
        System.out.println("Sending push notification: " + message);
    }
}

// High-level module depends on abstraction
class NotificationService {
    private final MessageSender sender;
    
    // Dependency injection via constructor
    public NotificationService(MessageSender sender) {
        this.sender = sender;
    }
    
    public void notify(String message) {
        sender.send(message);
    }
}

// Usage - easy to change behavior without modifying NotificationService
public class DIPDemo {
    public static void main(String[] args) {
        // Can easily switch between different senders
        NotificationService emailService = new NotificationService(new EmailSenderGood());
        NotificationService smsService = new NotificationService(new SMSSenderGood());
        NotificationService pushService = new NotificationService(new PushNotificationSender());
        
        emailService.notify("Hello via Email");
        smsService.notify("Hello via SMS");
        pushService.notify("Hello via Push");
    }
}`}
          />
        </QuestionCard>

        {/* Question 52: Coupling vs Cohesion */}
        <QuestionCard
          number={52}
          title="Coupling vs Cohesion"
          question="What is the difference between coupling and cohesion?"
          answer="Coupling measures how dependent different modules/classes are on each other. Low coupling is desirable for maintainability. Cohesion measures how focused a single module/class is - how closely its responsibilities are related. High cohesion is desirable. Good design aims for low coupling and high cohesion, making code more modular, reusable, and easier to maintain."
          marks={5}
          icon={<Eye className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Coupling vs Cohesion Comparison"
            headers={["Feature", "Coupling", "Cohesion"]}
            rows={[
              ["Definition", "Inter-module dependency", "Intra-module focus"],
              ["Desired Level", "Low (loose coupling)", "High (strong cohesion)"],
              ["Impact of Change", "Changes propagate", "Changes localized"],
              ["Reusability", "Hard with high coupling", "Easy with high cohesion"],
              ["Example", "Class A uses Class B directly", "Class only does one thing well"],
              ["Testing", "Harder with high coupling", "Easier with high cohesion"],
            ]}
          />
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Examples:</h4>
            <div className="space-y-2 text-sm">
              <div><span className="font-bold text-primary">Low Coupling:</span> Using interfaces, dependency injection</div>
              <div><span className="font-bold text-destructive">High Coupling:</span> Direct instantiation of concrete classes</div>
              <div><span className="font-bold text-primary">High Cohesion:</span> A Calculator class only does calculations</div>
              <div><span className="font-bold text-destructive">Low Cohesion:</span> A Utils class doing everything</div>
            </div>
          </div>
          <CodeBlock
            language="java"
            code={`// LOW COUPLING Example
interface Database {
    void save(String data);
}

class MySQLDatabase implements Database {
    public void save(String data) { /* MySQL specific */ }
}

class AppService {
    private Database db;  // Depends on abstraction
    public AppService(Database db) { this.db = db; }  // Dependency injection
    public void saveData(String data) { db.save(data); }
}

// HIGH COUPLING Example (bad)
class BadAppService {
    private MySQLDatabase db = new MySQLDatabase();  // Direct dependency
    public void saveData(String data) { db.save(data); }  // Hard to change
}

// HIGH COHESION Example
class UserValidator {
    public boolean isValidEmail(String email) { /* email validation */ }
    public boolean isValidAge(int age) { /* age validation */ }
}
// All methods are related - validating user data

// LOW COHESION Example (bad)
class UtilityClass {
    public void sendEmail() { /* email */ }
    public void calculateTax() { /* tax */ }
    public void compressFile() { /* compress */ }
    public void generatePDF() { /* pdf */ }
}
// Methods are unrelated - low cohesion`}
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
              <span className="font-bold text-primary">Q46:</span>
              <span className="text-muted-foreground ml-1">SOLID = SRP + OCP + LSP + ISP + DIP</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q47:</span>
              <span className="text-muted-foreground ml-1">SRP = One class, one responsibility</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q48:</span>
              <span className="text-muted-foreground ml-1">OCP = Extend, don't modify</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q49:</span>
              <span className="text-muted-foreground ml-1">LSP = Subtypes replace parents</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q50:</span>
              <span className="text-muted-foreground ml-1">ISP = Split fat interfaces</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q51:</span>
              <span className="text-muted-foreground ml-1">DIP = Depend on abstractions</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q52:</span>
              <span className="text-muted-foreground ml-1">Low coupling + High cohesion = Good design</span>
            </div>
          </div>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="SOLID Principles Quiz" />
    </TopicContent>
  );
}