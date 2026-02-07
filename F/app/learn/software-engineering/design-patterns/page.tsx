// app/topics/software-engineering/design-patterns/page.tsx
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Factory, Package, User, Layers, GitBranch, Settings, Target, Zap, Shield, Cpu, Database, FileCode, Wrench, Puzzle, Box } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

export default function DesignPatternsPage() {
  const result = getSubtopicBySlug("software-engineering", "design-patterns");
  if (!result) return null;

  const { topic, subtopic } = result;

  // Creational Patterns
  const creationalPatterns = [
    {
      name: "Singleton",
      icon: <Target className="h-6 w-6" />,
      description: "Ensures a class has only one instance and provides global access",
      intent: "Control object creation, limiting to single instance",
      structure: "Private constructor, static instance, static access method",
      useCases: ["Database connections", "Logging", "Configuration", "Cache"],
      example: `class Database {
  private static instance: Database;
  private connection: Connection;
  
  private constructor() {
    this.connection = new Connection();
  }
  
  public static getInstance(): Database {
    if (!Database.instance) {
      Database.instance = new Database();
    }
    return Database.instance;
  }
  
  public query(sql: string): Result {
    return this.connection.execute(sql);
  }
}`,
      pros: ["Controlled access", "Reduced memory usage", "Global state management"],
      cons: ["Difficult to test", "Hidden dependencies", "Violates Single Responsibility"],
    },
    {
      name: "Factory Method",
      icon: <Factory className="h-6 w-6" />,
      description: "Creates objects without specifying exact class, using factory method",
      intent: "Define interface for creating objects, let subclasses decide",
      structure: "Creator class with factory method, Concrete creators, Product interface",
      useCases: ["UI frameworks", "Document editors", "Payment processors"],
      example: `interface PaymentProcessor {
  process(amount: number): void;
}

class CreditCardProcessor implements PaymentProcessor {
  process(amount: number) {
    console.log(\`Processing $\{amount} via Credit Card\`);
  }
}

class PayPalProcessor implements PaymentProcessor {
  process(amount: number) {
    console.log(\`Processing $\{amount} via PayPal\`);
  }
}

abstract class PaymentFactory {
  abstract createProcessor(): PaymentProcessor;
  
  processPayment(amount: number) {
    const processor = this.createProcessor();
    processor.process(amount);
  }
}

class CreditCardFactory extends PaymentFactory {
  createProcessor(): PaymentProcessor {
    return new CreditCardProcessor();
  }
}`,
      pros: ["Loose coupling", "Easy to extend", "Single Responsibility"],
      cons: ["More classes", "Complex hierarchy", "Overhead for simple cases"],
    },
    {
      name: "Abstract Factory",
      icon: <Package className="h-6 w-6" />,
      description: "Creates families of related objects without specifying concrete classes",
      intent: "Provide interface for creating families of related objects",
      structure: "Abstract factory, Concrete factories, Product interfaces",
      useCases: ["Cross-platform UI", "Database drivers", "Theme systems"],
      example: `interface Button {
  render(): void;
}

interface Checkbox {
  render(): void;
}

class WindowsButton implements Button {
  render() { console.log("Windows button"); }
}

class MacButton implements Button {
  render() { console.log("Mac button"); }
}

interface GUIFactory {
  createButton(): Button;
  createCheckbox(): Checkbox;
}

class WindowsFactory implements GUIFactory {
  createButton(): Button { return new WindowsButton(); }
  createCheckbox(): Checkbox { return new WindowsCheckbox(); }
}`,
      pros: ["Product consistency", "Easy to swap families", "Promotes composition"],
      cons: ["Complex", "Hard to extend", "Many interfaces"],
    },
    {
      name: "Builder",
      icon: <Wrench className="h-6 w-6" />,
      description: "Constructs complex objects step by step, separating construction from representation",
      intent: "Separate construction of complex object from its representation",
      structure: "Builder interface, Concrete builders, Director, Product",
      useCases: ["SQL queries", "HTTP requests", "Complex configurations"],
      example: `class Pizza {
  private dough: string = '';
  private sauce: string = '';
  private toppings: string[] = [];
  
  setDough(dough: string) { this.dough = dough; }
  setSauce(sauce: string) { this.sauce = sauce; }
  addTopping(topping: string) { this.toppings.push(topping); }
}

interface PizzaBuilder {
  buildDough(): void;
  buildSauce(): void;
  buildToppings(): void;
  getPizza(): Pizza;
}

class HawaiianPizzaBuilder implements PizzaBuilder {
  private pizza: Pizza = new Pizza();
  
  buildDough() { this.pizza.setDough("cross"); }
  buildSauce() { this.pizza.setSauce("mild"); }
  buildToppings() { 
    this.pizza.addTopping("ham");
    this.pizza.addTopping("pineapple");
  }
  getPizza() { return this.pizza; }
}`,
      pros: ["Fine control", "Reusable builders", "Immutability"],
      cons: ["Verbose", "Many classes", "Runtime errors possible"],
    },
    {
      name: "Prototype",
      icon: <Box className="h-6 w-6" />,
      description: "Creates new objects by cloning existing ones",
      intent: "Create new objects by copying existing prototype",
      structure: "Prototype interface, Concrete prototypes, Client",
      useCases: ["Game objects", "Document templates", "Caching"],
      example: `interface Shape {
  clone(): Shape;
  draw(): void;
}

class Circle implements Shape {
  private radius: number;
  private color: string;
  
  constructor(radius: number, color: string) {
    this.radius = radius;
    this.color = color;
  }
  
  clone(): Shape {
    return new Circle(this.radius, this.color);
  }
  
  draw() {
    console.log(\`Drawing circle with radius $\{this.radius}\`);
  }
}

// Usage
const circle1 = new Circle(10, "red");
const circle2 = circle1.clone();`,
      pros: ["Reduced subclassing", "Runtime object creation", "Performance"],
      cons: ["Cloning complex objects", "Deep vs shallow copy", "Circular references"],
    },
  ];

  // Structural Patterns
  const structuralPatterns = [
    {
      name: "Adapter",
      icon: <Puzzle className="h-6 w-6" />,
      description: "Allows incompatible interfaces to work together",
      intent: "Convert interface of class into another interface clients expect",
      structure: "Target interface, Adaptee, Adapter",
      useCases: ["Legacy integration", "Third-party libraries", "API wrappers"],
      example: `// Old system
class OldPrinter {
  printText(text: string) {
    console.log(\`Old printer: $\{text}\`);
  }
}

// New interface
interface Printer {
  print(content: string): void;
}

// Adapter
class PrinterAdapter implements Printer {
  private oldPrinter: OldPrinter;
  
  constructor(oldPrinter: OldPrinter) {
    this.oldPrinter = oldPrinter;
  }
  
  print(content: string): void {
    this.oldPrinter.printText(content);
  }
}`,
      pros: ["Reuse legacy code", "Single Responsibility", "Flexibility"],
      cons: ["Additional complexity", "Performance overhead", "Overuse warning"],
    },
    {
      name: "Decorator",
      icon: <Layers className="h-6 w-6" />,
      description: "Adds behavior to objects dynamically without inheritance",
      intent: "Attach additional responsibilities to object dynamically",
      structure: "Component interface, Concrete component, Decorator",
      useCases: ["Stream processing", "UI components", "Middleware"],
      example: `interface Coffee {
  cost(): number;
  description(): string;
}

class SimpleCoffee implements Coffee {
  cost() { return 5; }
  description() { return "Simple coffee"; }
}

abstract class CoffeeDecorator implements Coffee {
  protected coffee: Coffee;
  
  constructor(coffee: Coffee) {
    this.coffee = coffee;
  }
  
  cost() { return this.coffee.cost(); }
  description() { return this.coffee.description(); }
}

class MilkDecorator extends CoffeeDecorator {
  cost() { return this.coffee.cost() + 2; }
  description() { return this.coffee.description() + ", milk"; }
}`,
      pros: ["Flexible extension", "Avoids subclass explosion", "Runtime behavior"],
      cons: ["Many small objects", "Complex instantiation", "Debugging difficult"],
    },
    {
      name: "Facade",
      icon: <Package className="h-6 w-6" />,
      description: "Provides simplified interface to complex subsystem",
      intent: "Provide unified interface to set of interfaces in subsystem",
      structure: "Facade, Complex subsystem classes",
      useCases: ["API simplification", "Library wrappers", "System integration"],
      example: `class CPU {
  execute() { console.log("CPU executing"); }
}

class Memory {
  load() { console.log("Memory loading"); }
}

class HardDrive {
  read() { console.log("Hard drive reading"); }
}

class ComputerFacade {
  private cpu: CPU = new CPU();
  private memory: Memory = new Memory();
  private hardDrive: HardDrive = new HardDrive();
  
  start() {
    this.cpu.execute();
    this.memory.load();
    this.hardDrive.read();
    console.log("Computer started");
  }
}`,
      pros: ["Simplified interface", "Reduced coupling", "Better organization"],
      cons: ["Can become god object", "Limited flexibility", "Hidden functionality"],
    },
    {
      name: "Proxy",
      icon: <User className="h-6 w-6" />,
      description: "Provides surrogate or placeholder for another object",
      intent: "Provide surrogate or placeholder for another object",
      structure: "Subject interface, Real subject, Proxy",
      useCases: ["Lazy loading", "Access control", "Caching", "Logging"],
      example: `interface Image {
  display(): void;
}

class RealImage implements Image {
  private filename: string;
  
  constructor(filename: string) {
    this.filename = filename;
    this.loadFromDisk();
  }
  
  private loadFromDisk() {
    console.log(\`Loading $\{this.filename}\`);
  }
  
  display() {
    console.log(\`Displaying $\{this.filename}\`);
  }
}

class ImageProxy implements Image {
  private realImage: RealImage | null = null;
  private filename: string;
  
  constructor(filename: string) {
    this.filename = filename;
  }
  
  display() {
    if (!this.realImage) {
      this.realImage = new RealImage(this.filename);
    }
    this.realImage.display();
  }
}`,
      pros: ["Lazy initialization", "Access control", "Performance optimization"],
      cons: ["Response time", "Complexity", "Debugging overhead"],
    },
    {
      name: "Composite",
      icon: <GitBranch className="h-6 w-6" />,
      description: "Composes objects into tree structures to represent part-whole hierarchies",
      intent: "Compose objects into tree structures",
      structure: "Component interface, Leaf, Composite",
      useCases: ["File systems", "UI components", "Organization charts"],
      example: `interface FileSystemComponent {
  showDetails(indent: string): void;
}

class File implements FileSystemComponent {
  constructor(private name: string) {}
  
  showDetails(indent: string) {
    console.log(\`$\{indent}File: $\{this.name}\`);
  }
}

class Directory implements FileSystemComponent {
  private children: FileSystemComponent[] = [];
  
  constructor(private name: string) {}
  
  add(component: FileSystemComponent) {
    this.children.push(component);
  }
  
  showDetails(indent: string) {
    console.log(\`$\{indent}Directory: $\{this.name}\`);
    this.children.forEach(child => 
      child.showDetails(indent + "  ")
    );
  }
}`,
      pros: ["Uniform treatment", "Easy to add components", "Simplified client"],
      cons: ["Type safety", "Performance", "Limited operations"],
    },
  ];

  // Behavioral Patterns
  const behavioralPatterns = [
    {
      name: "Observer",
      icon: <Zap className="h-6 w-6" />,
      description: "Defines one-to-many dependency between objects",
      intent: "Define one-to-many dependency between objects",
      structure: "Subject, Observer interface, Concrete observers",
      useCases: ["Event handling", "Pub/Sub systems", "MVC architecture"],
      example: `interface Observer {
  update(temperature: number, humidity: number): void;
}

interface Subject {
  registerObserver(o: Observer): void;
  removeObserver(o: Observer): void;
  notifyObservers(): void;
}

class WeatherStation implements Subject {
  private observers: Observer[] = [];
  private temperature: number = 0;
  private humidity: number = 0;
  
  registerObserver(o: Observer) {
    this.observers.push(o);
  }
  
  removeObserver(o: Observer) {
    const index = this.observers.indexOf(o);
    if (index > -1) this.observers.splice(index, 1);
  }
  
  notifyObservers() {
    this.observers.forEach(observer => 
      observer.update(this.temperature, this.humidity)
    );
  }
  
  setMeasurements(temp: number, humidity: number) {
    this.temperature = temp;
    this.humidity = humidity;
    this.notifyObservers();
  }
}`,
      pros: ["Loose coupling", "Broadcast communication", "Dynamic relationships"],
      cons: ["Memory leaks", "Unexpected updates", "Performance"],
    },
    {
      name: "Strategy",
      icon: <Settings className="h-6 w-6" />,
      description: "Defines family of algorithms, encapsulates each, makes interchangeable",
      intent: "Define family of algorithms, encapsulate each, make interchangeable",
      structure: "Strategy interface, Concrete strategies, Context",
      useCases: ["Sorting algorithms", "Payment methods", "Compression algorithms"],
      example: `interface PaymentStrategy {
  pay(amount: number): void;
}

class CreditCardStrategy implements PaymentStrategy {
  private cardNumber: string;
  
  constructor(cardNumber: string) {
    this.cardNumber = cardNumber;
  }
  
  pay(amount: number) {
    console.log(\`Paid $\{amount} using credit card $\{this.cardNumber}\`);
  }
}

class ShoppingCart {
  private paymentStrategy: PaymentStrategy;
  
  setPaymentStrategy(strategy: PaymentStrategy) {
    this.paymentStrategy = strategy;
  }
  
  checkout(amount: number) {
    this.paymentStrategy.pay(amount);
  }
}`,
      pros: ["Eliminates conditionals", "Open/Closed principle", "Runtime algorithm switching"],
      cons: ["Increased objects", "Client awareness", "Communication overhead"],
    },
    {
      name: "Command",
      icon: <FileCode className="h-6 w-6" />,
      description: "Encapsulates request as object, allowing parameterization and queuing",
      intent: "Encapsulate request as object",
      structure: "Command interface, Concrete command, Invoker, Receiver",
      useCases: ["Undo/Redo", "Job queues", "Transactional systems"],
      example: `interface Command {
  execute(): void;
  undo(): void;
}

class Light {
  turnOn() { console.log("Light is ON"); }
  turnOff() { console.log("Light is OFF"); }
}

class LightOnCommand implements Command {
  private light: Light;
  
  constructor(light: Light) {
    this.light = light;
  }
  
  execute() { this.light.turnOn(); }
  undo() { this.light.turnOff(); }
}

class RemoteControl {
  private command: Command;
  
  setCommand(command: Command) {
    this.command = command;
  }
  
  pressButton() {
    this.command.execute();
  }
}`,
      pros: ["Decouples invoker", "Undo/Redo support", "Command queuing"],
      cons: ["Increased classes", "Complex implementation", "Memory usage"],
    },
    {
      name: "State",
      icon: <Cpu className="h-6 w-6" />,
      description: "Allows object to alter behavior when internal state changes",
      intent: "Allow object to alter behavior when its internal state changes",
      structure: "Context, State interface, Concrete states",
      useCases: ["TCP connections", "Document states", "Game characters"],
      example: `interface State {
  handle(context: DocumentContext): void;
}

class DocumentContext {
  private state: State;
  
  constructor(state: State) {
    this.state = state;
  }
  
  setState(state: State) {
    this.state = state;
  }
  
  request() {
    this.state.handle(this);
  }
}

class DraftState implements State {
  handle(context: DocumentContext) {
    console.log("Document is in draft state");
    context.setState(new ReviewState());
  }
}

class ReviewState implements State {
  handle(context: DocumentContext) {
    console.log("Document is under review");
    context.setState(new PublishedState());
  }
}`,
      pros: ["Localizes state behavior", "Eliminates conditionals", "Easy to add states"],
      cons: ["Many classes", "Context coupling", "Complex transitions"],
    },
    {
      name: "Template Method",
      icon: <Database className="h-6 w-6" />,
      description: "Defines skeleton of algorithm in superclass, lets subclasses override steps",
      intent: "Define skeleton of algorithm in operation, deferring steps to subclasses",
      structure: "Abstract class, Concrete classes",
      useCases: ["Framework hooks", "Data processing", "Report generation"],
      example: `abstract class DataProcessor {
  // Template method
  process() {
    this.readData();
    this.transformData();
    this.saveData();
  }
  
  abstract readData(): void;
  abstract transformData(): void;
  
  saveData() {
    console.log("Saving data to database...");
  }
}

class CSVProcessor extends DataProcessor {
  readData() {
    console.log("Reading CSV file...");
  }
  
  transformData() {
    console.log("Transforming CSV data...");
  }
}

class JSONProcessor extends DataProcessor {
  readData() {
    console.log("Reading JSON file...");
  }
  
  transformData() {
    console.log("Transforming JSON data...");
  }
}`,
      pros: ["Code reuse", "Inversion of control", "Consistent algorithm"],
      cons: ["Limited flexibility", "Inheritance issues", "Debugging difficulty"],
    },
  ];

  // Real-world Examples
  const realWorldExamples = [
    {
      pattern: "Singleton",
      application: "Database Connection Pool",
      description: "Manages limited database connections across application",
      benefits: ["Resource optimization", "Thread safety", "Centralized management"],
      code: `class ConnectionPool {
  private static instance: ConnectionPool;
  private connections: Connection[] = [];
  private MAX_CONNECTIONS = 10;
  
  private constructor() {
    this.initializePool();
  }
  
  public static getInstance(): ConnectionPool {
    if (!ConnectionPool.instance) {
      ConnectionPool.instance = new ConnectionPool();
    }
    return ConnectionPool.instance;
  }
  
  public getConnection(): Connection {
    return this.connections.pop() || new Connection();
  }
}`,
    },
    {
      pattern: "Observer",
      application: "Stock Market Notifications",
      description: "Notifies multiple investors when stock prices change",
      benefits: ["Real-time updates", "Decoupled components", "Scalable"],
      code: `class StockMarket {
  private investors: Investor[] = [];
  private stockPrice: number = 100;
  
  subscribe(investor: Investor) {
    this.investors.push(investor);
  }
  
  setPrice(price: number) {
    this.stockPrice = price;
    this.investors.forEach(investor => 
      investor.update(this.stockPrice)
    );
  }
}`,
    },
    {
      pattern: "Strategy",
      application: "E-commerce Shipping",
      description: "Different shipping strategies (Standard, Express, Overnight)",
      benefits: ["Flexible pricing", "Easy to add new methods", "Clean code"],
      code: `interface ShippingStrategy {
  calculate(weight: number): number;
}

class StandardShipping implements ShippingStrategy {
  calculate(weight: number): number {
    return weight * 1.5;
  }
}

class Order {
  private shipping: ShippingStrategy;
  
  setShipping(strategy: ShippingStrategy) {
    this.shipping = strategy;
  }
  
  calculateShipping(weight: number): number {
    return this.shipping.calculate(weight);
  }
}`,
    },
    {
      pattern: "Decorator",
      application: "Web Request Middleware",
      description: "Adds functionality to HTTP requests (logging, auth, compression)",
      benefits: ["Modular", "Chainable", "Runtime composition"],
      code: `interface Handler {
  handle(request: Request): Response;
}

class LoggingDecorator implements Handler {
  constructor(private next: Handler) {}
  
  handle(request: Request): Response {
    console.log(\`Request: $\{request.url}\`);
    return this.next.handle(request);
  }
}`,
    },
  ];

  // Pattern Categories
  const patternCategories = [
    {
      category: "Creational",
      icon: <Factory className="h-5 w-5" />,
      description: "Deal with object creation mechanisms",
      patterns: ["Singleton", "Factory Method", "Abstract Factory", "Builder", "Prototype"],
      purpose: "Control object creation, provide flexibility in what gets created",
      whenToUse: "When object creation is complex or needs to be controlled",
    },
    {
      category: "Structural",
      icon: <Puzzle className="h-5 w-5" />,
      description: "Deal with object composition and relationships",
      patterns: ["Adapter", "Decorator", "Facade", "Proxy", "Composite", "Bridge", "Flyweight"],
      purpose: "Form larger structures from individual objects",
      whenToUse: "When you need to compose objects or manage relationships",
    },
    {
      category: "Behavioral",
      icon: <Zap className="h-5 w-5" />,
      description: "Deal with object interaction and responsibility distribution",
      patterns: ["Observer", "Strategy", "Command", "State", "Template Method", "Iterator", "Mediator"],
      purpose: "Define communication between objects and assign responsibilities",
      whenToUse: "When objects need to communicate or algorithms need to vary",
    },
  ];

  // Anti-patterns
  const antiPatterns = [
    {
      name: "God Object",
      description: "One class does too many things, knows too much",
      symptoms: ["3000+ lines of code", "Many dependencies", "Frequent changes"],
      fix: "Apply Single Responsibility Principle, split into smaller classes",
      example: `// BAD: God object
class Application {
  // Handles everything: UI, business logic, database, logging, etc.
}

// GOOD: Separate responsibilities
class UserService { }
class DatabaseService { }
class Logger { }`,
    },
    {
      name: "Spaghetti Code",
      description: "Unstructured, tangled code with no clear separation",
      symptoms: ["Goto statements", "Deep nesting", "Global variables"],
      fix: "Apply design patterns, refactor with clear structure",
      example: `// BAD: Spaghetti code
if (condition1) {
  if (condition2) {
    // ... 10 more nested levels
  }
}

// GOOD: Clean structure
function processCondition1() {
  if (!condition1) return;
  processCondition2();
}`,
    },
    {
      name: "Golden Hammer",
      description: "Using same solution (pattern) for every problem",
      symptoms: ["Singleton everywhere", "Factory for simple objects", "Over-engineered"],
      fix: "Choose patterns based on problem, not habit",
      example: `// BAD: Using Singleton unnecessarily
class Logger {
  private static instance: Logger;
  // ... but we only need one logger instance?
}

// GOOD: Simple static class might suffice
class Logger {
  static log(message: string) { console.log(message); }
}`,
    },
    {
      name: "Circular Dependency",
      description: "Two or more classes depend on each other",
      symptoms: ["Import cycles", "Compilation errors", "Tight coupling"],
      fix: "Use Dependency Injection, introduce interfaces, refactor",
      example: `// BAD: Circular dependency
class A { constructor(private b: B) {} }
class B { constructor(private a: A) {} }

// GOOD: Break the cycle
interface IA { /* methods */ }
class A implements IA { constructor(private b: B) {} }
class B { constructor(private a: IA) {} }`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which pattern ensures a class has only one instance?",
      options: ["Factory Method", "Singleton", "Builder", "Prototype"],
      correctAnswer: 1,
      explanation: "Singleton pattern ensures only one instance of a class exists and provides global access.",
    },
    {
      id: 2,
      question: "What pattern would you use to add logging to HTTP requests without modifying the core handler?",
      options: ["Adapter", "Decorator", "Facade", "Proxy"],
      correctAnswer: 1,
      explanation: "Decorator pattern allows adding behavior dynamically without modifying the original object.",
    },
    {
      id: 3,
      question: "Which pattern is best for implementing undo/redo functionality?",
      options: ["Observer", "Command", "Strategy", "State"],
      correctAnswer: 1,
      explanation: "Command pattern encapsulates requests as objects, making undo/redo operations possible.",
    },
    {
      id: 4,
      question: "What pattern defines a family of algorithms and makes them interchangeable?",
      options: ["Template Method", "Strategy", "Observer", "Factory Method"],
      correctAnswer: 1,
      explanation: "Strategy pattern defines a family of algorithms, encapsulates each, and makes them interchangeable.",
    },
    {
      id: 5,
      question: "Which pattern converts the interface of a class into another interface clients expect?",
      options: ["Adapter", "Bridge", "Proxy", "Facade"],
      correctAnswer: 0,
      explanation: "Adapter pattern allows incompatible interfaces to work together by converting one interface to another.",
    },
    {
      id: 6,
      question: "What pattern provides a simplified interface to a complex subsystem?",
      options: ["Facade", "Proxy", "Decorator", "Composite"],
      correctAnswer: 0,
      explanation: "Facade pattern provides a simplified interface to a complex subsystem of classes.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Design Patterns</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Design Patterns</strong> are reusable solutions to common software design problems. 
            They represent best practices evolved over time by experienced software developers.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Why Design Patterns Matter</h4>
                <p className="text-sm text-muted-foreground">
                  Think of design patterns as proven architectural blueprints for software. 
                  Just as architects use patterns for buildings (like "open plan" or "split level"), 
                  developers use patterns for software structure. They provide shared vocabulary, 
                  prevent reinventing the wheel, and create maintainable, scalable code.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Pattern Categories */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Pattern Categories</h2>
          <div className="grid md:grid-cols-3 gap-4">
            {patternCategories.map((category, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-3 bg-primary/10 rounded-lg">
                      {category.icon}
                    </div>
                    <div>
                      <h3 className="font-bold text-xl text-foreground">{category.category}</h3>
                      <p className="text-sm text-muted-foreground">{category.description}</p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">Common Patterns</h4>
                      <div className="flex flex-wrap gap-2">
                        {category.patterns.map((pattern, i) => (
                          <Badge key={i} variant="outline">{pattern}</Badge>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">Primary Purpose</h4>
                      <p className="text-sm text-muted-foreground">{category.purpose}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">When to Use</h4>
                      <p className="text-sm text-muted-foreground">{category.whenToUse}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Creational Patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Creational Patterns</h2>
          <p className="text-muted-foreground mb-6">
            Control object creation mechanisms, trying to create objects in a manner suitable to the situation.
          </p>
          
          <Tabs defaultValue="singleton" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              {creationalPatterns.map((pattern, idx) => (
                <TabsTrigger key={idx} value={pattern.name.toLowerCase().replace(' ', '-')}>
                  {pattern.name}
                </TabsTrigger>
              ))}
            </TabsList>
            
            {creationalPatterns.map((pattern, idx) => (
              <TabsContent key={idx} value={pattern.name.toLowerCase().replace(' ', '-')}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-3 bg-primary/10 rounded-lg">
                        {pattern.icon}
                      </div>
                      <div>
                        <h3 className="font-bold text-2xl text-foreground">{pattern.name}</h3>
                        <p className="text-muted-foreground mt-1">{pattern.description}</p>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Intent</h4>
                          <p className="text-muted-foreground">{pattern.intent}</p>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Structure</h4>
                          <div className="p-3 bg-muted rounded-lg">
                            <p className="text-sm font-mono text-muted-foreground">{pattern.structure}</p>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Common Use Cases</h4>
                          <div className="space-y-2">
                            {pattern.useCases.map((useCase, i) => (
                              <div key={i} className="flex items-center gap-2 p-2 bg-muted rounded">
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                                <span className="text-sm text-muted-foreground">{useCase}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Example Implementation</h4>
                          <div className="bg-[var(--code-bg)] p-4 rounded-lg overflow-x-auto">
                            <pre className="text-sm font-mono">
                              <code>{pattern.example}</code>
                            </pre>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Pros</h4>
                            <ul className="space-y-1">
                              {pattern.pros.map((pro, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-muted-foreground">{pro}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Cons</h4>
                            <ul className="space-y-1">
                              {pattern.cons.map((con, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-muted-foreground">{con}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Structural Patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Structural Patterns</h2>
          <p className="text-muted-foreground mb-6">
            Deal with object composition and relationships, forming larger structures from individual objects.
          </p>
          
          <Tabs defaultValue="adapter" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              {structuralPatterns.map((pattern, idx) => (
                <TabsTrigger key={idx} value={pattern.name.toLowerCase().replace(' ', '-')}>
                  {pattern.name}
                </TabsTrigger>
              ))}
            </TabsList>
            
            {structuralPatterns.map((pattern, idx) => (
              <TabsContent key={idx} value={pattern.name.toLowerCase().replace(' ', '-')}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-3 bg-primary/10 rounded-lg">
                        {pattern.icon}
                      </div>
                      <div>
                        <h3 className="font-bold text-2xl text-foreground">{pattern.name}</h3>
                        <p className="text-muted-foreground mt-1">{pattern.description}</p>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Intent</h4>
                          <p className="text-muted-foreground">{pattern.intent}</p>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Structure</h4>
                          <div className="p-3 bg-muted rounded-lg">
                            <p className="text-sm font-mono text-muted-foreground">{pattern.structure}</p>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Common Use Cases</h4>
                          <div className="space-y-2">
                            {pattern.useCases.map((useCase, i) => (
                              <div key={i} className="flex items-center gap-2 p-2 bg-muted rounded">
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                                <span className="text-sm text-muted-foreground">{useCase}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Example Implementation</h4>
                          <div className="bg-[var(--code-bg)] p-4 rounded-lg overflow-x-auto">
                            <pre className="text-sm font-mono">
                              <code>{pattern.example}</code>
                            </pre>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Pros</h4>
                            <ul className="space-y-1">
                              {pattern.pros.map((pro, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-muted-foreground">{pro}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Cons</h4>
                            <ul className="space-y-1">
                              {pattern.cons.map((con, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-muted-foreground">{con}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Behavioral Patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Behavioral Patterns</h2>
          <p className="text-muted-foreground mb-6">
            Deal with object interaction and responsibility distribution among objects.
          </p>
          
          <Tabs defaultValue="observer" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              {behavioralPatterns.map((pattern, idx) => (
                <TabsTrigger key={idx} value={pattern.name.toLowerCase().replace(' ', '-')}>
                  {pattern.name}
                </TabsTrigger>
              ))}
            </TabsList>
            
            {behavioralPatterns.map((pattern, idx) => (
              <TabsContent key={idx} value={pattern.name.toLowerCase().replace(' ', '-')}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-3 bg-primary/10 rounded-lg">
                        {pattern.icon}
                      </div>
                      <div>
                        <h3 className="font-bold text-2xl text-foreground">{pattern.name}</h3>
                        <p className="text-muted-foreground mt-1">{pattern.description}</p>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Intent</h4>
                          <p className="text-muted-foreground">{pattern.intent}</p>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Structure</h4>
                          <div className="p-3 bg-muted rounded-lg">
                            <p className="text-sm font-mono text-muted-foreground">{pattern.structure}</p>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Common Use Cases</h4>
                          <div className="space-y-2">
                            {pattern.useCases.map((useCase, i) => (
                              <div key={i} className="flex items-center gap-2 p-2 bg-muted rounded">
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                                <span className="text-sm text-muted-foreground">{useCase}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Example Implementation</h4>
                          <div className="bg-[var(--code-bg)] p-4 rounded-lg overflow-x-auto">
                            <pre className="text-sm font-mono">
                              <code>{pattern.example}</code>
                            </pre>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Pros</h4>
                            <ul className="space-y-1">
                              {pattern.pros.map((pro, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-muted-foreground">{pro}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Cons</h4>
                            <ul className="space-y-1">
                              {pattern.cons.map((con, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-muted-foreground">{con}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Real-world Examples */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Real-world Applications</h2>
          <div className="space-y-6">
            {realWorldExamples.map((example, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <div className="flex flex-col md:flex-row md:items-start gap-6">
                    <div className="md:w-2/3">
                      <div className="flex items-center gap-3 mb-4">
                        <Badge variant="secondary">{example.pattern}</Badge>
                        <h4 className="font-bold text-xl text-foreground">{example.application}</h4>
                      </div>
                      
                      <p className="text-muted-foreground mb-4">{example.description}</p>
                      
                      <div>
                        <h5 className="font-semibold text-foreground mb-2">Key Benefits</h5>
                        <div className="flex flex-wrap gap-2">
                          {example.benefits.map((benefit, i) => (
                            <Badge key={i} variant="outline" className="bg-green-50 text-green-700">
                              {benefit}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    <div className="md:w-1/3">
                      <div className="bg-[var(--code-bg)] p-4 rounded-lg overflow-x-auto">
                        <pre className="text-xs font-mono">
                          <code>{example.code}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Anti-patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Anti-patterns</h2>
          <p className="text-muted-foreground mb-6">
            These are common bad practices that design patterns help you avoid.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            {antiPatterns.map((antiPattern, idx) => (
              <Card key={idx} className="border-red-200">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <AlertCircle className="h-5 w-5 text-red-500" />
                    <h4 className="font-bold text-xl text-foreground">{antiPattern.name}</h4>
                  </div>
                  
                  <p className="text-muted-foreground mb-4">{antiPattern.description}</p>
                  
                  <div className="space-y-4">
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">Symptoms</h5>
                      <div className="flex flex-wrap gap-2">
                        {antiPattern.symptoms.map((symptom, i) => (
                          <Badge key={i} variant="destructive">{symptom}</Badge>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">Fix</h5>
                      <p className="text-sm text-muted-foreground">{antiPattern.fix}</p>
                    </div>
                    
                    <div>
                      <h5 className="font-semibold text-foreground mb-2">Example</h5>
                      <div className="bg-[var(--code-bg)] p-3 rounded-lg overflow-x-auto">
                        <pre className="text-xs font-mono">
                          <code>{antiPattern.example}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Pattern Selection Guide */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Pattern Selection Guide</h2>
          
          <Card>
            <CardContent className="p-6">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-muted">
                      <th className="border border-border p-3 text-left text-foreground">Problem</th>
                      <th className="border border-border p-3 text-left text-foreground">Consider These Patterns</th>
                      <th className="border border-border p-3 text-left text-foreground">Quick Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Need to ensure only one instance</td>
                      <td className="border border-border p-3">Singleton</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">Use for: DB connections, config, logging</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Creating complex objects</td>
                      <td className="border border-border p-3">Builder, Factory Method, Abstract Factory</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">Builder for step-by-step, Factory for families</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Adding behavior dynamically</td>
                      <td className="border border-border p-3">Decorator</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">Better than inheritance for runtime changes</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Simplifying complex system</td>
                      <td className="border border-border p-3">Facade</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">When clients need simple interface to complex code</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Algorithm selection at runtime</td>
                      <td className="border border-border p-3">Strategy</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">Replace conditional logic with objects</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">One-to-many notifications</td>
                      <td className="border border-border p-3">Observer</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">For event-driven, pub/sub systems</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Need undo/redo functionality</td>
                      <td className="border border-border p-3">Command</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">Encapsulate requests as objects</Badge>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground">Object behavior depends on state</td>
                      <td className="border border-border p-3">State</td>
                      <td className="border border-border p-3">
                        <Badge variant="outline">Replace state conditionals with objects</Badge>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              
              <div className="mt-4 p-4 bg-primary/5 rounded-lg">
                <h4 className="font-semibold mb-2 text-foreground">Golden Rule</h4>
                <p className="text-sm text-muted-foreground">
                  <strong>Don't force patterns.</strong> Use them when they solve actual problems in your code. 
                  Patterns should emerge from refactoring, not be applied prematurely.
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Best Practices */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Best Practices</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="h-5 w-5 text-primary" />
                <h4 className="font-semibold text-foreground">Start Simple</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                Begin with simple code. Apply patterns during refactoring when you see 
                duplication, complexity, or inflexibility. Don't over-engineer from the start.
              </p>
            </div>
            
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-5 w-5 text-primary" />
                <h4 className="font-semibold text-foreground">Know SOLID Principles</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                Patterns work best with SOLID principles. Understand Single Responsibility, 
                Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.
              </p>
            </div>
            
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Target className="h-5 w-5 text-primary" />
                <h4 className="font-semibold text-foreground">Choose Wisely</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                Not every problem needs a pattern. Sometimes a simple function or class is enough. 
                Use patterns to solve specific problems, not as a goal in themselves.
              </p>
            </div>
            
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Settings className="h-5 w-5 text-primary" />
                <h4 className="font-semibold text-foreground">Refactor Gradually</h4>
              </div>
              <p className="text-sm text-muted-foreground">
                Apply patterns incrementally. Refactor small parts at a time, ensuring tests pass. 
                This reduces risk and makes changes manageable.
              </p>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Pattern Knowledge</h2>
          <Quiz questions={quizQuestions} title="Design Patterns Quiz" />
        </section>

        {/* Resources */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Further Resources</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <Card>
              <CardContent className="p-6">
                <h4 className="font-semibold mb-3 text-foreground">Essential Books</h4>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Design Patterns: Elements of Reusable OO Software</span>
                    <Badge variant="outline">Gang of Four</Badge>
                  </li>
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Head First Design Patterns</span>
                    <Badge variant="outline">Beginner Friendly</Badge>
                  </li>
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Patterns of Enterprise Application Architecture</span>
                    <Badge variant="outline">Martin Fowler</Badge>
                  </li>
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Clean Architecture</span>
                    <Badge variant="outline">Robert C. Martin</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h4 className="font-semibold mb-3 text-foreground">Online Resources</h4>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Refactoring.Guru</span>
                    <Badge variant="default">Free</Badge>
                  </li>
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>SourceMaking Patterns</span>
                    <Badge variant="default">Free</Badge>
                  </li>
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Dofactory .NET Patterns</span>
                    <Badge variant="default">Free</Badge>
                  </li>
                  <li className="flex justify-between items-center p-2 bg-muted rounded">
                    <span>Pattern Language (C2 Wiki)</span>
                    <Badge variant="default">Community</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="md:col-span-2">
              <CardContent className="p-6">
                <h4 className="font-semibold mb-3 text-foreground">Practice Problems</h4>
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <h5 className="font-medium mb-2 text-foreground">Beginner</h5>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Logger with Singleton</li>
                      <li>• Payment system with Strategy</li>
                      <li>• File system with Composite</li>
                      <li>• Coffee shop with Decorator</li>
                    </ul>
                  </div>
                  <div className="p-3 bg-muted rounded-lg">
                    <h5 className="font-medium mb-2 text-foreground">Intermediate</h5>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• E-commerce with Factory</li>
                      <li>• Text editor with Command</li>
                      <li>• Weather station with Observer</li>
                      <li>• Game character with State</li>
                    </ul>
                  </div>
                  <div className="p-3 bg-muted rounded-lg">
                    <h5 className="font-medium mb-2 text-foreground">Advanced</h5>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• ORM with Proxy/Decorator</li>
                      <li>• Workflow engine with Template</li>
                      <li>• Microservices with Facade</li>
                      <li>• Plugin system with Strategy</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </TopicContent>
  );
}