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
  Box,
  GitBranch,
  Layers,
  Eye,
  Zap,
  ShoppingCart,
  Bell,
  Paintbrush,
} from "lucide-react";

const designPatternsCode = `// Design Patterns Examples

// 1. SINGLETON PATTERN
class Singleton {
    private static Singleton instance;
    private Singleton() {}  // Private constructor
    
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
    
    public void showMessage() {
        System.out.println("Singleton instance: " + this.hashCode());
    }
}

// 2. FACTORY PATTERN
interface Product {
    void create();
}

class ConcreteProductA implements Product {
    public void create() { System.out.println("Product A created"); }
}

class ConcreteProductB implements Product {
    public void create() { System.out.println("Product B created"); }
}

class ProductFactory {
    public static Product createProduct(String type) {
        if (type.equals("A")) return new ConcreteProductA();
        else if (type.equals("B")) return new ConcreteProductB();
        return null;
    }
}

// 3. STRATEGY PATTERN
interface PaymentStrategy {
    void pay(double amount);
}

class CreditCardPayment implements PaymentStrategy {
    public void pay(double amount) {
        System.out.println("Paid " + amount + " using Credit Card");
    }
}

class PayPalPayment implements PaymentStrategy {
    public void pay(double amount) {
        System.out.println("Paid " + amount + " using PayPal");
    }
}

class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    
    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }
    
    public void checkout(double amount) {
        paymentStrategy.pay(amount);
    }
}

// 4. OBSERVER PATTERN
import java.util.ArrayList;
import java.util.List;

interface Observer {
    void update(String message);
}

class User implements Observer {
    private String name;
    public User(String name) { this.name = name; }
    public void update(String message) {
        System.out.println(name + " received: " + message);
    }
}

class NewsAgency {
    private List<Observer> observers = new ArrayList<>();
    
    public void subscribe(Observer observer) {
        observers.add(observer);
    }
    
    public void unsubscribe(Observer observer) {
        observers.remove(observer);
    }
    
    public void publishNews(String news) {
        for (Observer observer : observers) {
            observer.update(news);
        }
    }
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What are design patterns?",
    options: [
      "Ready-to-use code snippets",
      "Reusable solutions to common software design problems",
      "Programming languages features",
      "Database design techniques",
    ],
    correctAnswer: 1,
    explanation: "Design patterns are proven, reusable solutions to common design problems that occur in software development.",
  },
  {
    id: 2,
    question: "What type of pattern is Singleton?",
    options: [
      "Structural",
      "Behavioral",
      "Creational",
      "Architectural",
    ],
    correctAnswer: 2,
    explanation: "Singleton is a creational pattern because it deals with object creation mechanisms.",
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

export default function DesignPatternsPage() {
  const result = getSubtopicBySlug("oop", "design-patterns");

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
            OOP Design Patterns
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Design patterns</strong> are reusable solutions to 
            common software design problems. They provide templates and best practices for solving 
            recurring design challenges.
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

        {/* Question 53: Design Patterns Classification */}
        <QuestionCard
          number={53}
          title="Design Patterns - Classification"
          question="What are design patterns? Classify them (Creational, Structural, Behavioral)."
          answer="Design patterns are proven reusable solutions to common software design problems. Three categories: Creational patterns deal with object creation (Singleton, Factory, Builder). Structural patterns deal with class/object composition (Adapter, Decorator, Facade). Behavioral patterns deal with object interaction and responsibility (Observer, Strategy, Command)."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Design Pattern Categories:</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
              <div>
                <span className="font-bold text-primary">Creational</span>
                <ul className="list-disc list-inside text-muted-foreground mt-1">
                  <li>Singleton</li>
                  <li>Factory</li>
                  <li>Builder</li>
                  <li>Prototype</li>
                  <li>Abstract Factory</li>
                </ul>
              </div>
              <div>
                <span className="font-bold text-primary">Structural</span>
                <ul className="list-disc list-inside text-muted-foreground mt-1">
                  <li>Adapter</li>
                  <li>Decorator</li>
                  <li>Facade</li>
                  <li>Proxy</li>
                  <li>Bridge</li>
                </ul>
              </div>
              <div>
                <span className="font-bold text-primary">Behavioral</span>
                <ul className="list-disc list-inside text-muted-foreground mt-1">
                  <li>Observer</li>
                  <li>Strategy</li>
                  <li>Command</li>
                  <li>Template</li>
                  <li>State</li>
                </ul>
              </div>
            </div>
          </div>
        </QuestionCard>

        {/* Question 54: Singleton Pattern */}
        <QuestionCard
          number={54}
          title="Singleton Pattern"
          question="Explain Singleton pattern with code."
          answer="Singleton ensures a class has only one instance and provides global access to it. Implementation: private constructor, static instance variable, static getInstance() method. Use cases: logging, configuration, database connections, thread pools. Thread-safe implementation is important in multi-threaded environments."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Basic Singleton (not thread-safe)
class SingletonBasic {
    private static SingletonBasic instance;
    
    private SingletonBasic() {}  // Private constructor
    
    public static SingletonBasic getInstance() {
        if (instance == null) {
            instance = new SingletonBasic();
        }
        return instance;
    }
    
    public void showMessage() {
        System.out.println("Singleton instance: " + this.hashCode());
    }
}

// Thread-safe Singleton (Double-checked locking)
class SingletonThreadSafe {
    private static volatile SingletonThreadSafe instance;
    
    private SingletonThreadSafe() {}
    
    public static SingletonThreadSafe getInstance() {
        if (instance == null) {
            synchronized (SingletonThreadSafe.class) {
                if (instance == null) {
                    instance = new SingletonThreadSafe();
                }
            }
        }
        return instance;
    }
}

// Eager initialization (always thread-safe)
class SingletonEager {
    private static final SingletonEager instance = new SingletonEager();
    
    private SingletonEager() {}
    
    public static SingletonEager getInstance() {
        return instance;
    }
}

// Bill Pugh Singleton (best practice)
class SingletonBillPugh {
    private SingletonBillPugh() {}
    
    private static class SingletonHelper {
        private static final SingletonBillPugh INSTANCE = new SingletonBillPugh();
    }
    
    public static SingletonBillPugh getInstance() {
        return SingletonHelper.INSTANCE;
    }
    
    // Business methods
    private String config = "Default Config";
    
    public String getConfig() { return config; }
    public void setConfig(String config) { this.config = config; }
}

// Usage
public class SingletonDemo {
    public static void main(String[] args) {
        // All getInstance calls return the same object
        SingletonBillPugh s1 = SingletonBillPugh.getInstance();
        SingletonBillPugh s2 = SingletonBillPugh.getInstance();
        
        System.out.println(s1 == s2);  // true
        System.out.println(s1.hashCode() == s2.hashCode());  // true
        
        s1.setConfig("Production Config");
        System.out.println(s2.getConfig());  // "Production Config"
    }
}`}
          />
        </QuestionCard>

        {/* Question 55: Factory Pattern */}
        <QuestionCard
          number={55}
          title="Factory Pattern"
          question="Explain Factory pattern with code."
          answer="Factory pattern provides an interface for creating objects without specifying the exact class. It delegates object creation to subclasses or a separate factory class. This promotes loose coupling and makes code more maintainable when adding new product types."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Product interface
interface Vehicle {
    void drive();
}

// Concrete products
class Car implements Vehicle {
    public void drive() {
        System.out.println("Driving a car");
    }
}

class Bike implements Vehicle {
    public void drive() {
        System.out.println("Riding a bike");
    }
}

class Truck implements Vehicle {
    public void drive() {
        System.out.println("Driving a truck");
    }
}

// Factory class
class VehicleFactory {
    public static Vehicle createVehicle(String type) {
        if (type == null) {
            return null;
        }
        switch (type.toLowerCase()) {
            case "car":
                return new Car();
            case "bike":
                return new Bike();
            case "truck":
                return new Truck();
            default:
                throw new IllegalArgumentException("Unknown vehicle type: " + type);
        }
    }
}

// Abstract Factory pattern (more advanced)
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
        // Simple factory
        Vehicle car = VehicleFactory.createVehicle("car");
        Vehicle bike = VehicleFactory.createVehicle("bike");
        
        car.drive();   // Driving a car
        bike.drive();  // Riding a bike
        
        // Abstract factory
        GUIFactory factory = new WindowsFactory();
        Button button = factory.createButton();
        button.render();
    }
}`}
          />
        </QuestionCard>

        {/* Question 56: Strategy Pattern */}
        <QuestionCard
          number={56}
          title="Strategy Pattern"
          question="Explain Strategy pattern with code."
          answer="Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. It lets the algorithm vary independently from clients that use it. Useful when you have multiple ways to perform an operation and want to switch between them dynamically."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Strategy interface
interface SortStrategy {
    void sort(int[] numbers);
}

// Concrete strategies
class BubbleSort implements SortStrategy {
    public void sort(int[] numbers) {
        System.out.println("Using Bubble Sort");
        int n = numbers.length;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (numbers[j] > numbers[j+1]) {
                    int temp = numbers[j];
                    numbers[j] = numbers[j+1];
                    numbers[j+1] = temp;
                }
            }
        }
    }
}

class QuickSort implements SortStrategy {
    public void sort(int[] numbers) {
        System.out.println("Using Quick Sort");
        quickSort(numbers, 0, numbers.length - 1);
    }
    
    private void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    private int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }
}

class MergeSort implements SortStrategy {
    public void sort(int[] numbers) {
        System.out.println("Using Merge Sort");
        mergeSort(numbers, 0, numbers.length - 1);
    }
    
    private void mergeSort(int[] arr, int l, int r) {
        if (l < r) {
            int m = l + (r - l) / 2;
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
            merge(arr, l, m, r);
        }
    }
    
    private void merge(int[] arr, int l, int m, int r) {
        // Merge implementation
    }
}

// Context class
class DataProcessor {
    private SortStrategy strategy;
    
    public void setStrategy(SortStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void processData(int[] data) {
        strategy.sort(data);
        System.out.print("Sorted data: ");
        for (int num : data) {
            System.out.print(num + " ");
        }
        System.out.println();
    }
}

// Real-world example: Payment methods
interface PaymentStrategy {
    void pay(double amount);
}

class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;
    
    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }
    
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using Credit Card ending in " + 
                           cardNumber.substring(cardNumber.length() - 4));
    }
}

class PayPalPayment implements PaymentStrategy {
    private String email;
    
    public PayPalPayment(String email) {
        this.email = email;
    }
    
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using PayPal account " + email);
    }
}

class CryptoPayment implements PaymentStrategy {
    private String walletAddress;
    
    public CryptoPayment(String walletAddress) {
        this.walletAddress = walletAddress;
    }
    
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " in Bitcoin to wallet " + walletAddress);
    }
}

class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    
    public void setPaymentMethod(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }
    
    public void checkout(double amount) {
        paymentStrategy.pay(amount);
    }
}

// Usage
public class StrategyDemo {
    public static void main(String[] args) {
        int[] data = {64, 34, 25, 12, 22, 11, 90};
        
        DataProcessor processor = new DataProcessor();
        
        // Use different strategies dynamically
        processor.setStrategy(new BubbleSort());
        processor.processData(data.clone());
        
        processor.setStrategy(new QuickSort());
        processor.processData(data.clone());
        
        // Payment example
        ShoppingCart cart = new ShoppingCart();
        cart.setPaymentMethod(new CreditCardPayment("1234-5678-9012-3456"));
        cart.checkout(99.99);
        
        cart.setPaymentMethod(new PayPalPayment("user@example.com"));
        cart.checkout(49.99);
        
        cart.setPaymentMethod(new CryptoPayment("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"));
        cart.checkout(199.99);
    }
}`}
          />
        </QuestionCard>

        {/* Question 57: Observer Pattern */}
        <QuestionCard
          number={57}
          title="Observer Pattern"
          question="Explain Observer pattern with code."
          answer="Observer pattern defines a one-to-many dependency where when one object (subject) changes state, all its dependents (observers) are notified automatically. Common in event handling systems, publish-subscribe systems, and MVC architecture."
          marks={5}
          icon={<Bell className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`import java.util.ArrayList;
import java.util.List;

// Observer interface
interface Observer {
    void update(String message);
}

// Subject interface
interface Subject {
    void attach(Observer observer);
    void detach(Observer observer);
    void notifyObservers();
}

// Concrete Subject
class NewsPublisher implements Subject {
    private List<Observer> observers = new ArrayList<>();
    private String latestNews;
    
    public void attach(Observer observer) {
        observers.add(observer);
        System.out.println("Observer attached");
    }
    
    public void detach(Observer observer) {
        observers.remove(observer);
        System.out.println("Observer detached");
    }
    
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(latestNews);
        }
    }
    
    public void publishNews(String news) {
        this.latestNews = news;
        System.out.println("\\nPublishing: " + news);
        notifyObservers();
    }
}

// Concrete Observers
class EmailSubscriber implements Observer {
    private String email;
    
    public EmailSubscriber(String email) {
        this.email = email;
    }
    
    public void update(String message) {
        System.out.println("Email sent to " + email + ": " + message);
    }
}

class SMSSubscriber implements Observer {
    private String phoneNumber;
    
    public SMSSubscriber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }
    
    public void update(String message) {
        System.out.println("SMS sent to " + phoneNumber + ": " + message);
    }
}

class PushNotificationSubscriber implements Observer {
    private String deviceId;
    
    public PushNotificationSubscriber(String deviceId) {
        this.deviceId = deviceId;
    }
    
    public void update(String message) {
        System.out.println("Push notification to " + deviceId + ": " + message);
    }
}

// Weather Station example
class WeatherData implements Subject {
    private List<Observer> observers = new ArrayList<>();
    private float temperature;
    private float humidity;
    private float pressure;
    
    public void attach(Observer observer) {
        observers.add(observer);
    }
    
    public void detach(Observer observer) {
        observers.remove(observer);
    }
    
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(String.format("Temp: %.1f°C, Humidity: %.1f%%, Pressure: %.1fhPa", 
                                         temperature, humidity, pressure));
        }
    }
    
    public void setMeasurements(float temp, float humidity, float pressure) {
        this.temperature = temp;
        this.humidity = humidity;
        this.pressure = pressure;
        measurementsChanged();
    }
    
    private void measurementsChanged() {
        notifyObservers();
    }
}

// Display implementations
class CurrentConditionsDisplay implements Observer {
    private float temperature;
    private float humidity;
    
    public void update(String message) {
        // Parse message or store data
        System.out.println("Current Conditions: " + message);
    }
}

class StatisticsDisplay implements Observer {
    private List<Float> temperatures = new ArrayList<>();
    
    public void update(String message) {
        System.out.println("Statistics updated: " + message);
    }
}

// Usage
public class ObserverDemo {
    public static void main(String[] args) {
        // News example
        NewsPublisher publisher = new NewsPublisher();
        
        Observer emailSub = new EmailSubscriber("user@example.com");
        Observer smsSub = new SMSSubscriber("+1234567890");
        Observer pushSub = new PushNotificationSubscriber("device-123");
        
        publisher.attach(emailSub);
        publisher.attach(smsSub);
        publisher.attach(pushSub);
        
        publisher.publishNews("Breaking: Major discovery!");
        publisher.publishNews("Weather: Sunny day ahead!");
        
        publisher.detach(smsSub);
        publisher.publishNews("Sports: Team wins championship!");
        
        // Weather Station example
        WeatherData weatherData = new WeatherData();
        
        CurrentConditionsDisplay currentDisplay = new CurrentConditionsDisplay();
        StatisticsDisplay statisticsDisplay = new StatisticsDisplay();
        
        weatherData.attach(currentDisplay);
        weatherData.attach(statisticsDisplay);
        
        weatherData.setMeasurements(25.5f, 65.0f, 1013.5f);
        weatherData.setMeasurements(26.0f, 70.0f, 1012.0f);
    }
}`}
          />
        </QuestionCard>

        {/* Question 58: MVC Pattern */}
        <QuestionCard
          number={58}
          title="MVC Pattern"
          question="Explain MVC (Model-View-Controller) pattern."
          answer="MVC is an architectural pattern that separates application into three components: Model (data and business logic), View (user interface), Controller (handles input and coordinates). This separation promotes modularity, multiple views of same data, and easier maintenance. Used extensively in web frameworks (Spring MVC, Django, Ruby on Rails)."
          marks={5}
          icon={<Layers className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">MVC Components:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li><strong>Model:</strong> Data, business logic, database operations</li>
              <li><strong>View:</strong> UI presentation, user interface</li>
              <li><strong>Controller:</strong> Handles user input, updates model, selects view</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`// MODEL - Student data and business logic
class Student {
    private String name;
    private int id;
    private String grade;
    
    public Student(String name, int id, String grade) {
        this.name = name;
        this.id = id;
        this.grade = grade;
    }
    
    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getId() { return id; }
    public String getGrade() { return grade; }
    public void setGrade(String grade) { this.grade = grade; }
}

// VIEW - Display student data
class StudentView {
    public void displayStudent(Student student) {
        System.out.println("=== Student Information ===");
        System.out.println("ID: " + student.getId());
        System.out.println("Name: " + student.getName());
        System.out.println("Grade: " + student.getGrade());
        System.out.println("=========================");
    }
    
    public void showMessage(String message) {
        System.out.println("Message: " + message);
    }
}

// CONTROLLER - Handles input and coordinates
class StudentController {
    private Student model;
    private StudentView view;
    
    public StudentController(Student model, StudentView view) {
        this.model = model;
        this.view = view;
    }
    
    public void setStudentName(String name) {
        model.setName(name);
        updateView();
    }
    
    public void setStudentGrade(String grade) {
        model.setGrade(grade);
        updateView();
    }
    
    public Student getStudent() {
        return model;
    }
    
    public void updateView() {
        view.displayStudent(model);
    }
    
    public void handleUserInput(String name, String grade) {
        if (name != null && !name.isEmpty()) {
            setStudentName(name);
            view.showMessage("Name updated to: " + name);
        }
        if (grade != null && !grade.isEmpty()) {
            setStudentGrade(grade);
            view.showMessage("Grade updated to: " + grade);
        }
    }
}

// Web MVC Example (conceptual)
/*
// Spring MVC style controller
@Controller
public class UserController {
    @Autowired
    private UserService userService;
    
    @GetMapping("/users/{id}")
    public String getUser(@PathVariable Long id, Model model) {
        User user = userService.findById(id);
        model.addAttribute("user", user);
        return "user-profile";  // View name
    }
    
    @PostMapping("/users")
    public String createUser(@ModelAttribute User user) {
        userService.save(user);
        return "redirect:/users";
    }
}
*/

// Usage
public class MVCDemo {
    public static void main(String[] args) {
        // Create model
        Student student = new Student("John Doe", 12345, "A");
        
        // Create view
        StudentView view = new StudentView();
        
        // Create controller
        StudentController controller = new StudentController(student, view);
        
        // Display initial data
        controller.updateView();
        
        // Handle user input (simulated)
        controller.handleUserInput("Jane Smith", "A+");
        controller.handleUserInput("Bob Johnson", "B");
        
        // Get updated model
        Student updatedStudent = controller.getStudent();
        System.out.println("\\nFinal student name: " + updatedStudent.getName());
    }
}`}
          />
        </QuestionCard>

        {/* Question 59: Decorator Pattern */}
        <QuestionCard
          number={59}
          title="Decorator Pattern"
          question="When would you use the Decorator pattern?"
          answer="Decorator pattern is used to add responsibilities to objects dynamically without subclassing. Use when: (1) You need to add features to objects at runtime, (2) Subclassing would create too many classes (class explosion), (3) You want to combine features in different ways. Common examples: UI components (scrollbars, borders), I/O streams (BufferedInputStream), pizza toppings, coffee add-ons."
          marks={5}
          icon={<Paintbrush className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`// Component interface
interface Coffee {
    double getCost();
    String getDescription();
}

// Concrete Component
class SimpleCoffee implements Coffee {
    public double getCost() {
        return 2.0;
    }
    
    public String getDescription() {
        return "Simple Coffee";
    }
}

// Decorator abstract class
abstract class CoffeeDecorator implements Coffee {
    protected Coffee decoratedCoffee;
    
    public CoffeeDecorator(Coffee coffee) {
        this.decoratedCoffee = coffee;
    }
    
    public double getCost() {
        return decoratedCoffee.getCost();
    }
    
    public String getDescription() {
        return decoratedCoffee.getDescription();
    }
}

// Concrete Decorators
class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }
    
    public double getCost() {
        return super.getCost() + 0.5;
    }
    
    public String getDescription() {
        return super.getDescription() + ", Milk";
    }
}

class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }
    
    public double getCost() {
        return super.getCost() + 0.2;
    }
    
    public String getDescription() {
        return super.getDescription() + ", Sugar";
    }
}

class WhippedCreamDecorator extends CoffeeDecorator {
    public WhippedCreamDecorator(Coffee coffee) {
        super(coffee);
    }
    
    public double getCost() {
        return super.getCost() + 0.7;
    }
    
    public String getDescription() {
        return super.getDescription() + ", Whipped Cream";
    }
}

// Pizza example
interface Pizza {
    String getDescription();
    double getPrice();
}

class BasicPizza implements Pizza {
    public String getDescription() {
        return "Basic Pizza";
    }
    
    public double getPrice() {
        return 8.0;
    }
}

abstract class ToppingDecorator implements Pizza {
    protected Pizza pizza;
    
    public ToppingDecorator(Pizza pizza) {
        this.pizza = pizza;
    }
}

class CheeseTopping extends ToppingDecorator {
    public CheeseTopping(Pizza pizza) {
        super(pizza);
    }
    
    public String getDescription() {
        return pizza.getDescription() + ", Cheese";
    }
    
    public double getPrice() {
        return pizza.getPrice() + 1.5;
    }
}

class PepperoniTopping extends ToppingDecorator {
    public PepperoniTopping(Pizza pizza) {
        super(pizza);
    }
    
    public String getDescription() {
        return pizza.getDescription() + ", Pepperoni";
    }
    
    public double getPrice() {
        return pizza.getPrice() + 2.0;
    }
}

class MushroomTopping extends ToppingDecorator {
    public MushroomTopping(Pizza pizza) {
        super(pizza);
    }
    
    public String getDescription() {
        return pizza.getDescription() + ", Mushrooms";
    }
    
    public double getPrice() {
        return pizza.getPrice() + 1.0;
    }
}

// Usage
public class DecoratorDemo {
    public static void main(String[] args) {
        // Coffee example
        Coffee coffee = new SimpleCoffee();
        System.out.println(coffee.getDescription() + " $" + coffee.getCost());
        
        Coffee milkCoffee = new MilkDecorator(new SimpleCoffee());
        System.out.println(milkCoffee.getDescription() + " $" + milkCoffee.getCost());
        
        Coffee sugarMilkCoffee = new SugarDecorator(new MilkDecorator(new SimpleCoffee()));
        System.out.println(sugarMilkCoffee.getDescription() + " $" + sugarMilkCoffee.getCost());
        
        Coffee deluxeCoffee = new WhippedCreamDecorator(
                              new SugarDecorator(
                              new MilkDecorator(
                              new SimpleCoffee())));
        System.out.println(deluxeCoffee.getDescription() + " $" + deluxeCoffee.getCost());
        
        // Pizza example
        Pizza pizza = new BasicPizza();
        System.out.println(pizza.getDescription() + " = $" + pizza.getPrice());
        
        Pizza cheesePizza = new CheeseTopping(new BasicPizza());
        System.out.println(cheesePizza.getDescription() + " = $" + cheesePizza.getPrice());
        
        Pizza deluxePizza = new PepperoniTopping(
                           new MushroomTopping(
                           new CheeseTopping(
                           new BasicPizza())));
        System.out.println(deluxePizza.getDescription() + " = $" + deluxePizza.getPrice());
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
              <span className="font-bold text-primary">Q53:</span>
              <span className="text-muted-foreground ml-1">Creational, Structural, Behavioral patterns</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q54:</span>
              <span className="text-muted-foreground ml-1">Singleton = one instance, global access</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q55:</span>
              <span className="text-muted-foreground ml-1">Factory = object creation abstraction</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q56:</span>
              <span className="text-muted-foreground ml-1">Strategy = interchangeable algorithms</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q57:</span>
              <span className="text-muted-foreground ml-1">Observer = one-to-many notification</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q58:</span>
              <span className="text-muted-foreground ml-1">MVC = Model-View-Controller separation</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q59:</span>
              <span className="text-muted-foreground ml-1">Decorator = dynamic feature addition</span>
            </div>
          </div>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="Design Patterns Quiz" />
    </TopicContent>
  );
}