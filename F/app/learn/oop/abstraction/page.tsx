"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { AbstractionVisualizer } from "@/components/visualizations/oop-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
  Eye,
  EyeOff,
} from "lucide-react";

const pythonCode = `# Abstraction in Python using ABC (Abstract Base Class)
from abc import ABC, abstractmethod

# Abstract class - cannot be instantiated directly
class Vehicle(ABC):
    def __init__(self, brand):
        self.brand = brand
    
    # Abstract methods - MUST be implemented by subclasses
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def accelerate(self):
        pass
    
    # Concrete method - can be used as-is
    def honk(self):
        return f"{self.brand} goes beep beep!"

# Concrete class implementing the abstraction
class Car(Vehicle):
    def __init__(self, brand, fuel_type):
        super().__init__(brand)
        self.fuel_type = fuel_type
        self._engine_temp = 0  # Hidden detail
    
    def start(self):
        # Complex internal process hidden
        self._check_fuel()
        self._initialize_engine()
        self._engine_temp = 90
        return f"{self.brand} car started!"
    
    def stop(self):
        self._engine_temp = 0
        return f"{self.brand} car stopped!"
    
    def accelerate(self):
        return f"{self.brand} car accelerating..."
    
    # Hidden implementation details
    def _check_fuel(self):
        print("Checking fuel level...")
    
    def _initialize_engine(self):
        print("Initializing engine systems...")

class ElectricBike(Vehicle):
    def __init__(self, brand, battery_capacity):
        super().__init__(brand)
        self.battery = battery_capacity
    
    def start(self):
        return f"{self.brand} e-bike powered on!"
    
    def stop(self):
        return f"{self.brand} e-bike powered off!"
    
    def accelerate(self):
        return f"{self.brand} e-bike accelerating silently..."

# Using abstraction - same interface, different implementations
def test_drive(vehicle: Vehicle):
    print(vehicle.start())
    print(vehicle.accelerate())
    print(vehicle.honk())
    print(vehicle.stop())

# Can't create Vehicle directly
# v = Vehicle("Generic")  # TypeError!

# Create concrete implementations
car = Car("Toyota", "Gasoline")
bike = ElectricBike("Tesla", "100kWh")

test_drive(car)
test_drive(bike)`;

const javascriptCode = `// Abstraction in JavaScript
// Note: JavaScript doesn't have built-in abstract classes,
// but we can simulate them

class Vehicle {
    constructor(brand) {
        if (new.target === Vehicle) {
            throw new Error("Cannot instantiate abstract class Vehicle");
        }
        this.brand = brand;
    }
    
    // Abstract methods - must be overridden
    start() {
        throw new Error("Method 'start()' must be implemented");
    }
    
    stop() {
        throw new Error("Method 'stop()' must be implemented");
    }
    
    accelerate() {
        throw new Error("Method 'accelerate()' must be implemented");
    }
    
    // Concrete method
    honk() {
        return \`\${this.brand} goes beep beep!\`;
    }
}

// Concrete implementation
class Car extends Vehicle {
    #engineTemp = 0;  // Private field
    
    constructor(brand, fuelType) {
        super(brand);
        this.fuelType = fuelType;
    }
    
    start() {
        this.#checkFuel();
        this.#initializeEngine();
        this.#engineTemp = 90;
        return \`\${this.brand} car started!\`;
    }
    
    stop() {
        this.#engineTemp = 0;
        return \`\${this.brand} car stopped!\`;
    }
    
    accelerate() {
        return \`\${this.brand} car accelerating...\`;
    }
    
    // Hidden details
    #checkFuel() {
        console.log("Checking fuel level...");
    }
    
    #initializeEngine() {
        console.log("Initializing engine systems...");
    }
}

class ElectricBike extends Vehicle {
    constructor(brand, batteryCapacity) {
        super(brand);
        this.battery = batteryCapacity;
    }
    
    start() {
        return \`\${this.brand} e-bike powered on!\`;
    }
    
    stop() {
        return \`\${this.brand} e-bike powered off!\`;
    }
    
    accelerate() {
        return \`\${this.brand} e-bike accelerating silently...\`;
    }
}

// Using abstraction
function testDrive(vehicle) {
    console.log(vehicle.start());
    console.log(vehicle.accelerate());
    console.log(vehicle.honk());
    console.log(vehicle.stop());
}

const car = new Car("Toyota", "Gasoline");
const bike = new ElectricBike("Tesla", "100kWh");

testDrive(car);
testDrive(bike);`;

const javaCode = `// Abstraction in Java using abstract class and interface

// Abstract class
abstract class Vehicle {
    protected String brand;
    
    public Vehicle(String brand) {
        this.brand = brand;
    }
    
    // Abstract methods - must be implemented
    public abstract String start();
    public abstract String stop();
    public abstract String accelerate();
    
    // Concrete method
    public String honk() {
        return brand + " goes beep beep!";
    }
}

// Interface - pure abstraction
interface Electric {
    void charge();
    int getBatteryLevel();
}

// Concrete class implementing abstract class
class Car extends Vehicle {
    private String fuelType;
    private int engineTemp = 0;
    
    public Car(String brand, String fuelType) {
        super(brand);
        this.fuelType = fuelType;
    }
    
    @Override
    public String start() {
        checkFuel();
        initializeEngine();
        engineTemp = 90;
        return brand + " car started!";
    }
    
    @Override
    public String stop() {
        engineTemp = 0;
        return brand + " car stopped!";
    }
    
    @Override
    public String accelerate() {
        return brand + " car accelerating...";
    }
    
    // Hidden implementation
    private void checkFuel() {
        System.out.println("Checking fuel...");
    }
    
    private void initializeEngine() {
        System.out.println("Initializing engine...");
    }
}

// Class implementing both abstract class and interface
class ElectricBike extends Vehicle implements Electric {
    private int batteryLevel = 100;
    
    public ElectricBike(String brand) {
        super(brand);
    }
    
    @Override
    public String start() {
        return brand + " e-bike powered on!";
    }
    
    @Override
    public String stop() {
        return brand + " e-bike powered off!";
    }
    
    @Override
    public String accelerate() {
        return brand + " e-bike accelerating silently...";
    }
    
    @Override
    public void charge() {
        batteryLevel = 100;
    }
    
    @Override
    public int getBatteryLevel() {
        return batteryLevel;
    }
}

public class Main {
    public static void testDrive(Vehicle v) {
        System.out.println(v.start());
        System.out.println(v.accelerate());
        System.out.println(v.stop());
    }
    
    public static void main(String[] args) {
        Vehicle car = new Car("Toyota", "Gasoline");
        Vehicle bike = new ElectricBike("Tesla");
        
        testDrive(car);
        testDrive(bike);
    }
}`;

const cppCode = `// Abstraction in C++ using abstract classes
#include <iostream>
#include <string>
using namespace std;

// Abstract class (has at least one pure virtual function)
class Vehicle {
protected:
    string brand;

public:
    Vehicle(string b) : brand(b) {}
    
    // Pure virtual functions = abstract methods
    virtual string start() = 0;
    virtual string stop() = 0;
    virtual string accelerate() = 0;
    
    // Concrete method
    string honk() {
        return brand + " goes beep beep!";
    }
    
    // Virtual destructor (important for polymorphism)
    virtual ~Vehicle() {}
};

// Concrete implementation
class Car : public Vehicle {
private:
    string fuelType;
    int engineTemp = 0;
    
    void checkFuel() {
        cout << "Checking fuel..." << endl;
    }
    
    void initializeEngine() {
        cout << "Initializing engine..." << endl;
    }

public:
    Car(string brand, string fuel) : Vehicle(brand), fuelType(fuel) {}
    
    string start() override {
        checkFuel();
        initializeEngine();
        engineTemp = 90;
        return brand + " car started!";
    }
    
    string stop() override {
        engineTemp = 0;
        return brand + " car stopped!";
    }
    
    string accelerate() override {
        return brand + " car accelerating...";
    }
};

class ElectricBike : public Vehicle {
private:
    int batteryLevel = 100;

public:
    ElectricBike(string brand) : Vehicle(brand) {}
    
    string start() override {
        return brand + " e-bike powered on!";
    }
    
    string stop() override {
        return brand + " e-bike powered off!";
    }
    
    string accelerate() override {
        return brand + " e-bike accelerating silently...";
    }
};

void testDrive(Vehicle* v) {
    cout << v->start() << endl;
    cout << v->accelerate() << endl;
    cout << v->honk() << endl;
    cout << v->stop() << endl;
}

int main() {
    Car car("Toyota", "Gasoline");
    ElectricBike bike("Tesla");
    
    testDrive(&car);
    cout << "---" << endl;
    testDrive(&bike);
    
    return 0;
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is abstraction in OOP?",
    options: [
      "Making all attributes private",
      "Hiding complex implementation details and showing only essential features",
      "Creating multiple objects from a class",
      "Inheriting from multiple classes",
    ],
    correctAnswer: 1,
    explanation:
      "Abstraction is about hiding complex implementation details and exposing only the essential features through a simplified interface.",
  },
  {
    id: 2,
    question: "What is an abstract class?",
    options: [
      "A class with no methods",
      "A class that cannot be instantiated and may contain abstract methods",
      "A class that is private",
      "A class with only static methods",
    ],
    correctAnswer: 1,
    explanation:
      "An abstract class cannot be instantiated directly and typically contains one or more abstract methods that must be implemented by subclasses.",
  },
  {
    id: 3,
    question: "What is the difference between abstraction and encapsulation?",
    options: [
      "They are the same thing",
      "Abstraction hides complexity by design; encapsulation hides data through access control",
      "Encapsulation is for methods; abstraction is for attributes",
      "Abstraction uses private; encapsulation uses public",
    ],
    correctAnswer: 1,
    explanation:
      "Abstraction focuses on hiding unnecessary details at the design level (what to show). Encapsulation focuses on hiding data at the implementation level (how to protect).",
  },
  {
    id: 4,
    question: "Can you create an instance of an abstract class?",
    options: [
      "Yes, always",
      "No, abstract classes cannot be instantiated",
      "Only in Python",
      "Only if it has no abstract methods",
    ],
    correctAnswer: 1,
    explanation:
      "Abstract classes cannot be instantiated directly. You must create a concrete subclass that implements all abstract methods, then instantiate that.",
  },
  {
    id: 5,
    question: "What is an interface (in Java/C#)?",
    options: [
      "A concrete class with methods",
      "A contract that defines methods a class must implement, with no implementation",
      "A private class",
      "A class that can be instantiated",
    ],
    correctAnswer: 1,
    explanation:
      "An interface defines a contract (method signatures) that implementing classes must follow. It contains no implementation (in traditional interfaces) - just method declarations.",
  },
  {
    id: 6,
    question: "Why is abstraction useful?",
    options: [
      "It makes code run faster",
      "It reduces complexity, improves maintainability, and allows implementation changes without affecting users",
      "It uses less memory",
      "It prevents inheritance",
    ],
    correctAnswer: 1,
    explanation:
      "Abstraction reduces complexity by hiding details, makes code more maintainable, and allows changing implementation without affecting code that uses the abstraction.",
  },
];

export default function AbstractionPage() {
  const result = getSubtopicBySlug("oop", "abstraction");

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
            What is Abstraction?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Abstraction</strong> is the concept of hiding complex 
            implementation details and showing only the necessary features of an object. It focuses 
            on <em>what</em> an object does rather than <em>how</em> it does it.
          </p>
          <p>
            Through abstraction, you create a simplified interface that users can interact with, 
            without needing to understand the underlying complexity.
          </p>
        </div>
      </section>

      {/* Real World Analogy */}
      <section className="mb-12">
        <div className="p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-start gap-3 mb-4">
            <Lightbulb className="h-5 w-5 text-primary mt-0.5" />
            <div>
              <h3 className="font-semibold text-foreground mb-2">Real-World Analogy</h3>
              <p className="text-muted-foreground">
                When you drive a <strong className="text-foreground">car</strong>, you use a simple interface: 
                steering wheel, accelerator, brake. You don't need to understand the combustion engine, 
                fuel injection, or transmission systems. The complexity is 
                <strong className="text-foreground"> abstracted away</strong> - you just press the gas 
                and the car moves!
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Visualizer */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Interactive Visualization
        </h2>
        <AbstractionVisualizer />
      </section>

      {/* Abstraction vs Encapsulation */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Abstraction vs Encapsulation
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-4 bg-card rounded-lg border-2 border-primary/30">
            <div className="flex items-center gap-2 mb-3">
              <EyeOff className="h-5 w-5 text-primary" />
              <h3 className="font-semibold text-foreground">Abstraction</h3>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Hides <strong className="text-foreground">complexity</strong></li>
              <li>• Design-level concept</li>
              <li>• Focuses on <em>what</em> to show</li>
              <li>• Achieved via abstract classes/interfaces</li>
              <li>• Example: Car interface (start, stop, drive)</li>
            </ul>
          </div>
          <div className="p-4 bg-card rounded-lg border-2 border-orange-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Eye className="h-5 w-5 text-orange-500" />
              <h3 className="font-semibold text-foreground">Encapsulation</h3>
            </div>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Hides <strong className="text-foreground">data</strong></li>
              <li>• Implementation-level concept</li>
              <li>• Focuses on <em>how</em> to protect</li>
              <li>• Achieved via access modifiers</li>
              <li>• Example: Private engine temperature variable</li>
            </ul>
          </div>
        </div>
        <div className="mt-4 p-3 bg-muted rounded-lg text-sm text-muted-foreground">
          <strong className="text-foreground">Remember:</strong> Abstraction and encapsulation work together. 
          Abstraction decides what features to expose, while encapsulation implements the protection.
        </div>
      </section>

      {/* Abstract Classes vs Interfaces */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Abstract Classes vs Interfaces
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-3 font-semibold text-foreground">Feature</th>
                <th className="text-left p-3 font-semibold text-foreground">Abstract Class</th>
                <th className="text-left p-3 font-semibold text-foreground">Interface</th>
              </tr>
            </thead>
            <tbody className="text-muted-foreground">
              <tr className="border-b border-border">
                <td className="p-3">Methods</td>
                <td className="p-3">Can have both abstract and concrete</td>
                <td className="p-3">Only abstract (traditionally)</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-3">Variables</td>
                <td className="p-3">Can have instance variables</td>
                <td className="p-3">Only constants (static final)</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-3">Inheritance</td>
                <td className="p-3">Single inheritance</td>
                <td className="p-3">Multiple interfaces allowed</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-3">Constructor</td>
                <td className="p-3">Can have constructors</td>
                <td className="p-3">Cannot have constructors</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-3">Use when</td>
                <td className="p-3">Classes share common code</td>
                <td className="p-3">Defining a contract/capability</td>
              </tr>
            </tbody>
          </table>
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
            { language: "python", label: "Python", code: pythonCode },
            { language: "javascript", label: "JavaScript", code: javascriptCode },
            { language: "java", label: "Java", code: javaCode },
            { language: "cpp", label: "C++", code: cppCode },
          ]}
        />
      </section>

      {/* Common Mistakes */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-destructive/10">
            <AlertTriangle className="h-5 w-5 text-destructive" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">Common Mistakes</h2>
        </div>
        <div className="space-y-3">
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Forgetting to implement all abstract methods
            </h4>
            <p className="text-sm text-muted-foreground">
              If you inherit from an abstract class, you must implement ALL abstract methods. 
              Missing even one will cause a compilation error (or runtime error in Python).
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Over-abstraction
            </h4>
            <p className="text-sm text-muted-foreground">
              Don't create abstract classes for everything. If you only have one implementation, 
              an abstract class might be unnecessary complexity. Use abstraction when you have 
              multiple implementations.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Trying to instantiate abstract classes
            </h4>
            <p className="text-sm text-muted-foreground">
              Abstract classes cannot be instantiated directly. You must create a concrete 
              subclass that implements all abstract methods.
            </p>
          </div>
        </div>
      </section>

      {/* Interview Tips */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <Target className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">Interview Tips</h2>
        </div>
        <div className="space-y-4">
          <div className="p-4 bg-muted/50 rounded-lg">
            <div className="flex items-start gap-2 mb-2">
              <span className="text-primary font-bold">Q:</span>
              <span className="text-muted-foreground">
                "When would you use an abstract class vs an interface?"
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-success font-bold">A:</span>
              <span className="text-foreground">
                "Use abstract class when classes share common code/state (like Vehicle with a brand field). 
                Use interface when defining a capability that unrelated classes might implement 
                (like Flyable for both Bird and Airplane). In Java, you can implement multiple interfaces 
                but extend only one class."
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Quiz */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Test Your Knowledge
        </h2>
        <Quiz questions={quizQuestions} title="Abstraction Quiz" />
      </section>
    </TopicContent>
  );
}
