"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { PolymorphismVisualizer } from "@/components/visualizations/oop-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
} from "lucide-react";

const pythonCode = `# Polymorphism in Python

# Runtime Polymorphism (Method Overriding)
class Shape:
    def area(self):
        pass
    
    def describe(self):
        return f"I am a shape with area {self.area()}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):  # Override
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):  # Override
        return 3.14159 * self.radius ** 2

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    
    def area(self):  # Override
        return 0.5 * self.base * self.height

# Polymorphism in action - same interface, different behavior
shapes = [Rectangle(4, 5), Circle(3), Triangle(6, 4)]

for shape in shapes:
    print(shape.describe())
# Output:
# I am a shape with area 20
# I am a shape with area 28.27431
# I am a shape with area 12.0

# Duck typing (Python-specific polymorphism)
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Duck:
    def speak(self):
        return "Quack!"

# No common parent needed in Python!
def make_speak(animal):
    print(animal.speak())

make_speak(Dog())   # Woof!
make_speak(Cat())   # Meow!
make_speak(Duck())  # Quack!`;

const javascriptCode = `// Polymorphism in JavaScript

// Runtime Polymorphism (Method Overriding)
class Shape {
    area() {
        return 0;
    }
    
    describe() {
        return \`I am a shape with area \${this.area()}\`;
    }
}

class Rectangle extends Shape {
    constructor(width, height) {
        super();
        this.width = width;
        this.height = height;
    }
    
    area() {  // Override
        return this.width * this.height;
    }
}

class Circle extends Shape {
    constructor(radius) {
        super();
        this.radius = radius;
    }
    
    area() {  // Override
        return Math.PI * this.radius ** 2;
    }
}

class Triangle extends Shape {
    constructor(base, height) {
        super();
        this.base = base;
        this.height = height;
    }
    
    area() {  // Override
        return 0.5 * this.base * this.height;
    }
}

// Polymorphism in action
const shapes = [
    new Rectangle(4, 5),
    new Circle(3),
    new Triangle(6, 4)
];

shapes.forEach(shape => {
    console.log(shape.describe());
});

// Duck typing in JavaScript
const dog = { speak: () => "Woof!" };
const cat = { speak: () => "Meow!" };

function makeSpeak(animal) {
    console.log(animal.speak());
}

makeSpeak(dog);  // Woof!
makeSpeak(cat);  // Meow!`;

const javaCode = `// Polymorphism in Java

// Abstract class for shapes
abstract class Shape {
    abstract double area();  // Abstract method
    
    String describe() {
        return "I am a shape with area " + area();
    }
}

class Rectangle extends Shape {
    private double width, height;
    
    public Rectangle(double w, double h) {
        width = w;
        height = h;
    }
    
    @Override
    double area() {
        return width * height;
    }
}

class Circle extends Shape {
    private double radius;
    
    public Circle(double r) {
        radius = r;
    }
    
    @Override
    double area() {
        return Math.PI * radius * radius;
    }
}

// Method Overloading (Compile-time Polymorphism)
class Calculator {
    // Same method name, different parameters
    int add(int a, int b) {
        return a + b;
    }
    
    double add(double a, double b) {
        return a + b;
    }
    
    int add(int a, int b, int c) {
        return a + b + c;
    }
}

public class Main {
    public static void main(String[] args) {
        // Runtime polymorphism
        Shape[] shapes = {
            new Rectangle(4, 5),
            new Circle(3)
        };
        
        for (Shape shape : shapes) {
            System.out.println(shape.describe());
        }
        
        // Compile-time polymorphism
        Calculator calc = new Calculator();
        System.out.println(calc.add(1, 2));       // 3
        System.out.println(calc.add(1.5, 2.5));   // 4.0
        System.out.println(calc.add(1, 2, 3));    // 6
    }
}`;

const cppCode = `// Polymorphism in C++
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

// Abstract base class
class Shape {
public:
    virtual double area() = 0;  // Pure virtual function
    
    string describe() {
        return "I am a shape with area " + to_string(area());
    }
    
    virtual ~Shape() {}  // Virtual destructor
};

class Rectangle : public Shape {
private:
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double area() override {
        return width * height;
    }
};

class Circle : public Shape {
private:
    double radius;
public:
    Circle(double r) : radius(r) {}
    
    double area() override {
        return M_PI * radius * radius;
    }
};

// Function Overloading (Compile-time Polymorphism)
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    double add(double a, double b) {
        return a + b;
    }
    
    int add(int a, int b, int c) {
        return a + b + c;
    }
};

int main() {
    // Runtime polymorphism using pointers
    vector<Shape*> shapes;
    shapes.push_back(new Rectangle(4, 5));
    shapes.push_back(new Circle(3));
    
    for (Shape* shape : shapes) {
        cout << shape->describe() << endl;
    }
    
    // Cleanup
    for (Shape* shape : shapes) {
        delete shape;
    }
    
    // Compile-time polymorphism
    Calculator calc;
    cout << calc.add(1, 2) << endl;       // 3
    cout << calc.add(1.5, 2.5) << endl;   // 4.0
    cout << calc.add(1, 2, 3) << endl;    // 6
    
    return 0;
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is polymorphism in OOP?",
    options: [
      "Creating multiple classes",
      "The ability of objects to take many forms or behave differently based on context",
      "Hiding implementation details",
      "Inheriting from multiple classes",
    ],
    correctAnswer: 1,
    explanation:
      "Polymorphism (meaning 'many forms') allows objects of different classes to be treated through the same interface, with each providing its own implementation.",
  },
  {
    id: 2,
    question: "What is the difference between method overloading and overriding?",
    options: [
      "They are the same thing",
      "Overloading: same name, different parameters. Overriding: same signature in child class",
      "Overriding: same name, different parameters. Overloading: same signature in child class",
      "Overloading only works in Python",
    ],
    correctAnswer: 1,
    explanation:
      "Method overloading has the same method name but different parameters (compile-time). Method overriding redefines a parent's method in the child class with the same signature (runtime).",
  },
  {
    id: 3,
    question: "What is runtime polymorphism?",
    options: [
      "Polymorphism resolved at compile time",
      "Polymorphism resolved during program execution through method overriding",
      "Creating objects at runtime",
      "Dynamic memory allocation",
    ],
    correctAnswer: 1,
    explanation:
      "Runtime (dynamic) polymorphism is when the method to call is determined at runtime based on the actual object type, typically achieved through method overriding.",
  },
  {
    id: 4,
    question: "What is 'duck typing' in Python?",
    options: [
      "A way to create duck objects",
      "If it walks like a duck and quacks like a duck, it's a duck - behavior matters, not type",
      "A type checking mechanism",
      "A way to inherit from multiple classes",
    ],
    correctAnswer: 1,
    explanation:
      "Duck typing means Python cares about what methods an object has, not what class it belongs to. If an object has the required method, it can be used regardless of its type.",
  },
  {
    id: 5,
    question: "In Java, what keyword indicates a method can be overridden?",
    options: [
      "override",
      "virtual",
      "Methods are overridable by default (non-final, non-private, non-static)",
      "abstract",
    ],
    correctAnswer: 2,
    explanation:
      "In Java, non-private, non-final, non-static methods can be overridden by default. The @Override annotation is optional but recommended for clarity and compile-time checking.",
  },
  {
    id: 6,
    question: "Why is polymorphism useful?",
    options: [
      "It makes code run faster",
      "It allows writing flexible, extensible code that can work with objects of multiple types",
      "It reduces memory usage",
      "It prevents inheritance",
    ],
    correctAnswer: 1,
    explanation:
      "Polymorphism allows you to write generic code that works with any object implementing a certain interface, making code more flexible, maintainable, and extensible.",
  },
];

export default function PolymorphismPage() {
  const result = getSubtopicBySlug("oop", "polymorphism");

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
            What is Polymorphism?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Polymorphism</strong> (from Greek, meaning "many forms") 
            is the ability of objects to take on multiple forms. It allows the same interface to be used 
            for different underlying data types.
          </p>
          <p>
            In practice, polymorphism means you can call the same method on different objects, and each 
            object responds in its own way based on its class implementation.
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
                Think of a universal remote control with a "power" button. When you press it, 
                the <strong className="text-foreground">TV</strong> turns on/off, the 
                <strong className="text-foreground"> speaker</strong> turns on/off, and the 
                <strong className="text-foreground"> fan</strong> turns on/off. Same action (press power), 
                different behaviors based on the device.
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
        <PolymorphismVisualizer />
      </section>

      {/* Types of Polymorphism */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Types of Polymorphism
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-4 bg-card rounded-lg border-2 border-primary/30">
            <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
              <span className="px-2 py-0.5 bg-primary/20 text-primary text-xs rounded">Runtime</span>
              Method Overriding
            </h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Also called dynamic polymorphism</li>
              <li>• Child class redefines parent's method</li>
              <li>• Resolved at runtime based on object type</li>
              <li>• Requires inheritance</li>
            </ul>
            <CodeBlock
              language="python"
              className="mt-3"
              code={`class Animal:
    def speak(self): pass

class Dog(Animal):
    def speak(self):  # Override
        return "Woof!"`}
            />
          </div>
          <div className="p-4 bg-card rounded-lg border-2 border-orange-500/30">
            <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
              <span className="px-2 py-0.5 bg-orange-500/20 text-orange-500 text-xs rounded">Compile-time</span>
              Method Overloading
            </h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Also called static polymorphism</li>
              <li>• Same method name, different parameters</li>
              <li>• Resolved at compile time</li>
              <li>• Available in Java, C++ (not Python)</li>
            </ul>
            <CodeBlock
              language="java"
              className="mt-3"
              code={`class Calc {
    int add(int a, int b)
    double add(double a, double b)
    int add(int a, int b, int c)
}`}
            />
          </div>
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

      {/* Duck Typing */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Duck Typing (Python/JS)</h2>
        <div className="p-4 bg-muted/50 rounded-lg mb-4">
          <p className="text-muted-foreground italic">
            "If it walks like a duck and quacks like a duck, it's a duck."
          </p>
        </div>
        <p className="text-muted-foreground mb-4">
          In dynamically-typed languages like Python and JavaScript, polymorphism doesn't require 
          inheritance. Any object with the required method can be used:
        </p>
        <CodeBlock
          language="python"
          code={`# No inheritance needed!
class Dog:
    def speak(self):
        return "Woof!"

class Robot:  # Not related to Dog at all
    def speak(self):
        return "Beep boop!"

def make_noise(thing):  # Works with any object that has speak()
    print(thing.speak())

make_noise(Dog())    # Woof!
make_noise(Robot())  # Beep boop!`}
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
              Confusing overloading with overriding
            </h4>
            <p className="text-sm text-muted-foreground">
              Overloading = same name, different parameters (compile-time). 
              Overriding = same signature, different class (runtime).
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Missing virtual keyword in C++
            </h4>
            <p className="text-sm text-muted-foreground">
              In C++, you must mark methods as <code className="bg-muted px-1 rounded">virtual</code> 
              for runtime polymorphism to work. Without it, the base class method is always called.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Not using @Override annotation in Java
            </h4>
            <p className="text-sm text-muted-foreground">
              While optional, @Override helps catch typos. If you misspell the method name, 
              the compiler will warn you instead of creating a new method.
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
                "Explain polymorphism with a real example."
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-success font-bold">A:</span>
              <span className="text-foreground">
                "Consider a payment system with CreditCard, PayPal, and Bitcoin classes, all implementing 
                a processPayment() method. The checkout code can call processPayment() on any payment method 
                without knowing the specific type - each handles payment its own way."
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
        <Quiz questions={quizQuestions} title="Polymorphism Quiz" />
      </section>
    </TopicContent>
  );
}
