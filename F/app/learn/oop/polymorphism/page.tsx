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
  FileQuestion,
  GitBranch,
  Layers,
  Zap,
  ArrowUp,
  ArrowDown,
  Hash,
  Cpu,
  Link,
  Network,
  Type,
  Coffee,
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
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2

# Polymorphism in action
shapes = [Rectangle(4, 5), Circle(3)]
for shape in shapes:
    print(shape.describe())

# Duck typing (Python-specific)
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

def make_speak(animal):
    print(animal.speak())

make_speak(Dog())   # Woof!
make_speak(Cat())   # Meow!`;

const javascriptCode = `// Polymorphism in JavaScript

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
    
    area() {
        return this.width * this.height;
    }
}

class Circle extends Shape {
    constructor(radius) {
        super();
        this.radius = radius;
    }
    
    area() {
        return Math.PI * this.radius ** 2;
    }
}

const shapes = [new Rectangle(4, 5), new Circle(3)];
shapes.forEach(shape => {
    console.log(shape.describe());
});`;

const javaCode = `// Polymorphism in Java

abstract class Shape {
    abstract double area();
    
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
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a + b + c; }
}

public class Main {
    public static void main(String[] args) {
        Shape[] shapes = { new Rectangle(4, 5), new Circle(3) };
        for (Shape shape : shapes) {
            System.out.println(shape.describe());
        }
        
        Calculator calc = new Calculator();
        System.out.println(calc.add(1, 2));
        System.out.println(calc.add(1.5, 2.5));
    }
}`;

const cppCode = `// Polymorphism in C++
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

class Shape {
public:
    virtual double area() = 0;
    
    string describe() {
        return "I am a shape with area " + to_string(area());
    }
    
    virtual ~Shape() {}
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() override { return width * height; }
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area() override { return M_PI * radius * radius; }
};

// Operator Overloading
class Complex {
    double real, imag;
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    Complex operator+(const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }
    
    void display() { cout << real << " + " << imag << "i" << endl; }
};

int main() {
    vector<Shape*> shapes = { new Rectangle(4, 5), new Circle(3) };
    for (Shape* s : shapes) cout << s->describe() << endl;
    
    Complex c1(3, 4), c2(1, 2);
    Complex c3 = c1 + c2;
    c3.display();
    
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
    explanation: "Polymorphism (meaning 'many forms') allows objects of different classes to be treated through the same interface, with each providing its own implementation.",
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
    explanation: "Method overloading has the same method name but different parameters (compile-time). Method overriding redefines a parent's method in the child class with the same signature (runtime).",
  },
  {
    id: 3,
    question: "What is a pure virtual function in C++?",
    options: [
      "A function that cannot be overridden",
      "A function declared with '= 0' that makes a class abstract",
      "A function that is automatically inlined",
      "A function that has no return type",
    ],
    correctAnswer: 1,
    explanation: "A pure virtual function is declared as 'virtual void func() = 0;' and makes the class abstract, meaning it cannot be instantiated directly.",
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
                  <td key={cellIdx} className="border border-border p-2 text-muted-foreground">
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
              Method Overriding / Virtual Functions
            </h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Also called dynamic polymorphism or late binding</li>
              <li>• Child class redefines parent's method</li>
              <li>• Resolved at runtime based on object type</li>
              <li>• Requires inheritance and virtual keyword (C++)</li>
            </ul>
          </div>
          <div className="p-4 bg-card rounded-lg border-2 border-orange-500/30">
            <h3 className="font-semibold text-foreground mb-2 flex items-center gap-2">
              <span className="px-2 py-0.5 bg-orange-500/20 text-orange-500 text-xs rounded">Compile-time</span>
              Method Overloading / Operator Overloading
            </h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Also called static polymorphism or early binding</li>
              <li>• Same method name, different parameters</li>
              <li>• Resolved at compile time</li>
              <li>• Available in Java, C++ (limited in Python)</li>
            </ul>
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

      {/* Practice Questions Section - 5 Mark Questions */}
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-primary/10">
            <Target className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            Practice Questions (5 Marks Each)
          </h2>
        </div>
        <p className="text-muted-foreground mb-6">
          These questions are commonly asked in university examinations and technical interviews. 
          Each question carries 5 marks and includes a complete answer with code example.
        </p>

        {/* Question 22: What is polymorphism? Compile-time vs Run-time */}
        <QuestionCard
          number={22}
          title="Polymorphism - Compile-time vs Run-time"
          question="What is polymorphism? Explain compile-time and run-time polymorphism with examples."
          answer="Polymorphism means 'many forms' - the ability of objects to respond differently to the same method call. Compile-time polymorphism (static/early binding) is resolved during compilation, achieved through method overloading and operator overloading. Run-time polymorphism (dynamic/late binding) is resolved during execution, achieved through method overriding and virtual functions. Compile-time is faster but less flexible; run-time enables more dynamic behavior but has slight overhead."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Compile-time vs Run-time Polymorphism"
            headers={["Feature", "Compile-time", "Run-time"]}
            rows={[
              ["Also known as", "Static/Early binding", "Dynamic/Late binding"],
              ["Resolved at", "Compilation", "Runtime"],
              ["Achieved via", "Overloading", "Overriding, Virtual functions"],
              ["Performance", "Faster", "Slight overhead"],
              ["Flexibility", "Less flexible", "More flexible"],
              ["Method binding", "Based on reference type", "Based on object type"],
            ]}
          />
          <CodeBlock
            language="cpp"
            code={`// Compile-time Polymorphism (Overloading)
class Calculator {
public:
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
};

// Run-time Polymorphism (Overriding)
class Animal {
public:
    virtual void speak() { cout << "Animal speaks" << endl; }
};

class Dog : public Animal {
public:
    void speak() override { cout << "Dog barks" << endl; }
};

int main() {
    // Compile-time: resolved when compiling
    Calculator calc;
    cout << calc.add(5, 10);     // Calls int version
    
    // Run-time: resolved when running
    Animal* ptr = new Dog();
    ptr->speak();  // Outputs: Dog barks
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 23: Function Overloading */}
        <QuestionCard
          number={23}
          title="Function Overloading"
          question="What is function overloading? Write an example in C++/Java."
          answer="Function overloading is a compile-time polymorphism feature where multiple functions can have the same name but different parameters (different number, type, or order of parameters). The compiler determines which function to call based on the arguments passed. Return type alone cannot distinguish overloaded functions. Overloading improves code readability by using the same name for related operations."
          marks={5}
          icon={<Hash className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class Math {
public:
    // Different number of parameters
    int add(int a, int b) {
        return a + b;
    }
    
    int add(int a, int b, int c) {
        return a + b + c;
    }
    
    // Different types of parameters
    double add(double a, double b) {
        return a + b;
    }
    
    // Different order of parameters
    string add(string a, int b) {
        return a + to_string(b);
    }
    
    string add(int a, string b) {
        return to_string(a) + b;
    }
};

int main() {
    Math m;
    cout << m.add(5, 10) << endl;        // 15
    cout << m.add(5, 10, 15) << endl;    // 30
    cout << m.add(5.5, 10.5) << endl;    // 16.0
    cout << m.add("Value: ", 42) << endl; // Value: 42
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 24: Operator Overloading */}
        <QuestionCard
          number={24}
          title="Operator Overloading"
          question="What is operator overloading? Write an example in C++. Does Java support operator overloading?"
          answer="Operator overloading allows customizing the behavior of operators (+, -, *, etc.) for user-defined types. C++ supports operator overloading using the 'operator' keyword. Java does NOT support operator overloading (except for String concatenation '+') to keep the language simple and prevent abuse. This design choice avoids ambiguity and maintains clarity. C++ allows overloading most operators except ::, .*, ., ?:."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Java vs C++ on Operator Overloading:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li><strong>C++:</strong> Supports operator overloading (except ::, .*, ., ?:)</li>
              <li><strong>Java:</strong> Does NOT support operator overloading (only + for String)</li>
              <li><strong>Reason:</strong> Java designers prioritized simplicity and clarity</li>
            </ul>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class Complex {
private:
    double real, imag;
public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // Overload + operator
    Complex operator+(const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }
    
    // Overload - operator
    Complex operator-(const Complex& other) {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // Overload * operator
    Complex operator*(const Complex& other) {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }
    
    // Overload == operator
    bool operator==(const Complex& other) {
        return (real == other.real && imag == other.imag);
    }
    
    // Overload << operator for output
    friend ostream& operator<<(ostream& out, const Complex& c) {
        out << c.real << " + " << c.imag << "i";
        return out;
    }
};

int main() {
    Complex c1(3, 4), c2(1, 2);
    Complex c3 = c1 + c2;  // Uses overloaded +
    Complex c4 = c1 - c2;  // Uses overloaded -
    Complex c5 = c1 * c2;  // Uses overloaded *
    
    cout << "c1 + c2 = " << c3 << endl;
    cout << "c1 - c2 = " << c4 << endl;
    cout << "c1 * c2 = " << c5 << endl;
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 25: Virtual Function in C++ */}
        <QuestionCard
          number={25}
          title="Virtual Function in C++"
          question="What is a virtual function in C++? Why is it needed?"
          answer="A virtual function is a member function declared with the 'virtual' keyword that can be overridden in derived classes. It enables runtime polymorphism (dynamic binding) - the correct function is determined at runtime based on the actual object type, not the pointer/reference type. Without virtual, C++ uses early binding (compile-time), which would always call the base class version even when pointing to a derived object. Virtual functions are essential for achieving polymorphic behavior in C++."
          marks={5}
          icon={<Cpu className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Without virtual (Early Binding):</h4>
            <pre className="text-xs text-muted-foreground font-mono">
{`Base* ptr = new Derived();
ptr->show();  // Calls Base::show() (if not virtual)`}
            </pre>
            <h4 className="font-medium text-foreground mb-2 mt-3">With virtual (Late Binding):</h4>
            <pre className="text-xs text-muted-foreground font-mono">
{`Base* ptr = new Derived();
ptr->show();  // Calls Derived::show() (with virtual)`}
            </pre>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class Base {
public:
    // Without virtual - compile-time binding
    void nonVirtual() {
        cout << "Base::nonVirtual()" << endl;
    }
    
    // With virtual - runtime binding
    virtual void virtualFunc() {
        cout << "Base::virtualFunc()" << endl;
    }
};

class Derived : public Base {
public:
    void nonVirtual() {
        cout << "Derived::nonVirtual()" << endl;
    }
    
    void virtualFunc() override {
        cout << "Derived::virtualFunc()" << endl;
    }
};

int main() {
    Base* ptr = new Derived();
    
    // Without virtual - calls Base version (compile-time decision)
    ptr->nonVirtual();    // Output: Base::nonVirtual()
    
    // With virtual - calls Derived version (runtime decision)
    ptr->virtualFunc();   // Output: Derived::virtualFunc()
    
    // Virtual functions enable polymorphism
    Base* arr[] = {new Base(), new Derived()};
    for(int i = 0; i < 2; i++) {
        arr[i]->virtualFunc();  // Calls appropriate version
    }
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 26: Pure Virtual Function & Abstract Class */}
        <QuestionCard
          number={26}
          title="Pure Virtual Function & Abstract Class"
          question="What is a pure virtual function? What is an abstract class?"
          answer="A pure virtual function is a virtual function declared with '= 0' (e.g., virtual void func() = 0;). It has no implementation in the base class and must be overridden in derived classes. An abstract class is a class that contains at least one pure virtual function. Abstract classes cannot be instantiated directly - they serve as interfaces/blueprints for derived classes. They enforce derived classes to provide implementations for all pure virtual functions."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Key Points:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>Abstract class = has ≥ 1 pure virtual function</li>
              <li>Cannot create objects of abstract class</li>
              <li>Can have pointers/references to abstract class</li>
              <li>Derived classes MUST override all pure virtual functions</li>
              <li>Used to define interfaces (similar to Java interfaces)</li>
            </ul>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

// Abstract class (contains pure virtual function)
class Shape {
public:
    // Pure virtual function
    virtual double area() = 0;
    
    // Concrete method (optional)
    void display() {
        cout << "Area: " << area() << endl;
    }
    
    virtual ~Shape() {}
};

// Derived class - MUST implement area()
class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double area() override {
        return width * height;
    }
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    
    double area() override {
        return 3.14159 * radius * radius;
    }
};

int main() {
    // Shape s;  // ERROR! Cannot instantiate abstract class
    
    Shape* shapes[] = {new Rectangle(4, 5), new Circle(3)};
    
    for(Shape* s : shapes) {
        s->display();  // Polymorphic behavior
    }
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 27: Dynamic Binding vs Static Binding */}
        <QuestionCard
          number={27}
          title="Dynamic vs Static Binding"
          question="What is dynamic binding vs static binding? Explain with examples."
          answer="Static binding (early binding) resolves method calls at compile time based on the reference type. It's used for overloaded methods, private methods, static methods, and final methods. Dynamic binding (late binding) resolves method calls at runtime based on the actual object type. It's used for overridden methods (virtual functions in C++). Dynamic binding enables polymorphism but has slight performance overhead due to vtable lookup."
          marks={5}
          icon={<Network className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Static vs Dynamic Binding"
            headers={["Feature", "Static Binding", "Dynamic Binding"]}
            rows={[
              ["Also called", "Early binding", "Late binding"],
              ["Resolution time", "Compile time", "Runtime"],
              ["Based on", "Reference type", "Actual object type"],
              ["Used for", "Overloading, static, private", "Overriding, virtual"],
              ["Performance", "Faster", "Slight overhead"],
              ["Flexibility", "Less flexible", "More flexible"],
            ]}
          />
          <CodeBlock
            language="java"
            code={`public class BindingDemo {
    
    // Static binding - determined at compile time
    static class Parent {
        static void staticMethod() {
            System.out.println("Parent static");
        }
        
        void instanceMethod() {
            System.out.println("Parent instance");
        }
    }
    
    static class Child extends Parent {
        static void staticMethod() {
            System.out.println("Child static");
        }
        
        @Override
        void instanceMethod() {
            System.out.println("Child instance");
        }
    }
    
    public static void main(String[] args) {
        Parent ref = new Child();
        
        // Static binding - based on reference type (Parent)
        ref.staticMethod();    // Output: Parent static
        
        // Dynamic binding - based on object type (Child)
        ref.instanceMethod();  // Output: Child instance
        
        // Overloading example - static binding
        Calculator calc = new Calculator();
        calc.add(5, 10);       // Calls int version at compile time
        calc.add(5.5, 10.5);   // Calls double version at compile time
    }
}

class Calculator {
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
}`}
          />
        </QuestionCard>

        {/* Question 28: Late Binding Example */}
        <QuestionCard
          number={28}
          title="Late Binding Example"
          question="Explain the concept of late binding with an example."
          answer="Late binding (dynamic dispatch) is when the method to be executed is determined at runtime based on the actual object type, not the reference type. In C++, this requires virtual functions. The compiler creates a virtual table (vtable) containing function pointers. When a virtual function is called, the program looks up the correct function from the vtable at runtime. This enables polymorphic behavior where a base class pointer can call derived class implementations."
          marks={5}
          icon={<Type className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">How Late Binding Works:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>Compiler creates a vtable for each class with virtual functions</li>
              <li>Each object has a hidden vptr pointing to its class's vtable</li>
              <li>Virtual function call: obj-&gt;vptr-&gt;vtable[index]</li>
              <li>Resolution happens at runtime</li>
            </ul>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
#include <vector>
using namespace std;

// Base class with virtual function
class Employee {
protected:
    string name;
public:
    Employee(string n) : name(n) {}
    
    virtual double calculateSalary() {
        return 30000;  // Base salary
    }
    
    virtual void display() {
        cout << name << " earns: $" << calculateSalary() << endl;
    }
    
    virtual ~Employee() {}
};

class Manager : public Employee {
    double bonus;
public:
    Manager(string n, double b) : Employee(n), bonus(b) {}
    
    double calculateSalary() override {
        return 50000 + bonus;  // Higher base + bonus
    }
};

class Intern : public Employee {
public:
    Intern(string n) : Employee(n) {}
    
    double calculateSalary() override {
        return 15000;  // Lower salary
    }
};

int main() {
    // Late binding in action
    vector<Employee*> employees;
    employees.push_back(new Employee("John"));
    employees.push_back(new Manager("Alice", 10000));
    employees.push_back(new Intern("Bob"));
    
    // Same call, different behavior based on actual object type
    for(Employee* emp : employees) {
        emp->display();  // Late binding determines which calculateSalary() to call
    }
    
    // Output:
    // John earns: $30000
    // Alice earns: $60000
    // Bob earns: $15000
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 29: Overriding Static Methods */}
        <QuestionCard
          number={29}
          title="Static Method Overriding"
          question="Can we override static methods? Why or why not?"
          answer="No, static methods cannot be overridden. They are associated with the class, not with objects. Static methods are resolved at compile-time using static binding (early binding) based on the reference type, not runtime based on object type. When a subclass defines a static method with the same signature, it's called method hiding, not overriding. The method called depends on the reference variable type, not the actual object type."
          marks={5}
          icon={<Coffee className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Key Differences: Overriding vs Hiding</h4>
            <DifferenceTable
              title=""
              headers={["Feature", "Instance Method (Overriding)", "Static Method (Hiding)"]}
              rows={[
                ["Binding", "Runtime (dynamic)", "Compile-time (static)"],
                ["Resolution", "Based on object type", "Based on reference type"],
                ["Keyword", "@Override (Java)", "Method hiding (no keyword)"],
                ["Polymorphism", "Yes", "No"],
              ]}
            />
          </div>
          <CodeBlock
            language="java"
            code={`public class StaticOverrideDemo {
    
    static class Parent {
        static void staticMethod() {
            System.out.println("Parent static method");
        }
        
        void instanceMethod() {
            System.out.println("Parent instance method");
        }
    }
    
    static class Child extends Parent {
        // This is method HIDING, not overriding
        static void staticMethod() {
            System.out.println("Child static method");
        }
        
        @Override
        void instanceMethod() {
            System.out.println("Child instance method");
        }
    }
    
    public static void main(String[] args) {
        Parent ref = new Child();
        
        // Static method - based on reference type (Parent)
        ref.staticMethod();    // Output: Parent static method
        
        // Instance method - based on object type (Child)
        ref.instanceMethod();  // Output: Child instance method
        
        // To call Child's static method, use Child class directly
        Child.staticMethod();  // Output: Child static method
    }
}`}
          />
        </QuestionCard>

        {/* Summary Section */}
        <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q22:</span>
              <span className="text-muted-foreground ml-1">Compile-time (overloading) vs Run-time (overriding)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q23:</span>
              <span className="text-muted-foreground ml-1">Function overloading = same name, different parameters</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q24:</span>
              <span className="text-muted-foreground ml-1">Operator overloading: C++ ✓ | Java ✗</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q25:</span>
              <span className="text-muted-foreground ml-1">Virtual function = runtime polymorphism in C++</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q26:</span>
              <span className="text-muted-foreground ml-1">Pure virtual (=0) → Abstract class (cannot instantiate)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q27:</span>
              <span className="text-muted-foreground ml-1">Static binding (compile) vs Dynamic binding (runtime)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q28:</span>
              <span className="text-muted-foreground ml-1">Late binding = vtable lookup at runtime</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q29:</span>
              <span className="text-muted-foreground ml-1">Static methods = Hiding, not Overriding</span>
            </div>
          </div>
        </div>

        {/* Marks Distribution */}
        <div className="mt-6 flex justify-between items-center text-sm text-muted-foreground border-t border-border pt-4">
          <span>Total Questions: 8</span>
          <span>Marks per Question: 5</span>
          <span>Total Marks: 40</span>
          <span>Time Suggested: 2 hours</span>
        </div>
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
              Forgetting virtual keyword in C++
            </h4>
            <p className="text-sm text-muted-foreground">
              Without 'virtual', C++ uses early binding - base class method always called regardless of object type.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Confusing overloading with overriding
            </h4>
            <p className="text-sm text-muted-foreground">
              Overloading = same name, different parameters (compile-time). Overriding = same signature, different class (runtime).
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Trying to override static methods
            </h4>
            <p className="text-sm text-muted-foreground">
              Static methods are bound at compile-time based on reference type - they can't be overridden, only hidden.
            </p>
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