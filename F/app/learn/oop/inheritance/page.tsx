"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { InheritanceVisualizer } from "@/components/visualizations/oop-visualizer";
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
  Diamond,
  Lock,
  Shield,
  Layers,
  Zap,
  ArrowUp,
  ArrowDown,
} from "lucide-react";

const pythonCode = `# Inheritance in Python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def eat(self):
        return f"{self.name} is eating"
    
    def sleep(self):
        return f"{self.name} is sleeping"

# Dog inherits from Animal
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)  # Call parent constructor
        self.breed = breed
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def fetch(self):
        return f"{self.name} is fetching the ball"

# Cat inherits from Animal
class Cat(Animal):
    def __init__(self, name, age, indoor=True):
        super().__init__(name, age)
        self.indoor = indoor
    
    def meow(self):
        return f"{self.name} says Meow!"

# Using inheritance
dog = Dog("Buddy", 3, "Golden Retriever")
print(dog.eat())    # Inherited: Buddy is eating
print(dog.bark())   # Own method: Buddy says Woof!
print(dog.breed)    # Own attribute: Golden Retriever

cat = Cat("Whiskers", 2)
print(cat.sleep())  # Inherited: Whiskers is sleeping
print(cat.meow())   # Own method: Whiskers says Meow!`;

const javascriptCode = `// Inheritance in JavaScript
class Animal {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    eat() {
        return \`\${this.name} is eating\`;
    }
    
    sleep() {
        return \`\${this.name} is sleeping\`;
    }
}

// Dog inherits from Animal
class Dog extends Animal {
    constructor(name, age, breed) {
        super(name, age);  // Call parent constructor
        this.breed = breed;
    }
    
    bark() {
        return \`\${this.name} says Woof!\`;
    }
    
    fetch() {
        return \`\${this.name} is fetching the ball\`;
    }
}

// Cat inherits from Animal
class Cat extends Animal {
    constructor(name, age, indoor = true) {
        super(name, age);
        this.indoor = indoor;
    }
    
    meow() {
        return \`\${this.name} says Meow!\`;
    }
}

// Using inheritance
const dog = new Dog("Buddy", 3, "Golden Retriever");
console.log(dog.eat());    // Inherited
console.log(dog.bark());   // Own method

const cat = new Cat("Whiskers", 2);
console.log(cat.sleep());  // Inherited
console.log(cat.meow());   // Own method`;

const javaCode = `// Inheritance in Java
class Animal {
    protected String name;
    protected int age;
    
    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public String eat() {
        return name + " is eating";
    }
    
    public String sleep() {
        return name + " is sleeping";
    }
}

// Dog inherits from Animal
class Dog extends Animal {
    private String breed;
    
    public Dog(String name, int age, String breed) {
        super(name, age);  // Call parent constructor
        this.breed = breed;
    }
    
    public String bark() {
        return name + " says Woof!";
    }
    
    public String getBreed() {
        return breed;
    }
}

// Cat inherits from Animal
class Cat extends Animal {
    private boolean indoor;
    
    public Cat(String name, int age, boolean indoor) {
        super(name, age);
        this.indoor = indoor;
    }
    
    public String meow() {
        return name + " says Meow!";
    }
}

// Usage
public class Main {
    public static void main(String[] args) {
        Dog dog = new Dog("Buddy", 3, "Golden Retriever");
        System.out.println(dog.eat());   // Inherited
        System.out.println(dog.bark());  // Own method
        
        Cat cat = new Cat("Whiskers", 2, true);
        System.out.println(cat.sleep()); // Inherited
        System.out.println(cat.meow());  // Own method
    }
}`;

const cppCode = `// Inheritance in C++
#include <iostream>
#include <string>
using namespace std;

class Animal {
protected:
    string name;
    int age;

public:
    Animal(string n, int a) : name(n), age(a) {}
    
    string eat() {
        return name + " is eating";
    }
    
    string sleep() {
        return name + " is sleeping";
    }
};

// Dog inherits from Animal
class Dog : public Animal {
private:
    string breed;

public:
    Dog(string n, int a, string b) : Animal(n, a), breed(b) {}
    
    string bark() {
        return name + " says Woof!";
    }
    
    string getBreed() {
        return breed;
    }
};

// Cat inherits from Animal
class Cat : public Animal {
private:
    bool indoor;

public:
    Cat(string n, int a, bool i = true) : Animal(n, a), indoor(i) {}
    
    string meow() {
        return name + " says Meow!";
    }
};

int main() {
    Dog dog("Buddy", 3, "Golden Retriever");
    cout << dog.eat() << endl;   // Inherited
    cout << dog.bark() << endl;  // Own method
    
    Cat cat("Whiskers", 2, true);
    cout << cat.sleep() << endl; // Inherited
    cout << cat.meow() << endl;  // Own method
    
    return 0;
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is inheritance in OOP?",
    options: [
      "Creating multiple objects from a class",
      "A mechanism where a class acquires properties and methods from another class",
      "Hiding implementation details",
      "Having multiple methods with the same name",
    ],
    correctAnswer: 1,
    explanation:
      "Inheritance is a mechanism that allows a class (child/derived) to inherit attributes and methods from another class (parent/base), promoting code reuse.",
  },
  {
    id: 2,
    question: "What does 'super()' or 'super' keyword do?",
    options: [
      "Creates a new object",
      "Deletes the parent class",
      "Calls the parent class constructor or methods",
      "Makes a method private",
    ],
    correctAnswer: 2,
    explanation:
      "'super()' is used to call the parent class's constructor or methods. It's essential for initializing inherited attributes in the child class.",
  },
  {
    id: 3,
    question: "If class B inherits from class A, what is B called?",
    options: [
      "Parent class",
      "Base class",
      "Child/Derived class",
      "Abstract class",
    ],
    correctAnswer: 2,
    explanation:
      "Class B is called the child class, derived class, or subclass. Class A is the parent class, base class, or superclass.",
  },
  {
    id: 4,
    question: "Can a child class access private members of the parent class?",
    options: [
      "Yes, always",
      "No, private members are not accessible to child classes",
      "Only if using super()",
      "Only in Python",
    ],
    correctAnswer: 1,
    explanation:
      "Private members are not directly accessible to child classes. Use protected members (accessible to children) or public getters/setters to access parent data.",
  },
  {
    id: 5,
    question: "What is method overriding?",
    options: [
      "Adding new methods to a class",
      "A child class providing its own implementation of a parent's method",
      "Having multiple methods with the same name but different parameters",
      "Hiding a method from other classes",
    ],
    correctAnswer: 1,
    explanation:
      "Method overriding is when a child class provides its own implementation of a method that is already defined in its parent class, replacing the inherited behavior.",
  },
  {
    id: 6,
    question: "What type of inheritance does Python support?",
    options: [
      "Single inheritance only",
      "Multiple inheritance",
      "No inheritance",
      "Only interface inheritance",
    ],
    correctAnswer: 1,
    explanation:
      "Python supports multiple inheritance, meaning a class can inherit from multiple parent classes. Java only supports single inheritance for classes (but multiple for interfaces).",
  },
];

// Additional code examples for practice questions

const inheritanceTypesCode = `// C++ - Types of Inheritance
#include <iostream>
#include <string>
using namespace std;

// 1. SINGLE INHERITANCE
class Animal {
protected:
    string name;
public:
    Animal(string n) : name(n) {}
    void eat() { cout << name << " is eating" << endl; }
};

class Dog : public Animal {
public:
    Dog(string n) : Animal(n) {}
    void bark() { cout << name << " is barking" << endl; }
};

// 2. MULTILEVEL INHERITANCE
class Mammal : public Animal {
public:
    Mammal(string n) : Animal(n) {}
    void breathe() { cout << name << " is breathing" << endl; }
};

class Cat : public Mammal {
public:
    Cat(string n) : Mammal(n) {}
    void meow() { cout << name << " says meow" << endl; }
};

// 3. MULTIPLE INHERITANCE
class Flying {
public:
    void fly() { cout << "Flying in the sky" << endl; }
};

class Swimming {
public:
    void swim() { cout << "Swimming in water" << endl; }
};

class Duck : public Flying, public Swimming {
public:
    void quack() { cout << "Duck says quack" << endl; }
};

// 4. HIERARCHICAL INHERITANCE
class Shape {
protected:
    string color;
public:
    Shape(string c) : color(c) {}
    virtual void draw() = 0;
};

class Circle : public Shape {
public:
    Circle(string c) : Shape(c) {}
    void draw() override { cout << "Drawing " << color << " circle" << endl; }
};

class Rectangle : public Shape {
public:
    Rectangle(string c) : Shape(c) {}
    void draw() override { cout << "Drawing " << color << " rectangle" << endl; }
};

int main() {
    cout << "=== SINGLE INHERITANCE ===" << endl;
    Dog dog("Buddy");
    dog.eat();
    dog.bark();
    
    cout << "\\n=== MULTILEVEL INHERITANCE ===" << endl;
    Cat cat("Whiskers");
    cat.eat();
    cat.breathe();
    cat.meow();
    
    cout << "\\n=== MULTIPLE INHERITANCE ===" << endl;
    Duck duck;
    duck.fly();
    duck.swim();
    duck.quack();
    
    cout << "\\n=== HIERARCHICAL INHERITANCE ===" << endl;
    Circle circle("Red");
    Rectangle rect("Blue");
    circle.draw();
    rect.draw();
    
    return 0;
}`;

const javaMultipleInheritanceCode = `// WHY JAVA DOESN'T SUPPORT MULTIPLE INHERITANCE

// DIAMOND PROBLEM DEMONSTRATION
/* 
class A {
    void display() { System.out.println("A's display"); }
}

class B extends A {
    void display() { System.out.println("B's display"); }
}

class C extends A {
    void display() { System.out.println("C's display"); }
}

// DIAMOND PROBLEM - Which display() would D inherit?
class D extends B, C {  // NOT ALLOWED IN JAVA
    // Ambiguity: B's display() or C's display()?
}
*/

// Java's Solution: Single Inheritance + Multiple Interface Implementation
interface Drawable {
    void draw();
}

interface Colorable {
    void setColor(String color);
}

class Circle implements Drawable, Colorable {
    private String color;
    
    @Override
    public void draw() {
        System.out.println("Drawing circle");
    }
    
    @Override
    public void setColor(String color) {
        this.color = color;
        System.out.println("Circle color set to: " + color);
    }
}

interface Vehicle {
    default void start() {
        System.out.println("Vehicle starting");
    }
}

interface Electric {
    default void start() {
        System.out.println("Electric vehicle starting silently");
    }
}

class ElectricCar implements Vehicle, Electric {
    @Override
    public void start() {
        Vehicle.super.start();
        Electric.super.start();
        System.out.println("ElectricCar started");
    }
}

public class MultipleInheritanceDemo {
    public static void main(String[] args) {
        System.out.println("=== WHY JAVA DOESN'T SUPPORT MULTIPLE INHERITANCE ===");
        System.out.println("1. DIAMOND PROBLEM - Ambiguity in method inheritance");
        System.out.println("2. Complexity - Makes code harder to understand");
        
        Circle circle = new Circle();
        circle.draw();
        circle.setColor("Red");
        
        ElectricCar eCar = new ElectricCar();
        eCar.start();
    }
}`;

const accessSpecifiersCode = `// C++ - Inheritance Access Specifiers
#include <iostream>
using namespace std;

class Base {
private:
    int privateVar = 1;
protected:
    int protectedVar = 2;
public:
    int publicVar = 3;
};

// 1. PUBLIC INHERITANCE
class PublicDerived : public Base {
public:
    void show() {
        // cout << privateVar;  // ERROR! Private not accessible
        cout << "Protected: " << protectedVar << endl;
        cout << "Public: " << publicVar << endl;
    }
};

// 2. PROTECTED INHERITANCE
class ProtectedDerived : protected Base {
public:
    void show() {
        cout << "Protected: " << protectedVar << endl;
        cout << "Public (now protected): " << publicVar << endl;
    }
};

// 3. PRIVATE INHERITANCE
class PrivateDerived : private Base {
public:
    void show() {
        cout << "Protected (now private): " << protectedVar << endl;
        cout << "Public (now private): " << publicVar << endl;
    }
};

int main() {
    cout << "=== PUBLIC INHERITANCE ===" << endl;
    PublicDerived pub;
    pub.show();
    pub.publicVar = 100;  // Accessible
    
    cout << "\\n=== PROTECTED INHERITANCE ===" << endl;
    ProtectedDerived prot;
    prot.show();
    // prot.publicVar;  // NOT accessible
    
    cout << "\\n=== PRIVATE INHERITANCE ===" << endl;
    PrivateDerived priv;
    priv.show();
    
    return 0;
}`;

const multilevelInheritanceCode = `// C++ - Multilevel Inheritance Program
#include <iostream>
#include <string>
using namespace std;

// LEVEL 1: Base class
class Student {
protected:
    string name;
    int rollNumber;
    string course;
    
public:
    Student(string n, int r, string c) : name(n), rollNumber(r), course(c) {
        cout << "Student constructor called for: " << name << endl;
    }
    
    void displayBasicInfo() {
        cout << "\\n=== STUDENT INFORMATION ===" << endl;
        cout << "Name: " << name << endl;
        cout << "Roll Number: " << rollNumber << endl;
        cout << "Course: " << course << endl;
    }
};

// LEVEL 2: Intermediate class
class Exam : public Student {
protected:
    int marksTheory;
    int marksPractical;
    float internalAssessment;
    
public:
    Exam(string n, int r, string c, int theory, int practical, float internal)
        : Student(n, r, c), marksTheory(theory), marksPractical(practical), internalAssessment(internal) {
        cout << "Exam constructor called for: " << name << endl;
    }
    
    float calculateTotal() {
        return marksTheory + marksPractical + internalAssessment;
    }
};

// LEVEL 3: Derived class
class Result : public Exam {
private:
    float totalMarks;
    char grade;
    string status;
    
public:
    Result(string n, int r, string c, int theory, int practical, float internal)
        : Exam(n, r, c, theory, practical, internal), totalMarks(0), grade('F'), status("Fail") {
        cout << "Result constructor called for: " << name << endl;
        calculateResult();
    }
    
    void calculateResult() {
        totalMarks = calculateTotal();
        float percentage = (totalMarks / 200) * 100;
        
        if (percentage >= 90) grade = 'A';
        else if (percentage >= 75) grade = 'B';
        else if (percentage >= 60) grade = 'C';
        else if (percentage >= 40) grade = 'D';
        else grade = 'F';
    }
    
    void displayResult() {
        displayBasicInfo();
        cout << "\\n=== FINAL RESULT ===" << endl;
        cout << "Total Marks: " << totalMarks << "/200" << endl;
        cout << "Grade: " << grade << endl;
    }
};

int main() {
    cout << "=== MULTILEVEL INHERITANCE: Student → Exam → Result ===" << endl;
    Result student("Alice Johnson", 2024001, "CS", 85, 45, 42);
    student.displayResult();
    return 0;
}`;

const methodOverridingCode = `// C++ - Method Overriding vs Overloading
#include <iostream>
#include <string>
using namespace std;

class Animal {
public:
    virtual void makeSound() {
        cout << "Animal makes a generic sound" << endl;
    }
    
    void eat() { cout << "Animal is eating" << endl; }
    void eat(string food) { cout << "Animal is eating " << food << endl; }
};

class Dog : public Animal {
public:
    void makeSound() override {
        cout << "Dog barks: Woof! Woof!" << endl;
    }
    
    void eat() { cout << "Dog is eating dog food" << endl; }
};

class Calculator {
public:
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a + b + c; }
};

int main() {
    cout << "=== METHOD OVERRIDING ===" << endl;
    Animal* ptr;
    Dog dog;
    ptr = &dog;
    ptr->makeSound();  // Calls Dog's version
    
    cout << "\\n=== METHOD OVERLOADING ===" << endl;
    Calculator calc;
    cout << calc.add(5, 10) << endl;
    cout << calc.add(5.5, 10.5) << endl;
    cout << calc.add(5, 10, 15) << endl;
    
    return 0;
}`;

const superVsBaseCode = `// C++ - Accessing Base Class Members (No 'base' keyword)
#include <iostream>
#include <string>
using namespace std;

class Base {
protected:
    string name;
    int value;
    
public:
    Base() : name("Base"), value(0) {}
    Base(string n, int v) : name(n), value(v) {}
    
    void display() {
        cout << "Base::display() - Name: " << name << ", Value: " << value << endl;
    }
    
    virtual void show() { cout << "Base::show()" << endl; }
};

class Derived : public Base {
private:
    string extra;
    
public:
    // C++: Call base constructor in initializer list
    Derived(string n, int v, string e) : Base(n, v), extra(e) {}
    
    void display() {
        Base::display();  // Call base class version
        cout << "Derived::display() - Extra: " << extra << endl;
    }
    
    void show() override {
        cout << "Derived::show()" << endl;
        Base::show();  // Call base class version
    }
    
    void accessBaseMembers() {
        cout << "Accessing base name: " << name << endl;
        cout << "Accessing base value: " << value << endl;
    }
};

int main() {
    cout << "=== C++ HAS NO 'base' KEYWORD ===" << endl;
    cout << "C++ uses: BaseClass::member or initializer list : Base(args)\\n" << endl;
    
    Derived d("DerivedObj", 100, "ExtraData");
    d.display();
    d.show();
    d.accessBaseMembers();
    
    cout << "\\n=== COMPARISON ===" << endl;
    cout << "Java 'super()' → C++ : Base(args)" << endl;
    cout << "Java 'super.method()' → C++ : Base::method()" << endl;
    
    return 0;
}`;

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
    </div>
  );
};

export default function InheritancePage() {
  const result = getSubtopicBySlug("oop", "inheritance");

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
            What is Inheritance?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Inheritance</strong> is one of the four pillars of OOP. 
            It allows a class to inherit attributes and methods from another class, promoting code reuse 
            and establishing a hierarchical relationship between classes.
          </p>
          <p>
            The class that inherits is called the <strong className="text-foreground">child class</strong> 
            (or derived/subclass), and the class being inherited from is the 
            <strong className="text-foreground"> parent class</strong> (or base/superclass).
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
                Think of biological inheritance: children inherit traits from their parents (eye color, 
                hair type) but also develop their own unique characteristics. Similarly, a 
                <strong className="text-foreground"> Dog</strong> class inherits from 
                <strong className="text-foreground"> Animal</strong> (can eat, sleep) but adds its own 
                abilities (bark, fetch).
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
        <InheritanceVisualizer />
      </section>

      {/* Types of Inheritance */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Types of Inheritance
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Single Inheritance</h3>
            <p className="text-sm text-muted-foreground mb-2">
              A class inherits from one parent class only.
            </p>
            <code className="text-xs bg-muted px-2 py-1 rounded">Dog → Animal</code>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Multiple Inheritance</h3>
            <p className="text-sm text-muted-foreground mb-2">
              A class inherits from multiple parent classes. (Python, C++)
            </p>
            <code className="text-xs bg-muted px-2 py-1 rounded">FlyingFish → Fish, Bird</code>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Multilevel Inheritance</h3>
            <p className="text-sm text-muted-foreground mb-2">
              A chain of inheritance (grandparent → parent → child).
            </p>
            <code className="text-xs bg-muted px-2 py-1 rounded">Puppy → Dog → Animal</code>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Hierarchical Inheritance</h3>
            <p className="text-sm text-muted-foreground mb-2">
              Multiple classes inherit from a single parent.
            </p>
            <code className="text-xs bg-muted px-2 py-1 rounded">Dog, Cat, Bird → Animal</code>
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

      {/* Method Overriding */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Method Overriding</h2>
        <p className="text-muted-foreground mb-4">
          Child classes can override parent methods to provide their own implementation:
        </p>
        <CodeBlock
          language="python"
          code={`class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):  # Override parent method
        return "Woof!"

class Cat(Animal):
    def speak(self):  # Override parent method
        return "Meow!"

# Polymorphism in action
animals = [Dog(), Cat(), Animal()]
for animal in animals:
    print(animal.speak())
# Output: Woof! Meow! Some sound`}
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
              Forgetting to call super().__init__()
            </h4>
            <p className="text-sm text-muted-foreground">
              If you define a constructor in the child class, you must call the parent's constructor 
              to initialize inherited attributes.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Deep inheritance hierarchies
            </h4>
            <p className="text-sm text-muted-foreground">
              Avoid inheritance chains more than 2-3 levels deep. It becomes hard to track where 
              methods come from. Prefer composition over deep inheritance.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Inheriting just for code reuse
            </h4>
            <p className="text-sm text-muted-foreground">
              Use inheritance for "is-a" relationships (Dog is an Animal), not just to reuse code. 
              For "has-a" relationships, use composition instead.
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
                "When would you use inheritance vs composition?"
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-success font-bold">A:</span>
              <span className="text-foreground">
                "Use inheritance for 'is-a' relationships (Car is a Vehicle). Use composition for 
                'has-a' relationships (Car has an Engine). Composition is more flexible and preferred 
                when the relationship isn't clearly hierarchical."
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
        <Quiz questions={quizQuestions} title="Inheritance Quiz" />
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

        {/* Question 14: Inheritance & Types */}
        <QuestionCard
          number={14}
          title="Inheritance & Its Types"
          question="What is inheritance? Explain different types of inheritance with examples."
          answer="Inheritance is an OOP mechanism where a class (derived class) acquires properties and behaviors from another class (base class). Types include: (1) Single - one base, one derived, (2) Multilevel - chain of inheritance, (3) Multiple - derived from multiple bases (C++ only), (4) Hierarchical - multiple derived from one base, (5) Hybrid - combination of types. Inheritance promotes code reuse, establishes relationships, and enables polymorphism."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Types of Inheritance - Comparison"
            headers={["Type", "Diagram", "# of Base Classes", "# of Derived Classes", "Supported in Java"]}
            rows={[
              ["Single", "A → B", "1", "1", "✓"],
              ["Multilevel", "A → B → C", "1 (chain)", "1", "✓"],
              ["Multiple", "A,B → C", "2+", "1", "✗ (use interfaces)"],
              ["Hierarchical", "A → B,C,D", "1", "Multiple", "✓"],
              ["Hybrid", "Combination", "Multiple", "Multiple", "✗ (use interfaces)"],
            ]}
          />
          <MultiLanguageCode
            codes={[
              { language: "cpp", label: "C++", code: inheritanceTypesCode },
              { language: "java", label: "Java", code: javaMultipleInheritanceCode },
            ]}
          />
        </QuestionCard>

        {/* Question 15: Java Multiple Inheritance */}
        <QuestionCard
          number={15}
          title="Java & Multiple Inheritance"
          question="Does Java support multiple inheritance? Why or why not?"
          answer="No, Java does not support multiple inheritance with classes to avoid the Diamond Problem. Diamond Problem occurs when a class inherits from two classes that have a common ancestor, causing ambiguity about which parent's method to inherit. Java's designers prioritized simplicity and clarity over flexibility. Instead, Java supports multiple inheritance through interfaces, which provide abstraction without implementation ambiguity (until default methods in Java 8+, which require explicit resolution)."
          marks={5}
          icon={<Diamond className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Diamond Problem Visualization:</h4>
            <pre className="text-xs text-muted-foreground font-mono">
{`      A
     / \\
    B   C
     \\ /
      D
      
If B and C override a method from A,
which version should D inherit?`}
            </pre>
          </div>
          <CodeBlock code={javaMultipleInheritanceCode} language="java" />
        </QuestionCard>

        {/* Question 16: Interfaces & Diamond Problem */}
        <QuestionCard
          number={16}
          title="Interfaces & The Diamond Problem"
          question="What are interfaces in Java? How do they solve the diamond problem?"
          answer="Interfaces are contracts that declare methods without implementation (before Java 8). A class can implement multiple interfaces. Interfaces solve the diamond problem because: (1) Before Java 8, interfaces had no implementation, so no ambiguity, (2) Class provides the actual implementation, (3) Java 8+ default methods require explicit resolution if conflict occurs. Interfaces provide multiple inheritance of type (not implementation), achieving polymorphism without complexity."
          marks={5}
          icon={<Zap className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Key Interface Features:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>100% abstraction (before Java 8)</li>
              <li>Multiple inheritance of TYPE</li>
              <li>Loose coupling between components</li>
              <li>Contract for implementing classes</li>
              <li>Default/static methods (Java 8+)</li>
            </ul>
          </div>
          <CodeBlock code={javaMultipleInheritanceCode} language="java" />
        </QuestionCard>

        {/* Question 17: Access Specifiers in C++ */}
        <QuestionCard
          number={17}
          title="Inheritance Access Specifiers (C++)"
          question="Explain different inheritance access specifiers: public, private, protected inheritance in C++."
          answer="C++ provides three inheritance access specifiers that determine how base class members are inherited: (1) public inheritance - public remains public, protected remains protected; (2) protected inheritance - public and protected become protected in derived; (3) private inheritance - public and protected become private in derived. Private members are never accessible directly regardless of inheritance type. The specifier controls access for further derived classes and outside code."
          marks={5}
          icon={<Lock className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Inheritance Access Specifiers Effect"
            headers={["Base Member Access", "public Inheritance", "protected Inheritance", "private Inheritance"]}
            rows={[
              ["public → becomes", "public", "protected", "private"],
              ["protected → becomes", "protected", "protected", "private"],
              ["private → becomes", "inaccessible", "inaccessible", "inaccessible"],
            ]}
          />
          <CodeBlock code={accessSpecifiersCode} language="cpp" />
        </QuestionCard>

        {/* Question 18: Access Modifiers */}
        <QuestionCard
          number={18}
          title="Public, Protected, Private Members"
          question="What is the difference between public, protected, and private members in a class?"
          answer="Access modifiers control visibility: (1) public - accessible from anywhere (inside class, derived classes, and outside code), (2) protected - accessible inside the class and derived classes, but not outside, (3) private - accessible only inside the class itself. This provides encapsulation - hiding internal details while exposing necessary interfaces. Private members offer strongest protection, protected enables inheritance while maintaining some control, public provides the interface."
          marks={5}
          icon={<Shield className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Access Modifiers Comparison"
            headers={["Access Level", "Same Class", "Derived Class", "Outside Class"]}
            rows={[
              ["public", "✓", "✓", "✓"],
              ["protected", "✓", "✓", "✗"],
              ["private", "✓", "✗", "✗"],
            ]}
          />
          <CodeBlock code={accessSpecifiersCode} language="cpp" />
        </QuestionCard>

        {/* Question 19: Multilevel Inheritance */}
        <QuestionCard
          number={19}
          title="Multilevel Inheritance Program"
          question="Write a C++ program showing multilevel inheritance."
          answer="Multilevel inheritance involves a chain of inheritance where a derived class inherits from another derived class. Example: Student → Exam → Result. Each level adds more specific functionality. Benefits include code reuse at multiple levels, logical hierarchy representation, and progressive specialization. The program demonstrates constructors calling order (base to derived), member access across levels, and destructor order (reverse)."
          marks={5}
          icon={<Layers className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Construction/Destruction Order:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li><strong>Constructor Order:</strong> Student → Exam → Result</li>
              <li><strong>Destructor Order:</strong> Result → Exam → Student (reverse)</li>
              <li>Each level initializes its own members</li>
            </ul>
          </div>
          <CodeBlock code={multilevelInheritanceCode} language="cpp" />
        </QuestionCard>

        {/* Question 20: Overriding vs Overloading */}
        <QuestionCard
          number={20}
          title="Method Overriding vs Overloading"
          question="What is method overriding? How does it differ from method overloading?"
          answer="Method Overriding is redefining a base class method in a derived class with same signature (name, parameters, return type). It enables runtime polymorphism. Method Overloading is defining multiple methods with same name but different parameters within the same class, enabling compile-time polymorphism. Key differences: Overriding requires inheritance, same parameters; Overloading within same class, different parameters. Overriding is runtime binding; Overloading is compile-time binding."
          marks={5}
          icon={<ArrowUp className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Overriding vs Overloading - Detailed Comparison"
            headers={["Feature", "Overriding", "Overloading"]}
            rows={[
              ["Relationship", "Parent-child classes", "Same class"],
              ["Parameters", "Must be identical", "Must be different"],
              ["Return type", "Same or covariant", "Can be different"],
              ["Polymorphism", "Runtime (dynamic)", "Compile-time (static)"],
              ["Keyword", "virtual, override (C++)", "None needed"],
              ["Inheritance", "Required", "Not required"],
            ]}
          />
          <CodeBlock code={methodOverridingCode} language="cpp" />
        </QuestionCard>

        {/* Question 21: super vs base */}
        <QuestionCard
          number={21}
          title="super (Java) vs Base (C++)"
          question="What is the difference between super (Java) and Base (C++) keywords?"
          answer="Java has 'super' keyword to refer to parent class; C++ does NOT have a 'base' keyword. In C++, base class members are accessed using BaseClass::member syntax. Java 'super()' calls parent constructor; C++ uses member initializer list (: BaseClass(args)). Java 'super.method()' calls overridden parent method; C++ uses BaseClass::method(). The difference reflects design philosophy: Java provides dedicated keyword for clarity, C++ uses scope resolution operator for flexibility with multiple inheritance."
          marks={5}
          icon={<ArrowDown className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Java super vs C++ Base Access"
            headers={["Operation", "Java (super)", "C++ (no base keyword)"]}
            rows={[
              ["Call parent constructor", "super(args)", ": Base(args)"],
              ["Call parent method", "super.method()", "Base::method()"],
              ["Access parent member", "super.member", "Base::member or direct"],
              ["Must be first?", "Yes (constructor)", "Yes (initializer list)"],
              ["Multiple inheritance", "N/A (not allowed)", "Can specify multiple: Base1, Base2"],
            ]}
          />
          <CodeBlock code={superVsBaseCode} language="cpp" />
        </QuestionCard>

        {/* Summary Section */}
        <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q14:</span>
              <span className="text-muted-foreground ml-1">5 inheritance types</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q15:</span>
              <span className="text-muted-foreground ml-1">Java no multiple inheritance (Diamond Problem)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q16:</span>
              <span className="text-muted-foreground ml-1">Interfaces = multiple inheritance of type</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q17:</span>
              <span className="text-muted-foreground ml-1">public/protected/private inheritance (C++)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q18:</span>
              <span className="text-muted-foreground ml-1">public (everywhere) → protected (+derived) → private (only class)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q19:</span>
              <span className="text-muted-foreground ml-1">Multilevel: Student→Exam→Result</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q20:</span>
              <span className="text-muted-foreground ml-1">Override (runtime) vs Overload (compile-time)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q21:</span>
              <span className="text-muted-foreground ml-1">Java: super | C++: Base:: or :Base()</span>
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
    </TopicContent>
  );
}