"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { ClassObjectVisualizer } from "@/components/visualizations/oop-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
} from "lucide-react";

const pythonCode = `# Defining a class in Python
class Person:
    # Constructor (initializer)
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age
    
    # Instance method
    def greet(self):
        return f"Hello, I'm {self.name}!"
    
    def celebrate_birthday(self):
        self.age += 1
        return f"{self.name} is now {self.age}!"

# Creating objects (instances)
alice = Person("Alice", 25)
bob = Person("Bob", 30)

# Using object methods
print(alice.greet())           # Hello, I'm Alice!
print(bob.celebrate_birthday()) # Bob is now 31!

# Accessing attributes
print(alice.name)  # Alice
print(bob.age)     # 31`;

const javascriptCode = `// Defining a class in JavaScript
class Person {
    // Constructor
    constructor(name, age) {
        this.name = name;  // Instance property
        this.age = age;
    }
    
    // Instance method
    greet() {
        return \`Hello, I'm \${this.name}!\`;
    }
    
    celebrateBirthday() {
        this.age++;
        return \`\${this.name} is now \${this.age}!\`;
    }
}

// Creating objects (instances)
const alice = new Person("Alice", 25);
const bob = new Person("Bob", 30);

// Using object methods
console.log(alice.greet());           // Hello, I'm Alice!
console.log(bob.celebrateBirthday()); // Bob is now 31!

// Accessing properties
console.log(alice.name);  // Alice
console.log(bob.age);     // 31`;

const javaCode = `// Defining a class in Java
public class Person {
    // Instance variables (attributes)
    private String name;
    private int age;
    
    // Constructor
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // Instance methods
    public String greet() {
        return "Hello, I'm " + this.name + "!";
    }
    
    public String celebrateBirthday() {
        this.age++;
        return this.name + " is now " + this.age + "!";
    }
    
    // Getters
    public String getName() { return this.name; }
    public int getAge() { return this.age; }
}

// Main class to run the program
public class Main {
    public static void main(String[] args) {
        // Creating objects
        Person alice = new Person("Alice", 25);
        Person bob = new Person("Bob", 30);
        
        // Using methods
        System.out.println(alice.greet());
        System.out.println(bob.celebrateBirthday());
    }
}`;

const cppCode = `// Defining a class in C++
#include <iostream>
#include <string>
using namespace std;

class Person {
private:
    string name;
    int age;

public:
    // Constructor
    Person(string n, int a) : name(n), age(a) {}
    
    // Instance methods
    string greet() {
        return "Hello, I'm " + name + "!";
    }
    
    string celebrateBirthday() {
        age++;
        return name + " is now " + to_string(age) + "!";
    }
    
    // Getters
    string getName() { return name; }
    int getAge() { return age; }
};

int main() {
    // Creating objects
    Person alice("Alice", 25);
    Person bob("Bob", 30);
    
    // Using methods
    cout << alice.greet() << endl;
    cout << bob.celebrateBirthday() << endl;
    
    return 0;
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is a class in OOP?",
    options: [
      "A blueprint for creating objects",
      "An instance of an object",
      "A variable that stores data",
      "A function that returns values",
    ],
    correctAnswer: 0,
    explanation:
      "A class is a blueprint or template that defines the attributes (data) and methods (behavior) that objects of that class will have.",
  },
  {
    id: 2,
    question: "What is the purpose of a constructor?",
    options: [
      "To destroy an object",
      "To initialize an object when it's created",
      "To define class methods",
      "To inherit from another class",
    ],
    correctAnswer: 1,
    explanation:
      "A constructor is a special method that is automatically called when an object is created. It initializes the object's attributes with initial values.",
  },
  {
    id: 3,
    question: "What is the difference between a class and an object?",
    options: [
      "They are the same thing",
      "A class is a blueprint, an object is an instance of that blueprint",
      "An object is a blueprint, a class is an instance",
      "Classes contain data, objects contain methods",
    ],
    correctAnswer: 1,
    explanation:
      "A class is a template/blueprint that defines structure and behavior. An object is a concrete instance created from that class with its own specific data.",
  },
  {
    id: 4,
    question: "What does 'self' or 'this' refer to in a class method?",
    options: [
      "The class itself",
      "The current instance of the class",
      "The parent class",
      "A global variable",
    ],
    correctAnswer: 1,
    explanation:
      "'self' (Python) or 'this' (Java, C++, JavaScript) refers to the current instance of the class, allowing methods to access and modify that specific object's attributes.",
  },
  {
    id: 5,
    question: "Can you create multiple objects from the same class?",
    options: [
      "No, only one object per class",
      "Yes, each object will have its own copy of attributes",
      "Yes, but they share the same attribute values",
      "Only in certain programming languages",
    ],
    correctAnswer: 1,
    explanation:
      "Yes, you can create unlimited objects from a single class. Each object (instance) has its own independent copy of instance attributes.",
  },
  {
    id: 6,
    question: "What is an instance method?",
    options: [
      "A method that belongs to the class, not instances",
      "A method that operates on a specific object instance",
      "A method that creates new instances",
      "A static method",
    ],
    correctAnswer: 1,
    explanation:
      "An instance method is a function defined in a class that operates on a specific object instance. It can access and modify that instance's attributes using 'self' or 'this'.",
  },
];

export default function ClassesPage() {
  const result = getSubtopicBySlug("oop", "classes");

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
            What are Classes & Objects?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Classes</strong> are the fundamental building blocks of 
            Object-Oriented Programming. A class is like a blueprint or template that defines the structure 
            and behavior of objects.
          </p>
          <p>
            <strong className="text-foreground">Objects</strong> are instances of classes - concrete entities 
            created from the blueprint. Each object has its own set of data (attributes) and can perform 
            actions (methods).
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
                Think of a <strong className="text-foreground">class</strong> as a cookie cutter, 
                and <strong className="text-foreground">objects</strong> as the cookies. The cookie cutter 
                (class) defines the shape, but each cookie (object) can have different decorations (data). 
                All cookies share the same basic shape, but each one is unique.
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
        <ClassObjectVisualizer />
      </section>

      {/* Class Structure */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Anatomy of a Class
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              Attributes (Properties)
            </h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Variables that store object data</li>
              <li>• Define the state of an object</li>
              <li>• Each instance has its own copy</li>
              <li>• Example: name, age, email</li>
            </ul>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500" />
              Methods (Functions)
            </h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Functions that define behavior</li>
              <li>• Can access and modify attributes</li>
              <li>• Operate on object's data</li>
              <li>• Example: greet(), save(), update()</li>
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

      {/* Key Concepts */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Key Concepts</h2>
        <div className="space-y-4">
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Constructor</h3>
            <p className="text-muted-foreground text-sm">
              A special method called automatically when creating a new object. It initializes the object's 
              attributes. In Python it's <code className="bg-muted px-1 rounded">__init__</code>, in Java/C++ 
              it's the method with the same name as the class.
            </p>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Instance vs Class</h3>
            <p className="text-muted-foreground text-sm">
              <strong>Instance attributes</strong> belong to each object individually. <strong>Class attributes</strong> 
              are shared by all instances of the class. Most attributes are instance-level.
            </p>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">self / this</h3>
            <p className="text-muted-foreground text-sm">
              The <code className="bg-muted px-1 rounded">self</code> (Python) or <code className="bg-muted px-1 rounded">this</code> 
              (Java, C++, JS) keyword refers to the current instance. It's how methods know which object's 
              data to work with.
            </p>
          </div>
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
              Forgetting self/this in methods
            </h4>
            <p className="text-sm text-muted-foreground">
              In Python, always include <code className="bg-muted px-1 rounded">self</code> as the first 
              parameter of instance methods. Without it, you can't access instance attributes.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Confusing class with object
            </h4>
            <p className="text-sm text-muted-foreground">
              You can't use <code className="bg-muted px-1 rounded">Person.greet()</code> directly - you 
              need an instance like <code className="bg-muted px-1 rounded">alice.greet()</code>.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Modifying class attributes accidentally
            </h4>
            <p className="text-sm text-muted-foreground">
              If you define mutable default values (like lists) as class attributes, they're shared 
              across all instances. Use instance attributes in the constructor instead.
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
        <div className="p-4 bg-muted/50 rounded-lg space-y-3">
          <div className="flex items-start gap-2">
            <span className="text-primary font-bold">Q:</span>
            <span className="text-muted-foreground">
              "Explain the difference between a class and an object."
            </span>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-success font-bold">A:</span>
            <span className="text-foreground">
              "A class is a blueprint that defines attributes and methods. An object is a concrete instance 
              of that class with its own data. You can create multiple objects from one class."
            </span>
          </div>
        </div>
      </section>

      {/* Quiz */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Test Your Knowledge
        </h2>
        <Quiz questions={quizQuestions} title="Classes & Objects Quiz" />
      </section>
    </TopicContent>
  );
}
