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
    </TopicContent>
  );
}
