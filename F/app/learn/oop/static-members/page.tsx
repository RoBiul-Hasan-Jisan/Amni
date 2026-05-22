"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
  FileQuestion,
  Database,
  Clock,
  Lock,
  Users,
  Box,
} from "lucide-react";

const pythonCode = `# Static Members in Python

class Counter:
    # Static/Class variable
    object_count = 0
    
    def __init__(self, name):
        self.name = name
        Counter.object_count += 1  # Access static variable via class name
    
    @classmethod
    def get_count(cls):
        """Static/Class method - can access class variables"""
        return cls.object_count
    
    @staticmethod
    def display_message():
        """Static method - cannot access instance or class variables"""
        return "This is a static method"

# Usage
c1 = Counter("First")
c2 = Counter("Second")
c3 = Counter("Third")

print(f"Total objects created: {Counter.get_count()}")  # 3
print(Counter.display_message())`;

const javaCode = `// Static Members in Java

class Counter {
    // Static variable - shared across all instances
    private static int objectCount = 0;
    private String name;
    
    public Counter(String name) {
        this.name = name;
        objectCount++;  // Increment static variable
    }
    
    // Static method - can only access static members
    public static int getCount() {
        return objectCount;
    }
    
    // Instance method - can access both static and instance members
    public void display() {
        System.out.println(name + " is object #" + objectCount);
    }
    
    // Static block - runs once when class is loaded
    static {
        System.out.println("Class loaded, initializing static members");
        objectCount = 0;
    }
}

public class Main {
    // Static variable
    static int staticVar = 100;
    
    // Instance variable
    int instanceVar = 200;
    
    // Static method
    static void staticMethod() {
        System.out.println("Static method called");
        System.out.println("Can access staticVar: " + staticVar);
        // System.out.println(instanceVar); // ERROR! Cannot access non-static
    }
    
    // Instance method
    void instanceMethod() {
        System.out.println("Instance method called");
        System.out.println("Can access staticVar: " + staticVar);
        System.out.println("Can access instanceVar: " + instanceVar);
    }
    
    public static void main(String[] args) {
        Counter c1 = new Counter("First");
        Counter c2 = new Counter("Second");
        Counter c3 = new Counter("Third");
        
        System.out.println("Total objects: " + Counter.getCount());  // 3
        
        staticMethod();
        
        Main obj = new Main();
        obj.instanceMethod();
    }
}`;

const cppCode = `// Static Members in C++

#include <iostream>
using namespace std;

class Counter {
private:
    static int objectCount;  // Static member declaration
    string name;
    
public:
    Counter(string n) : name(n) {
        objectCount++;
        cout << "Created: " << name << ", Total: " << objectCount << endl;
    }
    
    ~Counter() {
        objectCount--;
        cout << "Destroyed: " << name << ", Remaining: " << objectCount << endl;
    }
    
    // Static method - can only access static members
    static int getCount() {
        return objectCount;
    }
    
    void display() {
        cout << name << " - Current count: " << objectCount << endl;
    }
};

// Static member definition (required outside class)
int Counter::objectCount = 0;

class Demo {
public:
    static int staticVar;
    int instanceVar;
    
    static void staticMethod() {
        cout << "Static method - can access staticVar: " << staticVar << endl;
        // cout << instanceVar; // ERROR! Cannot access non-static
    }
    
    void instanceMethod() {
        cout << "Instance method - can access staticVar: " << staticVar << endl;
        cout << "Instance method - can access instanceVar: " << instanceVar << endl;
    }
};

int Demo::staticVar = 100;

int main() {
    Counter c1("First");
    Counter c2("Second");
    Counter c3("Third");
    
    cout << "Total objects created: " << Counter::getCount() << endl;
    
    Demo::staticMethod();
    
    Demo d;
    d.instanceVar = 50;
    d.instanceMethod();
    
    return 0;
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is a static variable?",
    options: [
      "A variable that cannot be changed",
      "A variable shared across all instances of a class",
      "A variable that is local to a method",
      "A variable that is automatically initialized to zero",
    ],
    correctAnswer: 1,
    explanation: "Static variables belong to the class rather than any instance. All objects share the same static variable.",
  },
  {
    id: 2,
    question: "Can a static method access non-static members?",
    options: [
      "Yes, always",
      "No, never",
      "Only if the non-static member is public",
      "Only through an object reference",
    ],
    correctAnswer: 3,
    explanation: "Static methods cannot directly access non-static members because they don't have a 'this' reference. However, they can access non-static members through an object reference.",
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

export default function StaticMembersPage() {
  const result = getSubtopicBySlug("oop", "static-members");

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
            What are Static Members?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Static members</strong> belong to the class itself rather 
            than to individual instances (objects). They are shared across all objects of the class.
          </p>
          <p>
            Static variables (class variables) have a single copy that all objects share. Static methods 
            can be called without creating an object of the class.
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
                Think of a <strong className="text-foreground">classroom</strong>. Each student (instance) 
                has their own name and grade (instance variables). But the 
                <strong className="text-foreground"> classroom number</strong> is shared by all students 
                (static variable) and the <strong className="text-foreground">school rules</strong> apply 
                to everyone (static methods).
              </p>
            </div>
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
            { language: "java", label: "Java", code: javaCode },
            { language: "cpp", label: "C++", code: cppCode },
          ]}
        />
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

        {/* Question 30: Static Data Members & Functions */}
        <QuestionCard
          number={30}
          title="Static Data Members & Functions"
          question="What are static data members and static member functions? Explain with example."
          answer="Static data members (class variables) belong to the class rather than any instance - all objects share the same copy. They are initialized once and exist even without creating objects. Static member functions can be called without an object and can only access static members directly. They are commonly used for counting objects, utility functions, or implementing singleton patterns."
          marks={5}
          icon={<Database className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class Employee {
    // Static data member - shared across all instances
    private static int totalEmployees = 0;
    private static String companyName = "TechCorp";
    
    // Instance members
    private String name;
    private int id;
    
    public Employee(String name) {
        this.name = name;
        this.id = ++totalEmployees;
    }
    
    // Static method - can only access static members directly
    public static int getTotalEmployees() {
        return totalEmployees;
    }
    
    public static String getCompanyName() {
        return companyName;
    }
    
    // Instance method - can access both
    public void display() {
        System.out.println("ID: " + id + ", Name: " + name + 
                           ", Company: " + companyName);
    }
    
    public static void main(String[] args) {
        // Static method called without object
        System.out.println("Initial count: " + Employee.getTotalEmployees());
        
        Employee e1 = new Employee("Alice");
        Employee e2 = new Employee("Bob");
        
        System.out.println("Total employees: " + Employee.getTotalEmployees());
        System.out.println("Company: " + Employee.getCompanyName());
        
        e1.display();
        e2.display();
    }
}`}
          />
        </QuestionCard>

        {/* Question 31: Count Objects Program */}
        <QuestionCard
          number={31}
          title="Count Objects Using Static Variable"
          question="Write a program to count the number of objects created using a static variable."
          answer="A static variable is perfect for counting objects because it's shared across all instances. Increment it in the constructor, and optionally decrement in the destructor (C++) or finalize (Java). This demonstrates how static members maintain state across instances."
          marks={5}
          icon={<Users className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class ObjectCounter {
private:
    static int count;  // Static counter
    int objectId;
    string name;
    
public:
    // Constructor - increments count
    ObjectCounter(string n) : name(n) {
        count++;
        objectId = count;
        cout << "Object " << objectId << " (" << name << ") created" << endl;
        cout << "Total objects: " << count << endl;
    }
    
    // Destructor - decrements count (C++ specific)
    ~ObjectCounter() {
        cout << "Object " << objectId << " (" << name << ") destroyed" << endl;
        count--;
        cout << "Remaining objects: " << count << endl;
    }
    
    static int getCount() {
        return count;
    }
    
    void display() {
        cout << "Object ID: " << objectId << ", Name: " << name << endl;
    }
};

// Initialize static member
int ObjectCounter::count = 0;

int main() {
    cout << "Starting program..." << endl;
    cout << "Initial count: " << ObjectCounter::getCount() << endl;
    
    ObjectCounter obj1("First");
    ObjectCounter obj2("Second");
    
    {
        ObjectCounter obj3("Third - inside block");
        obj3.display();
    }  // obj3 destroyed here
    
    ObjectCounter obj4("Fourth");
    
    cout << "\\nFinal count: " << ObjectCounter::getCount() << endl;
    
    return 0;
}

/* Output:
Starting program...
Initial count: 0
Object 1 (First) created
Total objects: 1
Object 2 (Second) created
Total objects: 2
Object 3 (Third - inside block) created
Total objects: 3
Object ID: 3, Name: Third - inside block
Object 3 (Third - inside block) destroyed
Remaining objects: 2
Object 4 (Fourth) created
Total objects: 3
Final count: 3
*/`}
          />
        </QuestionCard>

        {/* Question 32: Static Function Access Non-Static */}
        <QuestionCard
          number={32}
          title="Static Function & Non-Static Members"
          question="Can a static function access non-static members? Why?"
          answer="No, static functions cannot directly access non-static members. Reason: Static methods belong to the class and have no 'this' reference. They don't know which object's instance variables to access. Non-static members require an object instance to exist. However, static methods CAN access non-static members through an object reference passed as parameter or created inside the method."
          marks={5}
          icon={<Lock className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">Why Static Can't Access Non-Static Directly:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li>Static methods have no 'this' pointer/reference</li>
              <li>Non-static members belong to specific objects</li>
              <li>Static method can be called without any object existing</li>
              <li>Without an object, instance variables don't exist in memory</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`public class AccessDemo {
    private int instanceVar = 10;
    private static int staticVar = 20;
    
    // Static method
    public static void staticMethod() {
        // Direct access - NOT allowed
        // System.out.println(instanceVar);  // COMPILE ERROR!
        
        // Can access static variables directly
        System.out.println("Static var: " + staticVar);
        
        // Can access instance members through an object
        AccessDemo obj = new AccessDemo();
        System.out.println("Instance var via object: " + obj.instanceVar);
        
        // Can call instance methods through an object
        obj.instanceMethod();
    }
    
    // Instance method
    public void instanceMethod() {
        // Instance methods can access everything directly
        System.out.println("Instance var: " + instanceVar);
        System.out.println("Static var: " + staticVar);
        staticMethod();  // Can call static method
    }
    
    public static void main(String[] args) {
        // Static method called without object
        staticMethod();
        
        // Instance method needs object
        AccessDemo obj = new AccessDemo();
        obj.instanceMethod();
    }
}`}
          />
        </QuestionCard>

        {/* Question 33: Static Variable Initialization */}
        <QuestionCard
          number={33}
          title="Static Variable Initialization"
          question="When are static variables initialized?"
          answer="Static variables are initialized when the class is first loaded into memory by the ClassLoader. This happens before any objects are created and before any static methods are called. In Java/C++, static initialization occurs once per class. Order: static blocks run in the order they appear, then static variables are initialized. In C++, static members must be defined outside the class."
          marks={5}
          icon={<Clock className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class StaticInitOrder {
    // Static variable
    static int staticVar = initializeStatic();
    
    // Instance variable
    int instanceVar = initializeInstance();
    
    // Static block - runs when class loads
    static {
        System.out.println("1. Static block executed");
        System.out.println("   staticVar = " + staticVar);
    }
    
    // Instance initializer block - runs before constructor
    {
        System.out.println("3. Instance initializer block");
    }
    
    // Constructor
    public StaticInitOrder() {
        System.out.println("4. Constructor executed");
    }
    
    static int initializeStatic() {
        System.out.println("   Static variable initialization");
        return 100;
    }
    
    int initializeInstance() {
        System.out.println("   Instance variable initialization");
        return 200;
    }
    
    public static void main(String[] args) {
        System.out.println("2. Main method starts");
        
        System.out.println("\\nCreating first object:");
        StaticInitOrder obj1 = new StaticInitOrder();
        
        System.out.println("\\nCreating second object:");
        StaticInitOrder obj2 = new StaticInitOrder();
        
        // Static block doesn't run again for second object
    }
}

/* Output:
   Static variable initialization
1. Static block executed
   staticVar = 100
2. Main method starts

Creating first object:
   Instance variable initialization
3. Instance initializer block
4. Constructor executed

Creating second object:
   Instance variable initialization
3. Instance initializer block
4. Constructor executed
*/`}
          />
        </QuestionCard>

        {/* Question 34: Static/Non-Static Output */}
        <QuestionCard
          number={34}
          title="Static vs Non-Static Output"
          question="Write the output of a code involving static and non-static members."
          answer="The output demonstrates the order of initialization and access rules: static members initialize first (when class loads), then instance members (when object created). Static methods can't access instance members directly. Each object gets its own copy of instance variables but shares static variables."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <CodeBlock
            language="java"
            code={`public class OutputDemo {
    static int staticCounter = 0;
    int instanceCounter = 0;
    
    public OutputDemo() {
        staticCounter++;
        instanceCounter++;
        System.out.println("Constructor: static=" + staticCounter + 
                           ", instance=" + instanceCounter);
    }
    
    public void increment() {
        staticCounter++;
        instanceCounter++;
        System.out.println("Increment: static=" + staticCounter + 
                           ", instance=" + instanceCounter);
    }
    
    public static void staticDisplay() {
        System.out.println("Static method: static=" + staticCounter);
        // System.out.println(instanceCounter); // ERROR
    }
    
    public static void main(String[] args) {
        System.out.println("=== Creating first object ===");
        OutputDemo obj1 = new OutputDemo();  // static=1, instance=1
        
        System.out.println("\\n=== Creating second object ===");
        OutputDemo obj2 = new OutputDemo();  // static=2, instance=1
        
        System.out.println("\\n=== Increment obj1 ===");
        obj1.increment();  // static=3, instance=2
        
        System.out.println("\\n=== Increment obj2 ===");
        obj2.increment();  // static=4, instance=2
        
        System.out.println("\\n=== Static method call ===");
        OutputDemo.staticDisplay();  // static=4
    }
}

/* Output:
=== Creating first object ===
Constructor: static=1, instance=1

=== Creating second object ===
Constructor: static=2, instance=1

=== Increment obj1 ===
Increment: static=3, instance=2

=== Increment obj2 ===
Increment: static=4, instance=2

=== Static method call ===
Static method: static=4
*/`}
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
              <span className="font-bold text-primary">Q30:</span>
              <span className="text-muted-foreground ml-1">Static members belong to class, shared by all instances</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q31:</span>
              <span className="text-muted-foreground ml-1">Use static counter in constructor to count objects</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q32:</span>
              <span className="text-muted-foreground ml-1">Static methods need object to access non-static members</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q33:</span>
              <span className="text-muted-foreground ml-1">Static variables initialize when class loads (once)</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q34:</span>
              <span className="text-muted-foreground ml-1">Static shared, instance unique per object</span>
            </div>
          </div>
        </div>
      </section>

      {/* Quiz */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Test Your Knowledge
        </h2>
        <Quiz questions={quizQuestions} title="Static Members Quiz" />
      </section>
    </TopicContent>
  );
}