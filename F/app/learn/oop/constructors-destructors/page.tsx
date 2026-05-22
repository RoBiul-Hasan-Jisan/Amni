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
  Box,
  Copy,
  Trash2,
  Link,
  Lock,
  Users,
  Clock,
} from "lucide-react";

const cppConstructorCode = `// C++ Constructors & Destructors

#include <iostream>
#include <string>
using namespace std;

class Student {
private:
    string name;
    int age;
    static int objectCount;
    
public:
    // 1. Default Constructor
    Student() {
        name = "Unknown";
        age = 0;
        objectCount++;
        cout << "Default constructor called. Total objects: " << objectCount << endl;
    }
    
    // 2. Parameterized Constructor
    Student(string n, int a) {
        name = n;
        age = a;
        objectCount++;
        cout << "Parameterized constructor called for " << name << endl;
    }
    
    // 3. Copy Constructor
    Student(const Student &other) {
        name = other.name;
        age = other.age;
        objectCount++;
        cout << "Copy constructor called for " << name << endl;
    }
    
    // 4. Destructor
    ~Student() {
        objectCount--;
        cout << "Destructor called for " << name << ". Remaining: " << objectCount << endl;
    }
    
    void display() {
        cout << "Name: " << name << ", Age: " << age << endl;
    }
    
    static int getCount() {
        return objectCount;
    }
};

int Student::objectCount = 0;

// Constructor Chaining Example
class Animal {
protected:
    string species;
public:
    Animal() : species("Unknown") {
        cout << "Animal default constructor" << endl;
    }
    
    Animal(string s) : species(s) {
        cout << "Animal parameterized constructor: " << species << endl;
    }
};

class Dog : public Animal {
private:
    string breed;
public:
    // Constructor chaining - calling base class constructor
    Dog() : Animal(), breed("Unknown") {
        cout << "Dog default constructor" << endl;
    }
    
    Dog(string s, string b) : Animal(s), breed(b) {
        cout << "Dog parameterized constructor: " << breed << endl;
    }
};

// Private Constructor - Singleton Pattern
class Singleton {
private:
    static Singleton* instance;
    
    // Private constructor
    Singleton() {
        cout << "Singleton instance created" << endl;
    }
    
public:
    static Singleton* getInstance() {
        if (instance == nullptr) {
            instance = new Singleton();
        }
        return instance;
    }
    
    void showMessage() {
        cout << "Hello from Singleton!" << endl;
    }
};

Singleton* Singleton::instance = nullptr;

int main() {
    cout << "=== Constructors Demo ===" << endl;
    Student s1;                    // Default constructor
    Student s2("Alice", 20);       // Parameterized constructor
    Student s3(s2);                // Copy constructor
    
    s1.display();
    s2.display();
    s3.display();
    
    cout << "\\n=== Constructor Chaining ===" << endl;
    Dog d1("Canine", "Labrador");
    
    cout << "\\n=== Singleton Pattern ===" << endl;
    Singleton* s = Singleton::getInstance();
    s->showMessage();
    
    return 0;
}`;

const javaConstructorCode = `// Java Constructors (No destructors)

class Student {
    private String name;
    private int age;
    private static int objectCount = 0;
    
    // 1. Default Constructor
    Student() {
        this("Unknown", 0);  // Constructor chaining
        System.out.println("Default constructor called");
    }
    
    // 2. Parameterized Constructor
    Student(String name, int age) {
        this.name = name;
        this.age = age;
        objectCount++;
        System.out.println("Parameterized constructor called for " + name);
    }
    
    // 3. Copy Constructor (No built-in, need to create manually)
    Student(Student other) {
        this.name = other.name;
        this.age = other.age;
        objectCount++;
        System.out.println("Copy constructor called for " + name);
    }
    
    void display() {
        System.out.println("Name: " + name + ", Age: " + age);
    }
    
    static int getCount() {
        return objectCount;
    }
    
    // Java doesn't have destructors - uses garbage collection
    @Override
    protected void finalize() throws Throwable {
        objectCount--;
        System.out.println("Finalize called for " + name);
        super.finalize();
    }
}

// Constructor Chaining with Inheritance
class Animal {
    protected String species;
    
    Animal() {
        this("Unknown");
        System.out.println("Animal default constructor");
    }
    
    Animal(String species) {
        this.species = species;
        System.out.println("Animal parameterized constructor: " + species);
    }
}

class Dog extends Animal {
    private String breed;
    
    Dog() {
        this("Unknown", "Unknown");
        System.out.println("Dog default constructor");
    }
    
    Dog(String species, String breed) {
        super(species);  // Call parent constructor - MUST be first
        this.breed = breed;
        System.out.println("Dog parameterized constructor: " + breed);
    }
}

// Singleton Pattern with Private Constructor
class Singleton {
    private static Singleton instance;
    
    private Singleton() {
        System.out.println("Singleton created");
    }
    
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
    
    public void showMessage() {
        System.out.println("Hello from Singleton!");
    }
}

// Object Counter using static
class ObjectCounter {
    private static int count = 0;
    private int id;
    
    public ObjectCounter() {
        count++;
        id = count;
        System.out.println("Object " + id + " created. Total: " + count);
    }
    
    public static int getCount() {
        return count;
    }
}

public class Main {
    public static void main(String[] args) {
        System.out.println("=== Constructors Demo ===");
        Student s1 = new Student();
        Student s2 = new Student("Alice", 20);
        Student s3 = new Student(s2);
        
        s1.display();
        s2.display();
        s3.display();
        
        System.out.println("\\n=== Constructor Chaining ===");
        Dog d1 = new Dog("Canine", "Labrador");
        
        System.out.println("\\n=== Singleton Pattern ===");
        Singleton s = Singleton.getInstance();
        s.showMessage();
        
        System.out.println("\\n=== Object Counter ===");
        ObjectCounter o1 = new ObjectCounter();
        ObjectCounter o2 = new ObjectCounter();
        ObjectCounter o3 = new ObjectCounter();
        System.out.println("Total objects: " + ObjectCounter.getCount());
        
        // Garbage collector may call finalize
        System.gc();
    }
}`;

const pythonConstructorCode = `# Python Constructors & Destructors

class Student:
    object_count = 0
    
    # Constructor (__init__)
    def __init__(self, name="Unknown", age=0):
        self.name = name
        self.age = age
        Student.object_count += 1
        print(f"Constructor called for {self.name}. Total: {Student.object_count}")
    
    # Destructor (__del__) - not guaranteed to run immediately
    def __del__(self):
        Student.object_count -= 1
        print(f"Destructor called for {self.name}. Remaining: {Student.object_count}")
    
    def display(self):
        print(f"Name: {self.name}, Age: {self.age}")
    
    @classmethod
    def get_count(cls):
        return cls.object_count
    
    # Copy constructor equivalent
    @classmethod
    def from_student(cls, other):
        return cls(other.name, other.age)

# Constructor Chaining with Inheritance
class Animal:
    def __init__(self, species="Unknown"):
        self.species = species
        print(f"Animal constructor: {species}")

class Dog(Animal):
    def __init__(self, species="Unknown", breed="Unknown"):
        super().__init__(species)  # Call parent constructor
        self.breed = breed
        print(f"Dog constructor: {breed}")

# Singleton Pattern with Private Constructor (using metaclass)
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        print("Singleton created")
    
    def show_message(self):
        print("Hello from Singleton!")

# Object Counter Demo
class ObjectCounter:
    _count = 0
    
    def __init__(self):
        ObjectCounter._count += 1
        self._id = ObjectCounter._count
        print(f"Object {self._id} created. Total: {ObjectCounter._count}")
    
    @classmethod
    def get_count(cls):
        return cls._count
    
    def __del__(self):
        ObjectCounter._count -= 1
        print(f"Object {self._id} destroyed. Remaining: {ObjectCounter._count}")

if __name__ == "__main__":
    print("=== Constructors Demo ===")
    s1 = Student()
    s2 = Student("Alice", 20)
    s3 = Student.from_student(s2)  # Copy
    
    s1.display()
    s2.display()
    s3.display()
    
    print(f"\\nTotal students: {Student.get_count()}")
    
    print("\\n=== Constructor Chaining ===")
    d1 = Dog("Canine", "Labrador")
    
    print("\\n=== Singleton Pattern ===")
    s = Singleton()
    t = Singleton()  # Same instance
    print(f"Same instance? {s is t}")
    s.show_message()
    
    print("\\n=== Object Counter ===")
    o1 = ObjectCounter()
    o2 = ObjectCounter()
    o3 = ObjectCounter()
    print(f"Total objects: {ObjectCounter.get_count()}")`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is a constructor?",
    options: [
      "A method that destroys an object",
      "A special method that initializes objects when created",
      "A method that copies an object",
      "A static method",
    ],
    correctAnswer: 1,
    explanation: "A constructor is a special method that is automatically called when an object is instantiated to initialize the object's state.",
  },
  {
    id: 2,
    question: "Does Java have destructors?",
    options: [
      "Yes, exactly like C++",
      "No, Java uses garbage collection instead",
      "Yes, but they are called finalize()",
      "Both B and C",
    ],
    correctAnswer: 3,
    explanation: "Java doesn't have destructors like C++. It uses garbage collection, and finalize() method is deprecated. try-with-resources is preferred for cleanup.",
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

export default function ConstructorsDestructorsPage() {
  const result = getSubtopicBySlug("oop", "constructors-destructors");

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
            Constructors & Destructors
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Constructors</strong> are special methods that initialize 
            objects when they are created. <strong className="text-foreground">Destructors</strong> clean up 
            resources when objects are destroyed (C++ only - Java uses garbage collection).
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
            Code Implementation
          </h2>
        </div>
        <MultiLanguageCode
          codes={[
            { language: "cpp", label: "C++", code: cppConstructorCode },
            { language: "java", label: "Java", code: javaConstructorCode },
            { language: "python", label: "Python", code: pythonConstructorCode },
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

        {/* Question 7: What is a constructor? Types */}
        <QuestionCard
          number={7}
          title="Constructor & Types"
          question="What is a constructor? Explain different types of constructors with examples."
          answer="A constructor is a special member function automatically called when an object is created. Types: (1) Default - no parameters, (2) Parameterized - accepts arguments for initialization, (3) Copy - creates object as copy of another, (4) Constructor overloading - multiple constructors with different parameters. Constructors have same name as class, no return type, can be overloaded."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Types of Constructors"
            headers={["Type", "Description", "Syntax Example", "When Called"]}
            rows={[
              ["Default", "No parameters", "Student() { }", "Student s1;"],
              ["Parameterized", "Takes arguments", "Student(string n) { }", "Student s1('John');"],
              ["Copy", "Creates copy", "Student(Student &s) { }", "Student s2(s1);"],
              ["Move (C++11)", "Transfers resources", "Student(Student &&s) { }", "Student s2(move(s1));"],
            ]}
          />
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class Example {
private:
    int value;
    string name;
    
public:
    // 1. Default Constructor
    Example() {
        value = 0;
        name = "Default";
        cout << "Default constructor called" << endl;
    }
    
    // 2. Parameterized Constructor
    Example(int v, string n) {
        value = v;
        name = n;
        cout << "Parameterized constructor called: " << name << endl;
    }
    
    // 3. Copy Constructor
    Example(const Example &other) {
        value = other.value;
        name = other.name + " (copy)";
        cout << "Copy constructor called for: " << name << endl;
    }
    
    // 4. Constructor Overloading
    Example(int v) : Example(v, "SingleParam") {
        cout << "Delegating constructor" << endl;
    }
    
    void display() {
        cout << "Name: " << name << ", Value: " << value << endl;
    }
};

int main() {
    Example e1;                    // Default
    Example e2(100, "Original");   // Parameterized
    Example e3(e2);                // Copy
    Example e4(50);                // Overloaded (delegating)
    
    e1.display();
    e2.display();
    e3.display();
    e4.display();
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 8: Copy Constructor */}
        <QuestionCard
          number={8}
          title="Copy Constructor"
          question="What is a copy constructor? When is it called? Write an example."
          answer="A copy constructor creates an object by copying another object of the same class. Called in three scenarios: (1) When creating object as copy of another (Class obj2(obj1)), (2) When passing object by value to function, (3) When returning object by value from function. Default copy constructor performs shallow copy; custom needed for deep copy with dynamic memory."
          marks={5}
          icon={<Copy className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">When Copy Constructor is Called:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li><strong>Explicit copy:</strong> MyClass obj2(obj1);</li>
              <li><strong>Pass by value:</strong> void func(MyClass obj) { }</li>
              <li>
  <strong className="text-foreground">Return by Value:</strong>{" "}
  <code>ClassName function() &#123; return obj; &#125;</code>
</li>
            </ul>
          </div>
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
#include <cstring>
using namespace std;

class String {
private:
    char* data;
    int length;
    
public:
    // Constructor
    String(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
        cout << "Constructor: " << data << endl;
    }
    
    // Copy Constructor (Deep Copy)
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];
        strcpy(data, other.data);
        cout << "Copy Constructor: " << data << endl;
    }
    
    // Destructor
    ~String() {
        cout << "Destructor: " << data << endl;
        delete[] data;
    }
    
    void display() {
        cout << "String: " << data << ", Length: " << length << endl;
    }
    
    void modify(const char* newStr) {
        delete[] data;
        length = strlen(newStr);
        data = new char[length + 1];
        strcpy(data, newStr);
    }
};

// Function that takes object by value (triggers copy)
void printString(String s) {
    s.display();
}

// Function that returns object by value (triggers copy)
String createString(const char* str) {
    String temp(str);
    return temp;  // Copy constructor called for return
}

int main() {
    cout << "=== Scenario 1: Explicit Copy ===" << endl;
    String s1("Hello");
    String s2(s1);  // Copy constructor called
    
    cout << "\\n=== Scenario 2: Pass by Value ===" << endl;
    printString(s1);  // Copy constructor called
    
    cout << "\\n=== Scenario 3: Return by Value ===" << endl;
    String s3 = createString("World");  // Copy constructor called
    
    cout << "\\n=== Deep Copy Demonstration ===" << endl;
    String original("Original");
    String copy = original;  // Deep copy
    
    original.modify("Modified");
    
    cout << "Original: ";
    original.display();
    cout << "Copy: ";
    copy.display();  // Unaffected by original's modification
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 9: Destructor - C++ vs Java */}
        <QuestionCard
          number={9}
          title="Destructor - C++ vs Java"
          question="What is a destructor? Why is it needed in C++ but not in Java?"
          answer="Destructor is a special method that cleans up resources when an object is destroyed. C++ needs destructors for manual memory management, releasing resources (file handles, network connections, dynamic memory). Java has automatic garbage collection that reclaims memory, and try-with-resources for resource cleanup. Java's finalize() is deprecated; C++ destructors are deterministic (called immediately when object goes out of scope)."
          marks={5}
          icon={<Trash2 className="h-3 w-3" />}
        >
          <div className="p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg mb-4">
            <h4 className="font-medium text-yellow-500 mb-1">⚠️ Key Difference</h4>
            <p className="text-sm text-muted-foreground">
              C++ destructors are called <strong>deterministically</strong> (immediately when object goes out of scope). 
              Java has <strong>non-deterministic</strong> garbage collection - you never know when an object is actually destroyed.
            </p>
          </div>
          <DifferenceTable
            title="Destructors: C++ vs Java"
            headers={["Feature", "C++", "Java"]}
            rows={[
              ["Destructor syntax", "~ClassName() { }", "No destructor (finalize() - deprecated)"],
              ["Called when", "Object goes out of scope / delete", "Unpredictable (GC)"],
              ["Deterministic", "✓ Yes", "✗ No"],
              ["Manual memory", "Required (delete, delete[])", "Automatic (GC)"],
              ["Resource cleanup", "RAII (Resource Acquisition Is Initialization)", "try-with-resources"],
              ["Order", "Reverse of construction", "Not guaranteed"],
            ]}
          />
          <CodeBlock
            language="cpp"
            code={`// C++ - Destructor Example
#include <iostream>
#include <fstream>
using namespace std;

class FileHandler {
private:
    FILE* file;
    string filename;
    
public:
    FileHandler(const string& name) : filename(name) {
        file = fopen(name.c_str(), "r");
        cout << "Opened file: " << filename << endl;
    }
    
    // Destructor - guaranteed to run when object goes out of scope
    ~FileHandler() {
        if (file) {
            fclose(file);
            cout << "Closed file: " << filename << endl;
        }
    }
    
    void read() {
        if (file) {
            cout << "Reading from: " << filename << endl;
        }
    }
};

class DatabaseConnection {
private:
    string connectionString;
    
public:
    DatabaseConnection(const string& conn) : connectionString(conn) {
        cout << "Connected to DB: " << connectionString << endl;
    }
    
    ~DatabaseConnection() {
        cout << "Disconnected from DB: " << connectionString << endl;
    }
    
    void query(const string& sql) {
        cout << "Executing: " << sql << endl;
    }
};

int main() {
    cout << "=== Destructor Demonstration ===\\n" << endl;
    
    {
        FileHandler fh("data.txt");
        fh.read();
        // Destructor called automatically when leaving scope
    }
    
    cout << "\\n=== Multiple Resources ===\\n" << endl;
    
    {
        DatabaseConnection db("localhost:3306");
        db.query("SELECT * FROM users");
        // Destructor called in reverse order of construction
    }
    
    cout << "\\n=== RAII (Resource Acquisition Is Initialization) ===\\n" << endl;
    
    // Resources automatically managed via destructors
    // No manual cleanup needed!
    
    return 0;
}

// Java Equivalent (try-with-resources)
/*
try (FileReader fr = new FileReader("data.txt");
     BufferedReader br = new BufferedReader(fr)) {
    // Auto-closed when try block exits
} catch (IOException e) {
    e.printStackTrace();
}
*/`}
          />
        </QuestionCard>

        {/* Question 10: Constructor Chaining */}
        <QuestionCard
          number={10}
          title="Constructor Chaining"
          question="What is constructor chaining? How is it done in C++ vs Java?"
          answer="Constructor chaining is calling one constructor from another within the same class or parent class to avoid code duplication. C++ uses member initializer list (constructorName(params) : member1(value) { }). Java uses this() for same class and super() for parent class - must be first statement. Python uses super().__init__()."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <DifferenceTable
            title="Constructor Chaining: C++ vs Java vs Python"
            headers={["Feature", "C++", "Java", "Python"]}
            rows={[
              ["Same class", "Delegating constructor (C++11)", "this()", "self.__init__()"],
              ["Parent class", "Member initializer list", "super()", "super().__init__()"],
              ["Must be first", "Yes (initializer list)", "Yes (first statement)", "Yes"],
              ["Multiple chaining", "Yes (multiple initializers)", "No (single this/super)", "Yes"],
            ]}
          />
          <CodeBlock
            language="cpp"
            code={`// C++ Constructor Chaining
class Employee {
protected:
    string name;
    int id;
    double salary;
    
public:
    // Primary constructor
    Employee(string n, int i, double s) : name(n), id(i), salary(s) {
        cout << "Employee: " << name << ", ID: " << id << endl;
    }
    
    // Delegating to primary constructor
    Employee(string n, int i) : Employee(n, i, 30000.0) {
        cout << "Employee with default salary" << endl;
    }
    
    Employee(string n) : Employee(n, 0, 30000.0) {
        cout << "Employee with default ID" << endl;
    }
};

class Manager : public Employee {
private:
    double bonus;
    
public:
    // Call parent constructor via initializer list
    Manager(string n, int i, double s, double b) 
        : Employee(n, i, s), bonus(b) {
        cout << "Manager: " << name << ", Bonus: " << bonus << endl;
    }
    
    // Chain to another constructor
    Manager(string n, int i) : Manager(n, i, 50000.0, 10000.0) {
        cout << "Manager with defaults" << endl;
    }
};

// Java Constructor Chaining
/*
class Employee {
    String name;
    int id;
    double salary;
    
    Employee(String name, int id, double salary) {
        this.name = name;
        this.id = id;
        this.salary = salary;
    }
    
    Employee(String name, int id) {
        this(name, id, 30000.0);  // this() must be first
    }
    
    Employee(String name) {
        this(name, 0);  // Chaining
    }
}

class Manager extends Employee {
    double bonus;
    
    Manager(String name, int id, double salary, double bonus) {
        super(name, id, salary);  // super() must be first
        this.bonus = bonus;
    }
    
    Manager(String name, int id) {
        super(name, id);  // Call parent constructor
        this.bonus = 10000.0;
    }
}
*/

int main() {
    cout << "=== Constructor Chaining Demo ===" << endl;
    Employee e1("Alice", 101, 50000);
    Employee e2("Bob", 102);
    Employee e3("Charlie");
    
    cout << "\\n=== Inheritance Chaining ===" << endl;
    Manager m1("David", 201, 75000, 15000);
    Manager m2("Eve", 202);
    
    return 0;
}`}
          />
        </QuestionCard>

        {/* Question 11: Private Constructor & Singleton */}
        <QuestionCard
          number={11}
          title="Private Constructor & Singleton"
          question="Can a constructor be private? What is the use of a private constructor? Explain with Singleton pattern."
          answer="Yes, constructors can be private. A private constructor prevents instantiation from outside the class. Uses: (1) Singleton pattern - ensure only one instance, (2) Utility classes with only static methods, (3) Factory methods controlling object creation, (4) Builder pattern. Singleton pattern uses private constructor with static getInstance() method returning single instance."
          marks={5}
          icon={<Lock className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg mb-4">
            <h4 className="font-medium text-foreground mb-2">When to Use Private Constructor:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li><strong>Singleton:</strong> Only one instance of class allowed</li>
              <li><strong>Utility class:</strong> Math class with only static methods</li>
              <li><strong>Factory pattern:</strong> Control object creation</li>
              <li><strong>Builder pattern:</strong> Complex object construction</li>
              <li><strong>Constants class:</strong> Group related constants</li>
            </ul>
          </div>
          <CodeBlock
            language="java"
            code={`// Singleton Pattern with Private Constructor
public class DatabaseConnection {
    // 1. Private static instance
    private static DatabaseConnection instance;
    
    private String connectionUrl;
    private boolean connected;
    
    // 2. Private constructor (prevents external instantiation)
    private DatabaseConnection() {
        connectionUrl = "jdbc:mysql://localhost:3306/mydb";
        connected = false;
        System.out.println("DatabaseConnection constructor called");
    }
    
    // 3. Public static getInstance method
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            instance = new DatabaseConnection();
        }
        return instance;
    }
    
    // Business methods
    public void connect() {
        if (!connected) {
            System.out.println("Connecting to: " + connectionUrl);
            connected = true;
        }
    }
    
    public void disconnect() {
        if (connected) {
            System.out.println("Disconnected from database");
            connected = false;
        }
    }
    
    public void executeQuery(String sql) {
        if (connected) {
            System.out.println("Executing: " + sql);
        } else {
            System.out.println("Not connected to database!");
        }
    }
}

// Thread-safe Singleton (Double-checked locking)
class ThreadSafeSingleton {
    private static volatile ThreadSafeSingleton instance;
    
    private ThreadSafeSingleton() { }
    
    public static ThreadSafeSingleton getInstance() {
        if (instance == null) {
            synchronized (ThreadSafeSingleton.class) {
                if (instance == null) {
                    instance = new ThreadSafeSingleton();
                }
            }
        }
        return instance;
    }
}

// Bill Pugh Singleton (Best Practice)
class BillPughSingleton {
    private BillPughSingleton() { }
    
    private static class SingletonHelper {
        private static final BillPughSingleton INSTANCE = new BillPughSingleton();
    }
    
    public static BillPughSingleton getInstance() {
        return SingletonHelper.INSTANCE;
    }
}

// Utility Class with Private Constructor
class MathUtils {
    // Private constructor prevents instantiation
    private MathUtils() {
        throw new UnsupportedOperationException("Utility class");
    }
    
    public static int add(int a, int b) { return a + b; }
    public static int multiply(int a, int b) { return a * b; }
    public static double circleArea(double radius) { return Math.PI * radius * radius; }
}

// Usage
public class SingletonDemo {
    public static void main(String[] args) {
        // Cannot create instance directly:
        // DatabaseConnection db = new DatabaseConnection(); // ERROR!
        
        // Get instance via static method
        DatabaseConnection db1 = DatabaseConnection.getInstance();
        DatabaseConnection db2 = DatabaseConnection.getInstance();
        
        System.out.println("Same instance? " + (db1 == db2));  // true
        
        db1.connect();
        db1.executeQuery("SELECT * FROM users");
        db2.executeQuery("SELECT * FROM orders");  // Same connection
        
        // Utility class usage
        System.out.println("Sum: " + MathUtils.add(5, 10));
        System.out.println("Area: " + MathUtils.circleArea(5));
        
        // MathUtils utils = new MathUtils(); // ERROR! Private constructor
    }
}`}
          />
        </QuestionCard>

        {/* Question 12: Count Objects with Static */}
        <QuestionCard
          number={12}
          title="Count Objects Using Static Variable"
          question="Write a class that counts how many objects have been created (using static member)."
          answer="Use a static member variable that increments in constructor and decrements in destructor (C++). Static members belong to class, not instances, making them perfect for object counting. This demonstrates object lifecycle tracking and static member usage."
          marks={5}
          icon={<Users className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
#include <string>
using namespace std;

class ObjectCounter {
private:
    static int count;           // Static counter
    static int maxCount;        // Track peak count
    int objectId;
    string name;
    
public:
    // Constructor - increments count
    ObjectCounter(const string& n = "Unnamed") : name(n) {
        count++;
        objectId = count;
        if (count > maxCount) {
            maxCount = count;
        }
        cout << "✓ Object " << objectId << " ('" << name << "') created" << endl;
        cout << "  Total objects: " << count << " (Peak: " << maxCount << ")" << endl;
    }
    
    // Copy Constructor - also increments
    ObjectCounter(const ObjectCounter& other) {
        name = other.name + " (copy)";
        count++;
        objectId = count;
        if (count > maxCount) {
            maxCount = count;
        }
        cout << "📋 Copy of '" << other.name << "' created as '" << name << "'" << endl;
        cout << "  Total objects: " << count << " (Peak: " << maxCount << ")" << endl;
    }
    
    // Destructor - decrements count
    ~ObjectCounter() {
        cout << "✗ Object " << objectId << " ('" << name << "') destroyed" << endl;
        count--;
        cout << "  Remaining objects: " << count << endl;
    }
    
    // Static methods to access count
    static int getCount() {
        return count;
    }
    
    static int getMaxCount() {
        return maxCount;
    }
    
    void display() {
        cout << "Object ID: " << objectId << ", Name: " << name << endl;
    }
};

// Initialize static members
int ObjectCounter::count = 0;
int ObjectCounter::maxCount = 0;

// Function to demonstrate object passing
void processObject(ObjectCounter obj) {
    cout << "  Processing object inside function" << endl;
    obj.display();
}

ObjectCounter createObject(const string& name) {
    ObjectCounter temp(name);
    cout << "  Returning from createObject" << endl;
    return temp;
}

int main() {
    cout << "=== OBJECT COUNTER DEMONSTRATION ===\\n" << endl;
    
    cout << "Initial count: " << ObjectCounter::getCount() << endl;
    
    cout << "\\n--- Creating objects ---" << endl;
    ObjectCounter obj1("First");
    ObjectCounter obj2("Second");
    ObjectCounter obj3("Third");
    
    cout << "\\n--- Copy constructor ---" << endl;
    ObjectCounter obj4(obj1);
    
    cout << "\\n--- Block scope ---" << endl;
    {
        ObjectCounter obj5("Temporary");
        obj5.display();
        cout << "  Exiting block..." << endl;
    }  // obj5 destroyed here
    
    cout << "\\n--- Pass by value (triggers copy) ---" << endl;
    processObject(obj2);
    
    cout << "\\n--- Return by value (triggers copy) ---" << endl;
    ObjectCounter obj6 = createObject("Created in function");
    
    cout << "\\n--- Array of objects ---" << endl;
    ObjectCounter* arr = new ObjectCounter[3] {
        ObjectCounter("Array1"),
        ObjectCounter("Array2"),
        ObjectCounter("Array3")
    };
    
    cout << "\\n--- Current statistics ---" << endl;
    cout << "Total objects: " << ObjectCounter::getCount() << endl;
    cout << "Peak objects: " << ObjectCounter::getMaxCount() << endl;
    
    cout << "\\n--- Deleting array ---" << endl;
    delete[] arr;
    
    cout << "\\n--- Final statistics ---" << endl;
    cout << "Final count: " << ObjectCounter::getCount() << endl;
    cout << "Peak count: " << ObjectCounter::getMaxCount() << endl;
    
    return 0;
}

/* Sample Output:
=== OBJECT COUNTER DEMONSTRATION ===

Initial count: 0

--- Creating objects ---
✓ Object 1 ('First') created
  Total objects: 1 (Peak: 1)
✓ Object 2 ('Second') created
  Total objects: 2 (Peak: 2)
✓ Object 3 ('Third') created
  Total objects: 3 (Peak: 3)

--- Copy constructor ---
📋 Copy of 'First' created as 'First (copy)'
  Total objects: 4 (Peak: 4)

--- Block scope ---
✓ Object 5 ('Temporary') created
  Total objects: 5 (Peak: 5)
Object ID: 5, Name: Temporary
  Exiting block...
✗ Object 5 ('Temporary') destroyed
  Remaining objects: 4
*/`}
          />
        </QuestionCard>

        {/* Question 13: Constructor/Destructor Order Output */}
        <QuestionCard
          number={13}
          title="Constructor/Destructor Order"
          question="What is the output of the following code related to constructor/destructor order?"
          answer="Constructor order: Base class constructors execute first, then derived class constructors (top-down). Destructor order: Derived class destructors execute first, then base class destructors (bottom-up). For member objects, constructors run before containing class constructor; destructors run after containing class destructor."
          marks={5}
          icon={<Clock className="h-3 w-3" />}
        >
          <CodeBlock
            language="cpp"
            code={`#include <iostream>
using namespace std;

class Member {
    string name;
public:
    Member(const string& n) : name(n) {
        cout << "  Member '" << name << "' constructor" << endl;
    }
    ~Member() {
        cout << "  Member '" << name << "' destructor" << endl;
    }
};

class Base {
    Member m_base;
public:
    Base() : m_base("BaseMember") {
        cout << "Base constructor" << endl;
    }
    virtual ~Base() {
        cout << "Base destructor" << endl;
    }
};

class Derived : public Base {
    Member m_derived;
public:
    Derived() : m_derived("DerivedMember") {
        cout << "Derived constructor" << endl;
    }
    ~Derived() {
        cout << "Derived destructor" << endl;
    }
};

class A {
public:
    A() { cout << "A constructor" << endl; }
    ~A() { cout << "A destructor" << endl; }
};

class B {
public:
    B() { cout << "B constructor" << endl; }
    ~B() { cout << "B destructor" << endl; }
};

class C {
    A a;
    B b;
public:
    C() { cout << "C constructor" << endl; }
    ~C() { cout << "C destructor" << endl; }
};

class D : public B, public A {  // Multiple inheritance
public:
    D() { cout << "D constructor" << endl; }
    ~D() { cout << "D destructor" << endl; }
};

int main() {
    cout << "=== Case 1: Single Inheritance ===" << endl;
    {
        Derived d;
    }
    cout << endl;
    
    cout << "=== Case 2: Member Objects ===" << endl;
    {
        C c;
    }
    cout << endl;
    
    cout << "=== Case 3: Multiple Inheritance ===" << endl;
    {
        D d;
    }
    cout << endl;
    
    cout << "=== Case 4: Virtual Inheritance ===" << endl;
    {
        class Grand { public: Grand() { cout << "Grand" << endl; } ~Grand() { cout << "~Grand" << endl; } };
        class Parent1 : virtual public Grand { public: Parent1() { cout << "Parent1" << endl; } ~Parent1() { cout << "~Parent1" << endl; } };
        class Parent2 : virtual public Grand { public: Parent2() { cout << "Parent2" << endl; } ~Parent2() { cout << "~Parent2" << endl; } };
        class Child : public Parent1, public Parent2 { public: Child() { cout << "Child" << endl; } ~Child() { cout << "~Child" << endl; } };
        Child c;
    }
    
    return 0;
}

/* OUTPUT:

=== Case 1: Single Inheritance ===
  Member 'BaseMember' constructor
Base constructor
  Member 'DerivedMember' constructor
Derived constructor
Derived destructor
  Member 'DerivedMember' destructor
Base destructor
  Member 'BaseMember' destructor

=== Case 2: Member Objects ===
  Member 'BaseMember' constructor
Base constructor
  Member 'DerivedMember' constructor
Derived constructor
Derived destructor
  Member 'DerivedMember' destructor
Base destructor
  Member 'BaseMember' destructor

=== Case 3: Multiple Inheritance ===
B constructor
A constructor
D constructor
D destructor
A destructor
B destructor

=== Case 4: Virtual Inheritance ===
Grand
Parent1
Parent2
Child
~Child
~Parent2
~Parent1
~Grand

KEY RULES:
1. Base classes constructed before derived (top-down)
2. Member objects constructed before containing class (in declaration order)
3. Destructors run in reverse order of constructors (bottom-up)
4. Virtual base classes constructed before non-virtual
*/`}
          />
        </QuestionCard>

        {/* Summary Section */}
        <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q7:</span>
              <span className="text-muted-foreground ml-1">Constructor types: Default, Parameterized, Copy, Move</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q8:</span>
              <span className="text-muted-foreground ml-1">Copy constructor: creates copy; called during pass/return by value</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q9:</span>
              <span className="text-muted-foreground ml-1">C++ destructors deterministic; Java uses GC + try-with-resources</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q10:</span>
              <span className="text-muted-foreground ml-1">Constructor chaining: this(), super() in Java; initializer list in C++</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q11:</span>
              <span className="text-muted-foreground ml-1">Private constructor: Singleton, Utility classes, Factory pattern</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q12:</span>
              <span className="text-muted-foreground ml-1">Static counter in constructor/destructor for object counting</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q13:</span>
              <span className="text-muted-foreground ml-1">Constructor: Base → Members → Derived | Destructor: reverse order</span>
            </div>
          </div>
        </div>

        {/* Marks Distribution */}
        <div className="mt-6 flex justify-between items-center text-sm text-muted-foreground border-t border-border pt-4">
          <span>Total Questions: 7</span>
          <span>Marks per Question: 5</span>
          <span>Total Marks: 35</span>
          <span>Time Suggested: 1.5 hours</span>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="Constructors & Destructors Quiz" />
    </TopicContent>
  );
}