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
  FileQuestion,
  GraduationCap,
  Shield,
  Users,
  Building,
  EyeOff,
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

// Question 1: Class vs Object Example Code
const classVsObjectCode = `// CLASS - The blueprint
class Car {
    String brand;
    String color;
    
    void start() {
        System.out.println(brand + " is starting");
    }
}

// OBJECTS - Instances of the class
public class Main {
    public static void main(String[] args) {
        Car car1 = new Car();  // Object 1
        car1.brand = "Toyota";
        car1.color = "Red";
        
        Car car2 = new Car();  // Object 2
        car2.brand = "Honda";
        car2.color = "Blue";
        
        car1.start();  // Output: Toyota is starting
        car2.start();  // Output: Honda is starting
    }
}`;

// Question 2: Encapsulation Example Code
const encapsulationCode = `// Bank Account with Encapsulation
class BankAccount {
    private double balance;  // Hidden from outside
    
    public BankAccount(double initialBalance) {
        if (initialBalance >= 0) {
            this.balance = initialBalance;
        }
    }
    
    // Controlled access through public methods
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Deposited: $" + amount);
        }
    }
    
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("Withdrawn: $" + amount);
        } else {
            System.out.println("Insufficient balance!");
        }
    }
    
    public double getBalance() {
        return balance;
    }
}

// Usage
BankAccount account = new BankAccount(1000);
account.deposit(500);      // Deposited: $500
account.withdraw(200);     // Withdrawn: $200
System.out.println("Balance: $" + account.getBalance()); // Balance: $1300
// account.balance = 9999; // ERROR! Cannot access private field`;

// Question 3: Bank Account Class Code (Java)
const bankAccountJavaCode = `class BankAccount {
    // Attributes (Private for encapsulation)
    private String accountNo;
    private double balance;
    
    // Constructor
    public BankAccount(String accountNo, double initialBalance) {
        this.accountNo = accountNo;
        this.balance = initialBalance;
    }
    
    // Deposit method
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Deposited: $" + amount);
        } else {
            System.out.println("Invalid deposit amount!");
        }
    }
    
    // Withdraw method
    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println("Withdrawn: $" + amount);
        } else {
            System.out.println("Insufficient balance or invalid amount!");
        }
    }
    
    // Display method
    public void display() {
        System.out.println("Account No: " + accountNo);
        System.out.println("Balance: $" + balance);
    }
}

// Test class
public class Main {
    public static void main(String[] args) {
        BankAccount account = new BankAccount("ACC12345", 5000);
        account.deposit(1000);     // Deposited: $1000
        account.withdraw(500);     // Withdrawn: $500
        account.display();
        // Output: Account No: ACC12345, Balance: $5500
    }
}`;

const bankAccountCppCode = `#include <iostream>
#include <string>
using namespace std;

class BankAccount {
private:
    string accountNo;
    double balance;
    
public:
    BankAccount(string accNo, double initialBalance) {
        accountNo = accNo;
        balance = initialBalance;
    }
    
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            cout << "Deposited: $" << amount << endl;
        }
    }
    
    void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            cout << "Withdrawn: $" << amount << endl;
        } else {
            cout << "Insufficient balance!" << endl;
        }
    }
    
    void display() {
        cout << "Account No: " << accountNo << endl;
        cout << "Balance: $" << balance << endl;
    }
};

int main() {
    BankAccount account("ACC12345", 5000);
    account.deposit(1000);
    account.withdraw(500);
    account.display();
    return 0;
}`;

// Question 4: Student Class Code
const studentClassCode = `class Student {
    // Attributes
    private String name;
    private int roll;
    private char grade;
    
    // Parameterized Constructor
    public Student(String name, int roll, char grade) {
        this.name = name;
        this.roll = roll;
        this.grade = grade;
    }
    
    // Default Constructor
    public Student() {
        this.name = "Unknown";
        this.roll = 0;
        this.grade = 'N';
    }
    
    // Setter methods
    public void setName(String name) { this.name = name; }
    public void setRoll(int roll) { this.roll = roll; }
    public void setGrade(char grade) { this.grade = grade; }
    
    // Getter methods
    public String getName() { return name; }
    public int getRoll() { return roll; }
    public char getGrade() { return grade; }
    
    // Display student details
    public void displayStudent() {
        System.out.println("+-------------------+");
        System.out.println("| Student Details   |");
        System.out.println("+-------------------+");
        System.out.println("| Name  : " + name);
        System.out.println("| Roll  : " + roll);
        System.out.println("| Grade : " + grade);
        System.out.println("+-------------------+");
    }
    
    // Check if student passed
    public boolean isPassed() {
        return grade == 'A' || grade == 'B' || grade == 'C';
    }
}

// Test class
public class Main {
    public static void main(String[] args) {
        Student student1 = new Student("Alice Johnson", 101, 'A');
        Student student2 = new Student("Bob Smith", 102, 'D');
        
        student1.displayStudent();
        System.out.println("Passed: " + student1.isPassed()); // true
        
        student2.displayStudent();
        System.out.println("Passed: " + student2.isPassed()); // false
    }
}`;

// Question 5: Class vs Structure Code
const classVsStructCode = `#include <iostream>
#include <string>
using namespace std;

// STRUCTURE - Members are public by default
struct EmployeeStruct {
    string name;    // public by default
    int id;         // public by default
    
    void display() {
        cout << "Name: " << name << ", ID: " << id << endl;
    }
};

// CLASS - Members are private by default
class EmployeeClass {
    string name;    // private by default
    int id;         // private by default
    
public:
    void setName(string n) { name = n; }
    void setId(int i) { id = i; }
    void display() {
        cout << "Name: " << name << ", ID: " << id << endl;
    }
};

int main() {
    // Structure - direct access allowed
    EmployeeStruct emp1;
    emp1.name = "John";    // Direct access - OK
    emp1.id = 101;         // Direct access - OK
    emp1.display();
    
    // Class - direct access NOT allowed
    EmployeeClass emp2;
    // emp2.name = "Jane";  // ERROR! Private member
    emp2.setName("Jane");  // Must use public method
    emp2.setId(102);
    emp2.display();
    
    return 0;
}`;

// Question 6: Data Hiding Code
const dataHidingCode = `// Demonstrating Data Hiding
class Employee {
    // HIDDEN DATA - Private members
    private String ssn;        // Social Security Number
    private double salary;
    private int loginAttempts;
    
    public Employee(String ssn, double salary) {
        this.ssn = ssn;
        this.salary = salary;
        this.loginAttempts = 0;
    }
    
    // CONTROLLED ACCESS through public methods
    public String getSsn(String password) {
        if (authenticate(password)) {
            return maskSSN(ssn);
        }
        loginAttempts++;
        return "Access Denied!";
    }
    
    public void setSalary(double newSalary, String password) {
        if (authenticate(password)) {
            if (newSalary >= 0) {
                this.salary = newSalary;
                System.out.println("Salary updated");
            }
        } else {
            System.out.println("Unauthorized access attempt!");
        }
    }
    
    // Private helper methods (also hidden)
    private boolean authenticate(String password) {
        return password.equals("secure123");
    }
    
    private String maskSSN(String ssn) {
        return "XXX-XX-" + ssn.substring(7);
    }
    
    public double getSalary() {
        return salary;
    }
}

// Demonstration
Employee emp = new Employee("123-45-6789", 75000);
// emp.salary = 100000;        // ERROR! Cannot access private field
System.out.println(emp.getSalary());              // 75000.0
System.out.println(emp.getSsn("secure123"));      // XXX-XX-6789
emp.setSalary(80000, "secure123");                // Salary updated`;

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
        {/* Question */}
        <div className="mb-4">
          <div className="flex items-start gap-2">
            <FileQuestion className="h-5 w-5 text-primary mt-0.5 shrink-0" />
            <div>
              <span className="text-sm font-medium text-muted-foreground">Question:</span>
              <p className="text-foreground font-medium mt-1">{question}</p>
            </div>
          </div>
        </div>
        
        {/* Answer */}
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
        
        {/* Example/Code */}
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
      {/* Page Header */}
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-foreground mb-3">
          Object Oriented Programming
        </h1>
        <p className="text-muted-foreground text-lg">
          Important 5-Mark Questions with Complete Answers and Examples
        </p>
        <div className="mt-3 flex justify-center gap-2">
          <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm">6 Questions</span>
          <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm">5 Marks Each</span>
          <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm">Total: 30 Marks</span>
        </div>
      </div>

      {/* Question 1: Class vs Object */}
      <QuestionCard
        number={1}
        title="Classes & Objects"
        question="What is a class? What is an object? Explain the difference between them."
        marks={5}
        answer="A class is a blueprint or template that defines the structure and behavior of objects. It encapsulates data (attributes) and methods (functions) that operate on that data. An object is an instance of a class - a concrete entity created from the class blueprint with its own unique state. The key differences are: a class defines the structure (no memory allocation), while an object is an instance (memory allocated); a class is created once, multiple objects can be created from it; the analogy is cookie cutter (class) vs cookies (objects)."
        icon={<BookOpen className="h-3 w-3" />}
      >
        <DifferenceTable
          title="Class vs Object - Key Differences"
          headers={["Feature", "Class", "Object"]}
          rows={[
            ["Definition", "Blueprint/template", "Instance of class"],
            ["Memory", "No memory allocation when defined", "Memory allocated when created"],
            ["Creation", "Defined once using 'class' keyword", "Created multiple times using 'new' keyword"],
            ["Example", "class Car { }", "Car myCar = new Car();"],
            ["Analogy", "Cookie cutter", "Cookie"],
          ]}
        />
        <div className="mt-4">
          <CodeBlock code={classVsObjectCode} language="java" />
        </div>
      </QuestionCard>

      {/* Question 2: Encapsulation */}
      <QuestionCard
        number={2}
        title="Encapsulation"
        question="Explain encapsulation with an example."
        marks={5}
        answer="Encapsulation is the bundling of data (attributes) and methods that operate on that data within a single unit (class), while hiding internal details from outside access. It is achieved using access modifiers (private, public, protected). Key benefits include: Data Hiding (internal state is protected), Controlled Access (access only through methods), Maintainability (internal changes don't affect external code), and Reusability (encapsulated code can be reused)."
        icon={<Shield className="h-3 w-3" />}
      >
        <div className="p-4 bg-muted/30 rounded-lg mb-4">
          <h4 className="font-medium text-foreground mb-2">Benefits of Encapsulation:</h4>
          <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
            <li><strong className="text-foreground">Data Hiding</strong> - Internal state is protected from unauthorized access</li>
            <li><strong className="text-foreground">Controlled Access</strong> - Access only through validated methods</li>
            <li><strong className="text-foreground">Maintainability</strong> - Internal changes don't affect external code</li>
            <li><strong className="text-foreground">Reusability</strong> - Encapsulated code can be reused across projects</li>
            <li><strong className="text-foreground">Flexibility</strong> - Can change internal implementation without breaking external code</li>
          </ul>
        </div>
        <CodeBlock code={encapsulationCode} language="java" />
      </QuestionCard>

      {/* Question 3: Bank Account Class */}
      <QuestionCard
        number={3}
        title="Bank Account Implementation"
        question="Write a C++/Java class for a Bank Account with attributes (accountNo, balance) and methods (deposit, withdraw, display)."
        marks={5}
        answer="The BankAccount class should have private attributes accountNo (String) and balance (double) to demonstrate encapsulation. The constructor initializes these attributes. The deposit() method adds a positive amount to the balance with validation. The withdraw() method subtracts the amount only if sufficient balance exists. The display() method prints account details. All methods include proper validation checks for security."
        icon={<Building className="h-3 w-3" />}
      >
        <div className="mb-4">
          <div className="flex gap-2 mb-2">
            <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">Java</span>
            <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">C++</span>
            <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">Encapsulation</span>
          </div>
        </div>
        <MultiLanguageCode
          codes={[
            { language: "java", label: "Java", code: bankAccountJavaCode },
            { language: "cpp", label: "C++", code: bankAccountCppCode },
          ]}
        />
      </QuestionCard>

      {/* Question 4: Student Class */}
      <QuestionCard
        number={4}
        title="Student Class Implementation"
        question="Write a class to represent a Student with name, roll, grade."
        marks={5}
        answer="The Student class encapsulates student information including name (String), roll number (int), and grade (char). It includes multiple constructors (parameterized and default), getter and setter methods for controlled access, a displayStudent() method to print formatted student details, and an isPassed() method to check if the student has passing grades (A, B, or C). This demonstrates proper encapsulation and object-oriented design principles."
        icon={<GraduationCap className="h-3 w-3" />}
      >
        <div className="bg-muted/30 p-3 rounded-lg mb-4">
          <h4 className="font-medium text-foreground text-sm mb-2">Sample Output:</h4>
          <pre className="text-xs text-muted-foreground bg-background p-2 rounded">
{`+-------------------+
| Student Details   |
+-------------------+
| Name  : Alice Johnson
| Roll  : 101
| Grade : A
+-------------------+
Passed: true

+-------------------+
| Student Details   |
+-------------------+
| Name  : Bob Smith
| Roll  : 102
| Grade : D
+-------------------+
Passed: false`}
          </pre>
        </div>
        <CodeBlock code={studentClassCode} language="java" />
      </QuestionCard>

      {/* Question 5: Class vs Structure in C++ */}
      <QuestionCard
        number={5}
        title="Class vs Structure in C++"
        question="What is the difference between a class and a structure in C++?"
        marks={5}
        answer="In C++, the main difference between a class and a structure is the default access specifier. Structure members are public by default, while class members are private by default. Additionally, inheritance default is public for structures and private for classes. Structures are typically used for simple data containers, while classes are used for complex objects with behavior. Both can have constructors, destructors, and member functions, but classes are preferred for proper encapsulation and OOP principles."
        icon={<Users className="h-3 w-3" />}
      >
        <DifferenceTable
          title="Class vs Structure - Detailed Comparison"
          headers={["Feature", "Class", "Structure (struct)"]}
          rows={[
            ["Default Access", "private", "public"],
            ["Inheritance Default", "private inheritance", "public inheritance"],
            ["Typical Usage", "Complex objects with behavior", "Simple data containers"],
            ["Member Functions", "Commonly used", "Less common"],
            ["Constructor/Destructor", "Always used", "Optional"],
            ["OOP Features", "Fully supports", "Limited support"],
          ]}
        />
        <div className="mt-4">
          <h4 className="font-medium text-foreground mb-2">When to Use What?</h4>
          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div className="p-3 bg-muted/30 rounded-lg">
              <span className="font-medium text-foreground">Use Structure When:</span>
              <ul className="mt-1 space-y-0.5 text-muted-foreground list-disc list-inside">
                <li>Only storing passive data</li>
                <li>C compatibility needed</li>
                <li>Simple data aggregates</li>
                <li>Public access is desired</li>
              </ul>
            </div>
            <div className="p-3 bg-muted/30 rounded-lg">
              <span className="font-medium text-foreground">Use Class When:</span>
              <ul className="mt-1 space-y-0.5 text-muted-foreground list-disc list-inside">
                <li>Need data hiding/encapsulation</li>
                <li>Implementing complex behavior</li>
                <li>Following OOP principles</li>
                <li>Need controlled access</li>
              </ul>
            </div>
          </div>
        </div>
        <CodeBlock code={classVsStructCode} language="cpp" />
      </QuestionCard>

      {/* Question 6: Data Hiding */}
      <QuestionCard
        number={6}
        title="Data Hiding"
        question="Explain the concept of data hiding. How is it achieved in OOP?"
        marks={5}
        answer="Data Hiding is an OOP principle that restricts direct access to an object's internal data, preventing unauthorized or accidental modification. Only the object's own methods can modify its internal state. Data Hiding is achieved through: (1) Access Modifiers - using 'private' and 'protected' keywords to restrict access, (2) Public Interface Methods - providing controlled access through getters and setters with validation, and (3) Encapsulation - bundling data with methods that operate on that data. This provides security, validation, data integrity, and maintainability."
        icon={<EyeOff className="h-3 w-3" />}
      >
        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="p-3 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-2">How Data Hiding is Achieved:</h4>
            <ol className="space-y-1 text-sm text-muted-foreground list-decimal list-inside">
              <li><strong className="text-foreground">Access Modifiers</strong> - private, protected</li>
              <li><strong className="text-foreground">Public Interface</strong> - Getters and Setters</li>
              <li><strong className="text-foreground">Encapsulation</strong> - Bundling data with methods</li>
              <li><strong className="text-foreground">Validation Logic</strong> - Inside setter methods</li>
              <li><strong className="text-foreground">Private Helpers</strong> - Internal helper methods</li>
            </ol>
          </div>
          <div className="p-3 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-2">Benefits Demonstrated:</h4>
            <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
              <li><strong>Security</strong> - Sensitive data protected</li>
              <li><strong>Validation</strong> - Invalid values prevented</li>
              <li><strong>Data Integrity</strong> - Consistent state maintained</li>
              <li><strong>Audit Trail</strong> - Access can be tracked</li>
              <li><strong>Flexibility</strong> - Internal changes without breaking code</li>
            </ul>
          </div>
        </div>
        
        {/* ASCII Diagram for Data Hiding */}
        <div className="bg-muted/30 p-3 rounded-lg mb-4 font-mono text-xs">
          <pre className="text-muted-foreground">
{`┌─────────────────────────────────┐
│           EMPLOYEE CLASS        │
│  ┌───────────────────────────┐  │
│  │      PUBLIC INTERFACE      │  │  ← Access through methods only
│  │  getSsn()  getSalary()     │  │
│  │  setSalary()               │  │
│  └───────────────────────────┘  │
│         ▲                        │
│         │ (Controlled Access)    │
│         │                        │
│  ┌───────────────────────────┐  │
│  │       HIDDEN DATA          │  │  ← Cannot access directly
│  │  - private String ssn      │  │
│  │  - private double salary   │  │
│  │  - private int loginAttempts│  │
│  └───────────────────────────┘  │
│         ▲                        │
│         │ (Internal use only)    │
│  ┌───────────────────────────┐  │
│  │     PRIVATE METHODS        │  │
│  │  authenticate() maskSSN()  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘`}
          </pre>
        </div>
        
        <CodeBlock code={dataHidingCode} language="java" />
      </QuestionCard>

      {/* Summary Section */}
      <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
        <div className="flex items-center gap-3 mb-4">
          <Target className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="p-3 bg-background rounded-lg">
            <span className="font-bold text-primary">Q1:</span>
            <span className="text-sm text-muted-foreground ml-2">Class = Blueprint, Object = Instance</span>
          </div>
          <div className="p-3 bg-background rounded-lg">
            <span className="font-bold text-primary">Q2:</span>
            <span className="text-sm text-muted-foreground ml-2">Encapsulation = Data + Methods + Hiding</span>
          </div>
          <div className="p-3 bg-background rounded-lg">
            <span className="font-bold text-primary">Q3:</span>
            <span className="text-sm text-muted-foreground ml-2">BankAccount = deposit() + withdraw() + display()</span>
          </div>
          <div className="p-3 bg-background rounded-lg">
            <span className="font-bold text-primary">Q4:</span>
            <span className="text-sm text-muted-foreground ml-2">Student = name + roll + grade + isPassed()</span>
          </div>
          <div className="p-3 bg-background rounded-lg">
            <span className="font-bold text-primary">Q5:</span>
            <span className="text-sm text-muted-foreground ml-2">Class(private) vs Struct(public) default access</span>
          </div>
          <div className="p-3 bg-background rounded-lg">
            <span className="font-bold text-primary">Q6:</span>
            <span className="text-sm text-muted-foreground ml-2">Data Hiding = private members + public methods</span>
          </div>
        </div>
      </div>

      {/* Marks Distribution */}
      <div className="mt-6 flex justify-between items-center text-sm text-muted-foreground border-t border-border pt-4">
        <span>Total Questions: 6</span>
        <span>Marks per Question: 5</span>
        <span>Total Marks: 30</span>
        <span>Time Suggested: 1.5 hours</span>
      </div>

      {/* What are Classes & Objects? - Tutorial Section */}
      <section className="mb-12 mt-12">
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
            <span className="text-green-500 font-bold">A:</span>
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