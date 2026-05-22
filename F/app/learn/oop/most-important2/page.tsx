"use client";

import { useState } from "react";

const chapters = [
  {
    id: "exceptions",
    label: "Exception Handling",
    icon: "⚡",
    color: "#ff6b6b",
    topics: [
      {
        id: 1,
        title: "try, catch, throw, finally",
        marks: 6,
        tag: "Very Important",
        content: `**Definition:** Exception handling is a mechanism to handle runtime errors gracefully without crashing the program. It separates error-handling code from normal code.

**Key Keywords:**

| Keyword | Purpose |
|---------|---------|
| try | Block that contains code which may throw an exception |
| throw | Used to explicitly throw an exception |
| catch | Block that handles the thrown exception |
| finally | (Java only) Always executes regardless of exception |

**C++ Example:**
\`\`\`cpp
double divide(int numerator, int denominator) {
    if (denominator == 0) {
        throw "Division by zero error!";   // throw exception
    }
    return (double)numerator / denominator;
}

int main() {
    try {                                    // try block
        double result = divide(10, 0);
        cout << "Result: " << result << endl;
    }
    catch (const char* message) {            // catch block
        cout << "Exception caught: " << message << endl;
    }
    cout << "Program continues normally..." << endl;
}
\`\`\`

**Java Example with finally:**
\`\`\`java
public class ExceptionDemo {
    public static void main(String[] args) {
        try {
            int[] arr = new int[5];
            arr[10] = 100;  // ArrayIndexOutOfBoundsException
        }
        catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Exception: " + e);
        }
        finally {
            System.out.println("Finally block always executes");
        }
        System.out.println("Program continues");
    }
}
\`\`\`

**Output:**
\`\`\`text
Exception: java.lang.ArrayIndexOutOfBoundsException: Index 10 out of bounds
Finally block always executes
Program continues
\`\`\``,
      },
      {
        id: 2,
        title: "Multiple Catch + Checked vs Unchecked",
        marks: 5,
        tag: "Important",
        content: `**Multiple Catch Blocks:** Handle different exception types differently. Order matters — catch more specific exceptions FIRST.

\`\`\`java
class MultipleCatchDemo {
    public static void main(String[] args) {
        try {
            String str = null;
            int[] arr = {1, 2, 3};
            System.out.println(arr[5]);      // ArrayIndexOutOfBoundsException
            System.out.println(str.length()); // NullPointerException
        }
        catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Array index error: " + e);
        }
        catch (NullPointerException e) {
            System.out.println("Null pointer error: " + e);
        }
        catch (Exception e) {               // Generic catch — must be LAST
            System.out.println("Some other error: " + e);
        }
    }
}
\`\`\`

**Checked vs Unchecked Exceptions:**

| Feature | Checked | Unchecked |
|---------|---------|-----------|
| Compile-time checking | Yes — must handle or declare | No — optional |
| Also called | Compile-time exceptions | Runtime exceptions |
| Parent class | Exception (except RuntimeException) | RuntimeException |
| Examples | IOException, SQLException | NullPointerException, ArithmeticException |
| Handling required | Yes (try-catch or throws) | No |

**throws keyword:**
\`\`\`java
class ThrowsDemo {
    public void readFile(String filename) throws IOException {
        FileReader file = new FileReader(filename);  // Checked exception
    }
    public static void main(String[] args) {
        ThrowsDemo obj = new ThrowsDemo();
        try {
            obj.readFile("data.txt");
        } catch (IOException e) {
            System.out.println("Handled in main");
        }
    }
}
\`\`\``,
      },
      {
        id: 3,
        title: "Custom Exception (User-defined)",
        marks: 5,
        tag: "Program",
        content: `**Concept:** Create your own exception class by extending \`Exception\` (checked) or \`RuntimeException\` (unchecked).

\`\`\`java
// Step 1: Create custom exception class
class InsufficientBalanceException extends Exception {
    private double deficit;
    
    public InsufficientBalanceException(double required, double available) {
        super("Insufficient balance! Required: " + required + ", Available: " + available);
        this.deficit = required - available;
    }
    
    public double getDeficit() { return deficit; }
}

// Step 2: Use it in a class
class BankAccount {
    private double balance;
    
    public BankAccount(double initialBalance) { this.balance = initialBalance; }
    
    public void withdraw(double amount) throws InsufficientBalanceException {
        if (amount > balance) {
            throw new InsufficientBalanceException(amount, balance);
        }
        balance -= amount;
        System.out.println("Withdrawn: " + amount + ", Balance: " + balance);
    }
}

// Step 3: Test it
public class CustomExceptionDemo {
    public static void main(String[] args) {
        BankAccount account = new BankAccount(5000);
        try {
            account.withdraw(2000);  // OK
            account.withdraw(4000);  // Throws!
        }
        catch (InsufficientBalanceException e) {
            System.out.println("Exception: " + e.getMessage());
            System.out.println("Deficit: " + e.getDeficit());
        }
    }
}
\`\`\`

**Output:**
\`\`\`text
Withdrawn: 2000.0, Balance: 3000.0
Exception: Insufficient balance! Required: 4000.0, Available: 3000.0
Deficit: 1000.0
\`\`\`

**C++ version:**
\`\`\`cpp
class InsufficientBalanceException : public exception {
    string message;
public:
    InsufficientBalanceException(double req, double avail) {
        message = "Need " + to_string(req) + " but only " + to_string(avail);
    }
    const char* what() const noexcept override {
        return message.c_str();
    }
};
\`\`\``,
      },
    ],
  },
  {
    id: "abstract",
    label: "Abstract & Interface",
    icon: "◎",
    color: "#38bdf8",
    topics: [
      {
        id: 4,
        title: "Abstract Class vs Interface",
        marks: 6,
        tag: "Most Asked",
        content: `**Abstract Class:** Cannot be instantiated; may contain abstract (unimplemented) methods.
**Interface:** A contract — defines WHAT a class should do, not HOW.

| Feature | Abstract Class | Interface |
|---------|---------------|-----------|
| Keyword | abstract class | interface |
| Multiple inheritance | No (extend only one) | Yes (implement multiple) |
| Instance variables | Can have | Only public static final |
| Constructors | Can have | Cannot have |
| Method implementation | Abstract + concrete | Abstract; Java 8+: default/static |
| Access modifiers | All (public, protected, private) | Only public (implicitly) |
| When to use | "is-a" with common code | "can-do" capability |

\`\`\`java
// ABSTRACT CLASS
abstract class Vehicle {
    protected String brand;
    public Vehicle(String brand) { this.brand = brand; }  // Constructor OK
    public abstract void start();        // Abstract method
    public void displayBrand() {         // Concrete method
        System.out.println("Brand: " + brand);
    }
}

// INTERFACES
interface Electric {
    void charge();                       // Abstract
    default void batteryStatus() {       // Java 8+ default
        System.out.println("Battery: 100%");
    }
}

interface GPS {
    void navigate(String destination);
}

// Using BOTH — extends one abstract, implements multiple interfaces
class ElectricCar extends Vehicle implements Electric, GPS {
    public ElectricCar(String brand) { super(brand); }
    public void start() { System.out.println("Started silently"); }
    public void charge() { System.out.println("Charging at 50kW"); }
    public void navigate(String dest) { System.out.println("Navigating to " + dest); }
}
\`\`\`

**When to use:**

| Use Abstract Class When | Use Interface When |
|------------------------|-------------------|
| You have common code to share | You need multiple inheritance |
| You want to define a template | You want to define a capability |
| You need non-final variables | Unrelated classes share behavior |`,
      },
      {
        id: 5,
        title: "Interface — Multiple Inheritance Solution",
        marks: 5,
        tag: "Important",
        content: `**Why Java needs interfaces:**
- Multiple inheritance of type (classes can't extend multiple parents)
- Loose coupling (code depends on interface, not concrete implementation)
- Polymorphism across unrelated class hierarchies
- API design — defines contracts without details
- Dependency injection for testable code

**The Problem:**
\`\`\`java
// NOT allowed in Java:
class AllInOne extends Printer, Scanner { }  // COMPILE ERROR
\`\`\`

**Solution using Interfaces:**
\`\`\`java
interface Printable {
    void print();
    default void printColor() { System.out.println("Printing in color"); }
}
interface Scannable {
    void scan();
}
interface Faxable {
    void fax();
}

class AllInOnePrinter implements Printable, Scannable, Faxable {
    public void print() { System.out.println("Printing document"); }
    public void scan()  { System.out.println("Scanning document"); }
    public void fax()   { System.out.println("Faxing document"); }
}

class Smartphone implements Printable, Scannable {
    public void print() { System.out.println("Sending to wireless printer"); }
    public void scan()  { System.out.println("Using camera as scanner"); }
}

// Interface polymorphism
Printable[] devices = {new AllInOnePrinter(), new Smartphone()};
for (Printable d : devices) { d.print(); }
\`\`\`

**Resolving default method conflicts (Java 8+):**
\`\`\`java
interface A { default void show() { System.out.println("A"); } }
interface B { default void show() { System.out.println("B"); } }

class C implements A, B {
    public void show() {
        A.super.show();   // Must override to resolve
        System.out.println("C's implementation");
    }
}
\`\`\`

**Why this solves the diamond problem:**

| Problem | Interface Solution |
|---------|-------------------|
| Diamond problem | No state in interfaces → no ambiguity |
| Multiple parents | Interfaces have no constructors/state |
| Method name conflict | Class must override to resolve |`,
      },
      {
        id: 6,
        title: "Pure Virtual Function (C++ ↔ Abstract)",
        marks: 5,
        tag: "C++ Connection",
        content: `**Pure Virtual Function:** Declared with \`= 0\`, no implementation in base class.
\`\`\`cpp
virtual returnType functionName(parameters) = 0;
\`\`\`

A class with at least one pure virtual function becomes **abstract** — cannot instantiate.

\`\`\`cpp
class Shape {                          // ABSTRACT CLASS
public:
    virtual double area() = 0;         // Pure virtual
    virtual double perimeter() = 0;    // Pure virtual
    virtual void draw() = 0;           // Pure virtual
    void display() {                   // Regular concrete function
        cout << "This is a shape" << endl;
    }
    virtual ~Shape() {}                // Virtual destructor
};

class Circle : public Shape {          // CONCRETE CLASS
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area() override { return 3.14159 * radius * radius; }
    double perimeter() override { return 2 * 3.14159 * radius; }
    void draw() override { cout << "Drawing circle of radius " << radius << endl; }
};

int main() {
    // Shape s;    // ERROR! Cannot instantiate abstract class
    Shape* shapes[2];
    shapes[0] = new Circle(5);
    shapes[1] = new Rectangle(4, 6);
    
    for (int i = 0; i < 2; i++) {
        shapes[i]->draw();
        cout << "Area: " << shapes[i]->area() << endl;
        delete shapes[i];
    }
}
\`\`\`

**C++ Abstract Class vs Java Interface:**

| C++ Abstract Class | Java Interface |
|-------------------|---------------|
| Can have data members | Only constants (public static final) |
| Can have constructors | No constructors |
| Multiple inheritance possible | Multiple implementation possible |
| Pure virtual = 0 | Abstract method (implicitly) |`,
      },
    ],
  },
  {
    id: "cpp-special",
    label: "C++ Special",
    icon: "⬡",
    color: "#a78bfa",
    topics: [
      {
        id: 7,
        title: "Friend Function",
        marks: 5,
        tag: "Important",
        content: `**Definition:** A friend function is NOT a member of the class, but has access to the class's private and protected members. Declared inside the class using \`friend\`.

**Why needed:**
- Operator overloading when left operand is not a class object
- Access private members of two different classes
- Avoid getter/setter overhead in performance-critical code

**Properties:**

| Property | Details |
|----------|---------|
| Not a member | Called like a normal function |
| Access specifier | Irrelevant where declared (public/private same) |
| Inheritance | NOT inherited |
| Symmetry | A is friend of B ≠ B is friend of A |

\`\`\`cpp
class Employee {
private:
    int employeeId;
    double salary;
    string name;
public:
    Employee(int id, double sal, string n) : employeeId(id), salary(sal), name(n) {}
    
    // Declare friend function
    friend void displayEmployeeDetails(Employee e);
    
    // Declare friend class
    friend class HR;
};

// Friend function definition (NOT Employee::)
void displayEmployeeDetails(Employee e) {
    cout << "Name: " << e.name << endl;     // Can access private!
    cout << "ID: " << e.employeeId << endl;
    cout << "Salary: " << e.salary << endl;
}

class HR {
public:
    void giveBonus(Employee& e, double bonus) {
        e.salary += bonus;    // HR is friend class — full access
        cout << "Bonus given. New salary: " << e.salary << endl;
    }
};

int main() {
    Employee emp(101, 50000, "Rahim");
    displayEmployeeDetails(emp);    // Friend function call (no object needed)
    HR hr;
    hr.giveBonus(emp, 5000);
}
\`\`\`

**Member vs Friend Function:**

| Aspect | Member Function | Friend Function |
|--------|----------------|-----------------|
| Called using | obj.func() | func(obj) |
| this pointer | Has it | Does NOT have it |
| Inheritance | Inherited | Not inherited |
| Scope | Class scope | Global/namespace |`,
      },
      {
        id: 8,
        title: "this Pointer",
        marks: 5,
        tag: "Concept",
        content: `**Definition:** The \`this\` pointer is an implicit parameter in all non-static member functions. It holds the memory address of the object the function was called on.

**Properties:**
- Available only inside non-static member functions
- A const pointer: \`ClassName* const this\` (cannot change where it points)
- Automatically passed by compiler
- NOT available in static functions

**Key Uses:**

| Use | Explanation |
|-----|-------------|
| Resolve name conflicts | When parameter name == member name |
| Method chaining | Return \`*this\` for fluent interface |
| Pass current object | \`someFunc(*this)\` |
| Self-assignment check | \`if (this != &other)\` |

\`\`\`cpp
class Student {
    string name;
    int roll;
    double marks;
public:
    // Use this to resolve name conflict
    Student(string name, int roll, double marks) {
        this->name = name;     // this->name = member
        this->roll = roll;     // roll = parameter
        this->marks = marks;
    }
    
    // Method chaining — return *this
    Student& setName(string name) {
        this->name = name;
        return *this;          // Return reference to current object
    }
    Student& setRoll(int roll)     { this->roll = roll; return *this; }
    Student& setMarks(double marks){ this->marks = marks; return *this; }
    
    // Check if two pointers refer to same object
    bool isSameObject(Student& other) {
        return this == &other;
    }
    
    // Assignment operator: self-assignment guard
    Student& operator=(const Student& other) {
        if (this != &other) {  // Prevent self-assignment
            this->name = other.name;
            this->roll = other.roll;
        }
        return *this;
    }
};

int main() {
    Student s1("Rahim", 101, 85.5);
    
    // Method chaining with this
    s1.setName("Rahim Ahmed").setRoll(201).setMarks(92.5);
    
    Student s2("Karim", 102, 90.0);
    cout << s1.isSameObject(s1) << endl;  // 1 (same object)
    cout << s1.isSameObject(s2) << endl;  // 0 (different)
}
\`\`\`

**Memory Diagram:**
\`\`\`text
Object s1
┌─────────────┐
│ name        │ ← this points here when s1.func() called
│ roll        │
│ marks       │
└─────────────┘
this = address of s1
*this = s1 itself
\`\`\``,
      },
      {
        id: 9,
        title: "Virtual Destructor",
        marks: 5,
        tag: "⚡ BUET Favorite",
        content: `**Definition:** A destructor with the \`virtual\` keyword in base class. Ensures the derived class destructor is called when a derived object is deleted through a base class pointer.

**Problem WITHOUT virtual destructor:**
\`\`\`cpp
class Base {
public:
    ~Base() { cout << "Base destructor\\n"; }  // NON-VIRTUAL
};
class Derived : public Base {
    int* data;
public:
    Derived() { data = new int[1000]; }
    ~Derived() { delete[] data; cout << "Derived destructor\\n"; }
};

Base* ptr = new Derived();
delete ptr;  // ONLY Base destructor runs! Memory leak!
// Output: Base destructor
// 1000 ints NEVER freed!
\`\`\`

**Solution WITH virtual destructor:**
\`\`\`cpp
class Base {
public:
    virtual ~Base() { cout << "Base destructor\\n"; }  // VIRTUAL!
};
class Derived : public Base {
    int* data;
public:
    Derived() { data = new int[1000]; }
    ~Derived() override { delete[] data; cout << "Derived destructor\\n"; }
};

Base* ptr = new Derived();
delete ptr;  // Both destructors run correctly!
// Output:
// Derived destructor   ← called first
// Base destructor
// No memory leak!
\`\`\`

**Virtual Destructor Rules:**

| Rule | Explanation |
|------|-------------|
| Always make base destructor virtual | If class has any virtual function |
| Derived destructor auto-virtual | When base destructor is virtual |
| Pure virtual destructor | Legal, but MUST provide body |
| Cost | Adds vptr overhead (4-8 bytes per object) |

**Pure virtual destructor (special case):**
\`\`\`cpp
class AbstractBase {
public:
    virtual ~AbstractBase() = 0;    // Pure virtual destructor
};
AbstractBase::~AbstractBase() { }  // MUST provide body!
\`\`\`

**Rule:** If your class has ANY virtual function, make its destructor virtual.`,
      },
      {
        id: 10,
        title: "Deep Copy vs Shallow Copy",
        marks: 5,
        tag: "Very Important",
        content: `| Type | Description | Problem |
|------|-------------|---------|
| Shallow Copy | Copies only memory addresses — both objects point to same memory | Double deletion, dangling pointers |
| Deep Copy | Allocates NEW memory and copies actual data — each object independent | No such issue |

**Problem with Shallow Copy:**
\`\`\`cpp
class ShallowString {
    char* data;
    int length;
public:
    ShallowString(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }
    // Default copy constructor (shallow) — DANGEROUS!
    ~ShallowString() { delete[] data; }  // BOTH objects delete same memory → CRASH!
};

ShallowString s1("Hello");
ShallowString s2 = s1;  // s1 and s2 share same memory!
// When both destructors run: DOUBLE DELETION → CRASH
\`\`\`

**Solution — Deep Copy:**
\`\`\`cpp
class DeepString {
    char* data;
    int length;
public:
    DeepString(const char* str) {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }
    
    // DEEP COPY CONSTRUCTOR
    DeepString(const DeepString& other) {
        length = other.length;
        data = new char[length + 1];   // NEW memory allocation
        strcpy(data, other.data);       // Copy actual data
    }
    
    // DEEP COPY ASSIGNMENT OPERATOR
    DeepString& operator=(const DeepString& other) {
        if (this != &other) {           // Self-assignment check
            delete[] data;              // Free old memory
            length = other.length;
            data = new char[length + 1];
            strcpy(data, other.data);
        }
        return *this;
    }
    
    ~DeepString() { delete[] data; }
};

DeepString s1("Hello");
DeepString s2 = s1;  // Deep copy — s2 gets its OWN "Hello"
s1.modify("World");  // s1 = "World", s2 = "Hello" (unaffected)
\`\`\`

**Rule of Three:** If you need a custom destructor, copy constructor, or copy assignment operator, you likely need all three.

| When to use Deep Copy |
|----------------------|
| Class has pointer members |
| Class manages dynamic memory |
| Class manages resources (files, sockets) |`,
      },
    ],
  },
  {
    id: "java-special",
    label: "Java Special",
    icon: "◈",
    color: "#34d399",
    topics: [
      {
        id: 11,
        title: "String Immutability in Java",
        marks: 5,
        tag: "Important",
        content: `**Definition:** Once a String object is created, its value cannot be changed. Any "modification" creates a new String object.

**Why immutable? (5 reasons):**

| Reason | Explanation |
|--------|-------------|
| String Pool | Allows safe sharing of literals |
| Security | Class loading, file paths can't be changed by malicious code |
| Thread Safety | Immutable objects are automatically thread-safe |
| Caching | Hashcode computed once and cached |
| Class Loading | String is used in class loading — must be immutable |

\`\`\`java
String s1 = "Hello";
System.out.println(System.identityHashCode(s1));  // e.g., 12345678

s1 = s1 + " World";  // Creates NEW String — doesn't modify old one
System.out.println(System.identityHashCode(s1));  // Different hashcode!

// String methods return NEW Strings
String s2 = "Java Programming";
s2.toUpperCase();    // Returns new string, s2 unchanged!
System.out.println(s2);  // "Java Programming" (original)
\`\`\`

**String Pool Demonstration:**
\`\`\`java
String str1 = "Hello";            // Literal — goes to pool
String str2 = "Hello";            // Reuses from pool (same object)
String str3 = new String("Hello"); // Creates new object OUTSIDE pool

System.out.println(str1 == str2);     // true  (same pool object)
System.out.println(str1 == str3);     // false (different objects)
System.out.println(str1.equals(str3));// true  (same content)
String str4 = str3.intern();
System.out.println(str1 == str4);     // true  (intern() returns pool version)
\`\`\`

**Immutability Cheat Sheet:**

| Operation | New Object Created? |
|-----------|-------------------|
| s = s + "x" | YES |
| s.toUpperCase() | YES |
| s.replace('a','b') | YES |
| s.substring(1,3) | YES |
| s.trim() | YES (if changed) |
| s1.equals(s2) | NO (returns boolean) |

**Mutable alternative — StringBuilder:**
\`\`\`java
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");   // Modifies SAME object — no new allocation
System.out.println(sb); // "Hello World"
\`\`\``,
      },
      {
        id: 12,
        title: "== vs .equals() in Java",
        marks: 5,
        tag: "⚡ Exam Common",
        content: `| Operator | Compares | Used for |
|----------|---------|---------|
| == | Reference (memory address) | Primitives; checking if same object |
| .equals() | Content (value/state) | Logical equality of objects |

**Complete Output Question:**
\`\`\`java
String s1 = "Java";
String s2 = "Java";
String s3 = new String("Java");
String s4 = s3.intern();

System.out.println(s1 == s2);      // true  (same pool object)
System.out.println(s1 == s3);      // false (new String() = heap object)
System.out.println(s1 == s4);      // true  (intern() returns pool ref)
System.out.println(s4 == s3);      // false (s4=pool, s3=heap)

// Runtime concatenation vs compile-time
String a = "Hello", b = "World";
String c = "HelloWorld";
String d = a + b;            // Runtime — new object
String e = "Hello" + "World"; // Compile-time constant — pool

System.out.println(c == d);  // false (d is new runtime object)
System.out.println(c == e);  // true  (compile-time becomes pool literal)
System.out.println(c.equals(d)); // true (content same)

// substring() creates new string
String str = "Programming";
String sub = str.substring(0, 4);
System.out.println(sub == "Prog");      // false (new object)
System.out.println(sub.equals("Prog")); // true
\`\`\`

**Null and StringBuilder:**
\`\`\`java
String nullStr = null;
String emptyStr = "";

System.out.println(emptyStr == "");         // true (pool literal)
System.out.println("Hello".equals(nullStr));// false (null-safe pattern)
System.out.println(nullStr == null);        // true

StringBuilder sb1 = new StringBuilder("Test");
StringBuilder sb2 = new StringBuilder("Test");
System.out.println(sb1 == sb2);             // false
System.out.println(sb1.equals(sb2));        // false! (not overridden)
System.out.println(sb1.toString().equals(sb2.toString())); // true
\`\`\`

**Wrapper Class Caching:**
\`\`\`java
Integer i1 = 127;  Integer i2 = 127;
Integer i3 = 128;  Integer i4 = 128;

System.out.println(i1 == i2);       // true  (cached -128 to 127)
System.out.println(i3 == i4);       // false (beyond cache range)
System.out.println(i3.equals(i4));  // true
\`\`\`

**Best Practice:**
- Primitives → use \`==\`
- Strings → use \`.equals()\`
- Custom objects → override \`.equals()\` and \`.hashCode()\`
- Null-safe → use \`Objects.equals(a, b)\``,
      },
      {
        id: 13,
        title: "final keyword — All Three Uses",
        marks: 5,
        tag: "Concept",
        content: `**The \`final\` keyword restricts modification. Meaning varies by context.**

**1. final variable — constant value:**
\`\`\`java
class FinalVariableDemo {
    final double PI = 3.14159;                   // Compile-time constant
    public static final int MAX_USERS = 100;     // Common pattern
    final int BLANK_FINAL;                       // Must init in constructor
    
    FinalVariableDemo(int value) {
        BLANK_FINAL = value;    // Initialized here
    }
    void demo() {
        // PI = 3.14;  // ERROR! Cannot reassign final
    }
}
\`\`\`

**2. final method — cannot be overridden:**
\`\`\`java
class Parent {
    public final void cannotOverride() {
        System.out.println("This cannot be overridden");
    }
    public void canOverride() {
        System.out.println("This can be overridden");
    }
}
class Child extends Parent {
    // public void cannotOverride() { } // ERROR!
    @Override
    public void canOverride() {  // OK — not final
        System.out.println("Child's version");
    }
}
\`\`\`

**3. final class — cannot be extended:**
\`\`\`java
final class ImmutableClass {
    private int value;
    ImmutableClass(int v) { value = v; }
    int getValue() { return value; }
}
// class Sub extends ImmutableClass { }  // ERROR! Cannot extend final class
\`\`\`

**final with objects — tricky:**
\`\`\`java
final StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");      // OK — object content can change
// sb = new StringBuilder(); // ERROR! Cannot reassign reference

final int[] arr = {1, 2, 3};
arr[0] = 10;              // OK — array content changes
// arr = new int[5];       // ERROR! Cannot change reference
\`\`\`

**Comparison Table:**

| Context | Meaning | Can Change? |
|---------|---------|-------------|
| final primitive variable | Value cannot change | No |
| final reference variable | Reference cannot change | Ref: No; Object: Yes |
| final method | Cannot be overridden | N/A |
| final class | Cannot be inherited | N/A |

**Benefits:** Security, performance (compiler inlining), thread safety (immutable objects), design clarity.`,
      },
      {
        id: 14,
        title: "static block vs instance block",
        marks: 5,
        tag: "Tricky",
        content: `| Block | Executed | Purpose |
|-------|----------|---------|
| static { } | Once when class first loaded | Initialize static members, load drivers |
| { } (instance) | Every time an object is created (before constructor) | Common init for all constructors |

**Execution Order:** Static blocks (once) → Instance blocks → Constructor (every object)

\`\`\`java
class Parent {
    static { System.out.println("Parent: Static block 1"); }
    { System.out.println("Parent: Instance block 1"); }
    Parent() { System.out.println("Parent: Constructor"); }
    static { System.out.println("Parent: Static block 2"); }
    { System.out.println("Parent: Instance block 2"); }
}

class Child extends Parent {
    static { System.out.println("Child: Static block 1"); }
    { System.out.println("Child: Instance block 1"); }
    Child() { System.out.println("Child: Constructor"); }
    static { System.out.println("Child: Static block 2"); }
    { System.out.println("Child: Instance block 2"); }
}

public class BlockDemo {
    public static void main(String[] args) {
        System.out.println("\\n=== First Child object ===");
        Child c1 = new Child();
        System.out.println("\\n=== Second Child object ===");
        Child c2 = new Child();
    }
}
\`\`\`

**Output:**
\`\`\`text
=== First Child object ===
Parent: Static block 1      ← Static blocks: parent first
Parent: Static block 2
Child: Static block 1       ← Then child static blocks
Child: Static block 2
Parent: Instance block 1    ← Instance: parent first
Parent: Instance block 2
Parent: Constructor
Child: Instance block 1     ← Then child instance + ctor
Child: Instance block 2
Child: Constructor

=== Second Child object ===
Parent: Instance block 1    ← Static blocks DON'T repeat!
Parent: Instance block 2
Parent: Constructor
Child: Instance block 1
Child: Instance block 2
Child: Constructor
\`\`\`

**Key Points:**

| Aspect | Static Block | Instance Block |
|--------|-------------|---------------|
| Runs | Once per class load | Every object creation |
| Access | Only static members | Static + non-static |
| Order | Parent static → Child static | Parent instance → Child instance |
| Use case | Load drivers, init static vars | Common constructor logic |`,
      },
    ],
  },
  {
    id: "design-patterns",
    label: "Patterns & UML",
    icon: "▷",
    color: "#fb923c",
    topics: [
      {
        id: 15,
        title: "Singleton Pattern",
        marks: 5,
        tag: "Design Pattern",
        content: `**Definition:** Ensures a class has ONLY ONE instance and provides a global access point. Used for: config managers, logging, DB connections, thread pools.

**3 Requirements:** Private constructor + Static private instance + Static public getInstance()

**5 Implementations:**

**1. Eager Initialization (simple, thread-safe):**
\`\`\`java
public class EagerSingleton {
    private static final EagerSingleton INSTANCE = new EagerSingleton();  // Created at load
    private EagerSingleton() { }
    public static EagerSingleton getInstance() { return INSTANCE; }
}
\`\`\`

**2. Lazy Initialization (NOT thread-safe):**
\`\`\`java
public class LazySingleton {
    private static LazySingleton instance;
    private LazySingleton() { }
    public static LazySingleton getInstance() {
        if (instance == null) instance = new LazySingleton();  // Race condition!
        return instance;
    }
}
\`\`\`

**3. Bill Pugh (Most Recommended):**
\`\`\`java
public class BillPughSingleton {
    private BillPughSingleton() { }
    
    private static class SingletonHelper {  // Loaded only when getInstance() called
        private static final BillPughSingleton INSTANCE = new BillPughSingleton();
    }
    
    public static BillPughSingleton getInstance() {
        return SingletonHelper.INSTANCE;
    }
}
\`\`\`

**4. Enum Singleton (Most robust):**
\`\`\`java
public enum EnumSingleton {
    INSTANCE;
    public void doWork() { System.out.println("Singleton working"); }
}
// Use: EnumSingleton.INSTANCE.doWork();
\`\`\`

**Demo:**
\`\`\`java
BillPughSingleton s1 = BillPughSingleton.getInstance();
BillPughSingleton s2 = BillPughSingleton.getInstance();
System.out.println(s1 == s2);          // true (same instance!)
System.out.println(s1.hashCode() == s2.hashCode()); // true
\`\`\`

**Comparison Table:**

| Implementation | Lazy Loading | Thread-safe | Recommended |
|---------------|-------------|-------------|-------------|
| Eager | No | Yes | Simple apps |
| Lazy | Yes | No | Avoid |
| Synchronized | Yes | Yes | Avoid (slow) |
| Bill Pugh | Yes | Yes | Most used |
| Enum | No | Yes | Serialization-safe |`,
      },
      {
        id: 16,
        title: "Class Diagram — Library Management System",
        marks: 5,
        tag: "UML",
        content: `**Class Diagram Components:**

| Component | Symbol | Description |
|-----------|--------|-------------|
| Class | Rectangle (3 sections) | Name, attributes, methods |
| Association | Solid line | Relationship between classes |
| Inheritance | Empty triangle + solid line | "is-a" relationship |
| Aggregation | Empty diamond | "has-a" (weak — part can exist independently) |
| Composition | Filled diamond | "has-a" (strong — part destroyed with whole) |
| Multiplicity | Numbers (1, *, 0..1) | How many instances related |

**Complete Class Diagram:**
\`\`\`text
┌──────────────┐  1     1  ┌───────────────┐
│   Library    │◄──────────│   Librarian   │
├──────────────┤           ├───────────────┤
│ -name        │           │ -empId        │
│ -address     │           │ -name         │
├──────────────┤           ├───────────────┤
│ +addBook()   │           │ +login()      │
│ +addMember() │           │ +issueBook()  │
└──────┬───────┘           └───────────────┘
       │ 1 (composition: Library ◆──── Address)
       │ contains *
       ▼
┌──────────────┐  borrowed by *   ┌──────────────┐
│     Book     │◄─────────────────│  Transaction │
├──────────────┤                  ├──────────────┤
│ -isbn        │                  │ -issueDate   │
│ -title       │                  │ -dueDate     │
│ -isAvailable │  borrows *       │ -fineAmount  │
│ ◇Category   │◄─────────────────│              │
└──────────────┘                  └───────┬──────┘
                                          │
           ┌──────────────────────────────┘ 1
           │ borrows *
    ┌──────▼──────┐
    │   Member    │  ← abstract base
    ├─────────────┤
    │ -memberId   │
    │ -name       │
    ├─────────────┤
    │ +borrow()   │
    └──────┬──────┘
           │ inherits
    ┌──────┴──────────────┐
    ▼                     ▼
┌───────────┐       ┌───────────┐
│  Student  │       │  Faculty  │
├───────────┤       ├───────────┤
│-studentId │       │-facultyId │
│-semester  │       │-designation│
├───────────┤       ├───────────┤
│+maxBooks()│       │+maxBooks()│
│ (returns 5)│      │ (returns 10)|
└───────────┘       └───────────┘
\`\`\`

**Relationships in this diagram:**

| Relationship | Between | Multiplicity | Meaning |
|-------------|---------|-------------|---------|
| Association | Library – Librarian | 1 to 1 | One library has one librarian |
| Composition ◆ | Library – Address | 1 to 1 | Address destroyed if Library destroyed |
| Aggregation ◇ | Book – Category | * to 1 | Category exists without Book |
| Inheritance | Member – Student/Faculty | - | Student "is-a" Member |
| Association | Member – Transaction | 1 to * | One member has many transactions |`,
      },
      {
        id: 17,
        title: "Aggregation vs Composition",
        marks: 5,
        tag: "UML Concepts",
        content: `| Concept | Symbol | Relationship | Lifetime |
|---------|--------|-------------|---------|
| Aggregation | ◇ Empty diamond | "has-a" (weak) | Parts survive whole destruction |
| Composition | ◆ Filled diamond | "part-of" (strong) | Parts destroyed with whole |

**Composition Example (strong — Room cannot exist without House):**
\`\`\`java
class Room {
    String name; double area;
    Room(String name, double area) { this.name = name; this.area = area; }
}

class House {
    String address;
    List<Room> rooms;
    
    House(String address) {
        this.address = address;
        this.rooms = new ArrayList<>();
        // Rooms created INSIDE House — they belong exclusively to it
        rooms.add(new Room("Bedroom", 200));
        rooms.add(new Room("Kitchen", 150));
    }
    // When House is destroyed, all Rooms are destroyed automatically
}
\`\`\`

**Aggregation Example (weak — Teacher can exist without Department):**
\`\`\`java
class Teacher {
    String name; String subject;
    Teacher(String name, String subject) { this.name = name; this.subject = subject; }
    void teach() { System.out.println(name + " teaches " + subject); }
}

class Department {
    String deptName;
    List<Teacher> teachers;
    
    Department(String name) { this.deptName = name; this.teachers = new ArrayList<>(); }
    
    void addTeacher(Teacher t) { teachers.add(t); }  // Teacher added FROM OUTSIDE
}

// Teacher exists BEFORE and AFTER the department
Teacher t1 = new Teacher("Dr. Rahim", "Physics");
Department dept = new Department("CS");
dept.addTeacher(t1);
// dept = null;  // Department gone, but t1 still lives!
t1.teach();     // Works!
\`\`\`

**Visual Summary:**
\`\`\`text
COMPOSITION (◆ filled)        AGGREGATION (◇ empty)
┌─────────┐                   ┌────────────┐
│  House  │◆                  │Department  │◇
└────┬────┘                   └─────┬──────┘
     │                              │
┌────┴────┐                   ┌─────┴──────┐
│  Room   │                   │  Teacher   │
└─────────┘                   └────────────┘
Room CANNOT exist             Teacher CAN exist
without House                 without Department
\`\`\`

**Exam Tips:**

| Statement | Aggregation | Composition |
|-----------|-------------|-------------|
| "Part can exist without whole" | ✓ | ✗ |
| "Whole destroyed → part destroyed" | ✗ | ✓ |
| "Part created inside whole" | ✗ | ✓ |
| "Part passed from outside" | ✓ | ✗ |`,
      },
      {
        id: 18,
        title: "Output Tracing — Mixed Comprehensive (C++)",
        marks: 6,
        tag: "⚡ Exam Common",
        content: `**Key Tracing Rules (memorize these):**

| Scenario | Rule |
|----------|------|
| Virtual function | Based on ACTUAL object type (runtime) |
| Non-virtual function | Based on POINTER/REFERENCE type (compile-time) |
| Constructor order | Base → Derived → MostDerived |
| Destructor order (virtual) | MostDerived → Derived → Base |
| Static variable | Shared; increments per object created |

**Tracing Example:**
\`\`\`cpp
class A {
public:
    A() { cout << "A()\\n"; }
    virtual void f1() { cout << "A::f1()\\n"; }
    void f2() { cout << "A::f2()\\n"; }
    virtual ~A() { cout << "~A()\\n"; }
};
class B : public A {
public:
    B() : A() { cout << "B()\\n"; }
    void f1() override { cout << "B::f1()\\n"; }
    virtual void f3() { cout << "B::f3()\\n"; }
    ~B() override { cout << "~B()\\n"; }
};
class C : public B {
public:
    C() : B() { cout << "C()\\n"; }
    void f1() override { cout << "C::f1()\\n"; }
    void f2() { cout << "C::f2()\\n"; }      // Hides B's f2, not override
    void f3() override { cout << "C::f3()\\n"; }
    ~C() override { cout << "~C()\\n"; }
};
\`\`\`

**Test 1 — Direct object \`C obj;\`:**
\`\`\`text
A()  B()  C()           ← Constructor chain
C::f1()                  ← Virtual → actual type C
C::f2()                  ← Non-virtual, object is C → C's version
C::f3()                  ← Virtual from B, overridden in C
~C()  ~B()  ~A()         ← Destructor: reverse order
\`\`\`

**Test 2 — \`A* ptr = new C();\`:**
\`\`\`text
A()  B()  C()
C::f1()                  ← Virtual → actual type C ✓
A::f2()                  ← Non-virtual → pointer type A* ✓
~C()  ~B()  ~A()         ← Virtual destructor works ✓
\`\`\`

**Test 3 — \`B* ptr = new C();\`:**
\`\`\`text
A()  B()  C()
C::f1()                  ← Virtual → actual type C ✓
B::f2()                  ← Non-virtual → pointer type B* ✓
C::f3()                  ← Virtual from B, overridden in C ✓
~C()  ~B()  ~A()
\`\`\`

**Test 4 — Array \`A* arr[] = {new A(), new B(), new C()}\`:**
\`\`\`text
A()                      ← arr[0]: Base only
A()  B()                 ← arr[1]: Base + Derived
A()  B()  C()            ← arr[2]: Full chain

arr[0]->f1() → A::f1()  ← Virtual, object is A
arr[1]->f1() → B::f1()  ← Virtual, object is B
arr[2]->f1() → C::f1()  ← Virtual, object is C
All arr[i]->f2() → A::f2() ← Non-virtual, ptr type A*
\`\`\`

**Test 5 — References \`A& ref1 = c; B& ref2 = c;\`:**
\`\`\`text
ref1.f1() → C::f1()   ← Virtual: A& points to C
ref1.f2() → A::f2()   ← Non-virtual: A& type
ref2.f1() → C::f1()   ← Virtual: B& points to C
ref2.f2() → B::f2()   ← Non-virtual: B& type
ref2.f3() → C::f3()   ← Virtual: C overrides B::f3()
\`\`\``,
      },
      {
        id: 19,
        title: "Java String Comparison Output",
        marks: 5,
        tag: "Output Tracing",
        content: `**Predict the output — classic exam question:**

\`\`\`java
String s1 = "Java";
String s2 = "Java";
String s3 = new String("Java");
String s4 = s3.intern();
String s5 = s4;

System.out.println(s1 == s2);      // ??
System.out.println(s1 == s3);      // ??
System.out.println(s1 == s4);      // ??
System.out.println(s4 == s3);      // ??
\`\`\`

**Output + Explanation:**
\`\`\`text
true   ← Both literals → same pool object
false  ← new String() → heap (outside pool)
true   ← intern() → returns pool version (= s1)
false  ← s4=pool, s3=heap (different objects)
\`\`\`

**Runtime vs Compile-time Concatenation:**
\`\`\`java
String a = "Hello", b = "World";
String c = "HelloWorld";
String d = a + b;           // Runtime — creates new object
String e = "Hello" + "World"; // Compile-time — optimized to pool literal

System.out.println(c == d); // false (runtime object)
System.out.println(c == e); // true  (compile-time constant)
\`\`\`

**Wrapper Class Caching (Java caches -128 to 127):**
\`\`\`java
Integer i1 = 127;  Integer i2 = 127;
Integer i3 = 128;  Integer i4 = 128;

System.out.println(i1 == i2);      // true  (cached)
System.out.println(i3 == i4);      // false (beyond cache)
System.out.println(i3.equals(i4)); // true
\`\`\`

**StringBuilder doesn't override equals():**
\`\`\`java
StringBuilder sb1 = new StringBuilder("Test");
StringBuilder sb2 = new StringBuilder("Test");

System.out.println(sb1 == sb2);              // false (different objects)
System.out.println(sb1.equals(sb2));         // false (not overridden!)
System.out.println(sb1.toString().equals(sb2.toString())); // true ✓
\`\`\`

**Quick Reference:**

| Type | == | .equals() |
|------|-----|----------|
| Primitives | Compares values ✓ | N/A |
| String literals | Same pool object → true | Compares content |
| new String() | Different objects → false | Compares content |
| StringBuilder | Reference compare | Reference compare (not overridden) |
| Integer (≤127) | Cached → true | Compares value |
| Integer (>127) | Different objects → false | Compares value |`,
      },
      {
        id: 20,
        title: "Output Tracing — Virtual + Inheritance (C++)",
        marks: 6,
        tag: "⚡ Exam Common",
        content: `**Static global counter + virtual + hierarchy — comprehensive trace:**

\`\`\`cpp
static int objectCount = 0;
class Base {
protected: int id;
public:
    Base() { id = ++objectCount; cout << "Base(" << id << ") constructor\\n"; }
    virtual void display() { cout << "Base::display() - " << id << endl; }
    void show() { cout << "Base::show() - " << id << endl; }
    virtual ~Base() { cout << "Base(" << id << ") destructor\\n"; }
};
class Derived : public Base {
    int* data;
public:
    Derived() { data = new int(100); cout << "Derived(" << id << ") constructor\\n"; }
    void display() override { cout << "Derived::display() - " << id << ", *data=" << *data << endl; }
    void show() { cout << "Derived::show() - " << id << endl; }
    ~Derived() override { delete data; cout << "Derived(" << id << ") destructor\\n"; }
};
\`\`\`

**Part 1 — \`Derived d1;\` then \`d1.display(); d1.show();\`:**
\`\`\`text
Base(1) constructor
Derived(1) constructor
Derived::display() - 1, *data=100   ← Virtual, object is Derived
Derived::show() - 1                  ← Non-virtual, but object is Derived (direct call)
\`\`\`

**Part 2 — \`Base* ptr = new Derived();\`:**
\`\`\`text
Base(2) constructor
Derived(2) constructor
Derived::display() - 2, *data=100   ← Virtual → Derived ✓
Base::show() - 2                     ← Non-virtual → pointer type Base* ✓
Derived(2) destructor                ← Virtual destructor!
Base(2) destructor
\`\`\`

**Part 3 — Array \`Base* arr[3] = {new Base(), new Derived(), new Derived()};\`:**
\`\`\`text
Base(3) constructor
Base(4) constructor → Derived(4) constructor
Base(5) constructor → Derived(5) constructor

arr[0]->display() → Base::display() - 3
arr[0]->show()    → Base::show() - 3
arr[1]->display() → Derived::display() - 4
arr[1]->show()    → Base::show() - 4      ← Still Base (pointer type!)
arr[2]->display() → Derived::display() - 5
arr[2]->show()    → Base::show() - 5
\`\`\`

**Final objectCount: 5** (increments per constructor call)

**Master Cheat Sheet:**
\`\`\`text
VIRTUAL function call → actual OBJECT TYPE (runtime)
NON-VIRTUAL call      → POINTER/REFERENCE TYPE (compile-time)

Direct object:  obj.show()   → always Derived::show()
Base pointer:   ptr->show()  → always Base::show()
Base reference: ref.show()   → always Base::show()

Constructor:  Base → Derived → MostDerived
Destructor:   MostDerived → Derived → Base  (needs virtual!)

Static objectCount increases once per CONSTRUCTOR call
(including copy constructor — don't forget!)
\`\`\``,
      },
    ],
  },
];

function renderContent(text: string) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;
  let keyCounter = 0;
  const key = () => keyCounter++;

  const renderInline = (line: string): React.ReactNode => {
    const parts = line.split(/(`[^`]+`|\*\*[^*]+\*\*)/g);
    return parts.map((part, pi) => {
      if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
        return (
          <code
            key={pi}
            style={{
              background: "rgba(255,255,255,0.08)",
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: "4px",
              padding: "1px 6px",
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              fontSize: "0.82em",
              color: "#e2c08d",
            }}
          >
            {part.slice(1, -1)}
          </code>
        );
      }
      if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        return (
          <strong key={pi} style={{ color: "#fff", fontWeight: 600 }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith("```")) {
      const lang = line.trim().slice(3);
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <div
          key={key()}
          style={{
            margin: "12px 0",
            borderRadius: "10px",
            overflow: "hidden",
            border: "1px solid rgba(255,255,255,0.1)",
          }}
        >
          {lang && (
            <div
              style={{
                background: "rgba(255,255,255,0.05)",
                padding: "4px 14px",
                fontSize: "0.7em",
                color: "rgba(255,255,255,0.4)",
                fontFamily: "monospace",
                letterSpacing: "0.05em",
                textTransform: "uppercase",
              }}
            >
              {lang}
            </div>
          )}
          <pre
            style={{
              background: "rgba(0,0,0,0.35)",
              padding: "14px 16px",
              margin: 0,
              overflowX: "auto",
              fontSize: "0.8em",
              lineHeight: 1.6,
              color: "#c9d1d9",
              fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
            }}
          >
            <code
              dangerouslySetInnerHTML={{
                __html: codeLines
                  .join("\n")
                  .replace(/&/g, "&amp;")
                  .replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;")
                  .replace(/(\/\/[^\n]*)/g, '<span style="color:#6a9955">$1</span>')
                  .replace(
                    /\b(class|public|private|protected|virtual|override|static|const|int|double|string|void|return|new|delete|bool|char|float|nullptr|true|false|if|else|for|while|cout|cin|endl|using|namespace|include|abstract|interface|extends|implements|final|throws|throw|try|catch|finally|super|this|import|System|String|StringBuilder)\b/g,
                    '<span style="color:#569cd6">$1</span>'
                  )
                  .replace(/"([^"]*)"/g, '<span style="color:#ce9178">"$1"</span>'),
              }}
            />
          </pre>
        </div>
      );
      i++;
      continue;
    }

    if (line.trim().startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const rows = tableLines.filter((l) => !l.match(/^\|[-|\s]+\|$/));
      elements.push(
        <div key={key()} style={{ margin: "12px 0", overflowX: "auto" }}>
         <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82em" }}>
  <tbody>
    {rows.map((row, ri) => {
      const isHeader = ri === 0;

      return (
        <tr
          key={ri}
          style={{
            background: isHeader ? "rgba(255,255,255,0.05)" : "transparent",
          }}
        >
          <td>{row}</td>
        </tr>
      );
    })}
  </tbody>
</table>
        </div>
      );
      continue;
    }

    if (line.startsWith("**") && line.endsWith("**") && line.length > 4 && !line.slice(2,-2).includes("**")) {
      elements.push(
        <p
          key={key()}
          style={{ fontWeight: 700, color: "#fff", margin: "14px 0 4px", fontSize: "0.92em" }}
        >
          {line.slice(2, -2)}
        </p>
      );
      i++;
      continue;
    }

    if (line.match(/^[-*]\s/)) {
      elements.push(
        <div
          key={key()}
          style={{
            display: "flex",
            gap: "8px",
            margin: "3px 0",
            color: "rgba(255,255,255,0.75)",
            fontSize: "0.86em",
          }}
        >
          <span style={{ color: "rgba(255,255,255,0.3)", flexShrink: 0 }}>›</span>
          <span>{renderInline(line.replace(/^[-*]\s/, ""))}</span>
        </div>
      );
      i++;
      continue;
    }

    if (line.trim() === "") {
      elements.push(<div key={key()} style={{ height: "6px" }} />);
      i++;
      continue;
    }

    elements.push(
      <p
        key={key()}
        style={{ color: "rgba(255,255,255,0.75)", fontSize: "0.86em", lineHeight: 1.65, margin: "4px 0" }}
      >
        {renderInline(line)}
      </p>
    );
    i++;
  }

  return elements;
}

export default function OOPStudyGuidePart2() {
  const [activeChapter, setActiveChapter] = useState(chapters[0].id);
  const [activeTopic, setActiveTopic] = useState<number | null>(null);
  const [search, setSearch] = useState("");

  const currentChapter = chapters.find((c) => c.id === activeChapter)!;

  const filteredTopics = search.trim()
    ? chapters
        .flatMap((c) => c.topics.map((t) => ({ ...t, chapterId: c.id, chapterColor: c.color })))
        .filter(
          (t) =>
            t.title.toLowerCase().includes(search.toLowerCase()) ||
            t.tag.toLowerCase().includes(search.toLowerCase())
        )
    : currentChapter.topics.map((t) => ({
        ...t,
        chapterId: currentChapter.id,
        chapterColor: currentChapter.color,
      }));

  const openTopic =
    activeTopic !== null
      ? chapters.flatMap((c) => c.topics).find((t) => t.id === activeTopic)
      : null;
  const openTopicColor =
    activeTopic !== null
      ? chapters.find((c) => c.topics.some((t) => t.id === activeTopic))?.color
      : undefined;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0d0f14",
        color: "#fff",
        fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        .topic-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.15) !important; }
        .nav-tab:hover { background: rgba(255,255,255,0.06) !important; }
        .close-btn:hover { background: rgba(255,255,255,0.1) !important; }
      `}</style>

      {/* Header */}
      <div
        style={{
          padding: "28px 32px 20px",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(255,255,255,0.02)",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <div
          style={{
            maxWidth: "1100px",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            gap: "20px",
            flexWrap: "wrap",
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "2px" }}>
              <div
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #ff6b6b, #fb923c)",
                  boxShadow: "0 0 10px #ff6b6b60",
                }}
              />
              <span
                style={{
                  fontSize: "0.7em",
                  letterSpacing: "0.15em",
                  textTransform: "uppercase",
                  color: "rgba(255,255,255,0.4)",
                  fontWeight: 500,
                }}
              >
                CSE — OOP Advanced Topics
              </span>
            </div>
            <h1
              style={{
                fontSize: "1.6em",
                fontWeight: 700,
                margin: 0,
                background: "linear-gradient(135deg, #fff 40%, rgba(255,255,255,0.5))",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                letterSpacing: "-0.02em",
              }}
            >
              Study Guide — Part 2
            </h1>
          </div>

          <div style={{ position: "relative" }}>
            <span
              style={{
                position: "absolute",
                left: "12px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "rgba(255,255,255,0.3)",
                fontSize: "0.85em",
              }}
            >
              ⌕
            </span>
            <input
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setActiveTopic(null);
              }}
              placeholder="Search topics..."
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px",
                padding: "8px 14px 8px 32px",
                color: "#fff",
                fontSize: "0.85em",
                outline: "none",
                width: "200px",
              }}
            />
          </div>

          <div style={{ display: "flex", gap: "16px" }}>
            {[
              { label: "Topics", value: chapters.flatMap((c) => c.topics).length },
              { label: "Chapters", value: chapters.length },
            ].map((s) => (
              <div key={s.label} style={{ textAlign: "center" }}>
                <div style={{ fontSize: "1.3em", fontWeight: 700, color: "#ff6b6b" }}>
                  {s.value}
                </div>
                <div
                  style={{
                    fontSize: "0.65em",
                    color: "rgba(255,255,255,0.35)",
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                  }}
                >
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", width: "100%", padding: "0 24px", flex: 1 }}>
        {/* Chapter Navigation */}
        {!search && (
          <div
            style={{
              display: "flex",
              gap: "6px",
              padding: "16px 0",
              overflowX: "auto",
              scrollbarWidth: "none",
            }}
          >
            {chapters.map((ch) => (
              <button
                key={ch.id}
                className="nav-tab"
                onClick={() => {
                  setActiveChapter(ch.id);
                  setActiveTopic(null);
                }}
                style={{
                  background:
                    activeChapter === ch.id ? `${ch.color}20` : "rgba(255,255,255,0.04)",
                  border: `1px solid ${
                    activeChapter === ch.id ? ch.color + "60" : "rgba(255,255,255,0.08)"
                  }`,
                  borderRadius: "8px",
                  padding: "8px 16px",
                  color: activeChapter === ch.id ? ch.color : "rgba(255,255,255,0.55)",
                  cursor: "pointer",
                  fontSize: "0.83em",
                  fontWeight: activeChapter === ch.id ? 600 : 400,
                  whiteSpace: "nowrap",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  transition: "all 0.2s",
                  fontFamily: "inherit",
                }}
              >
                <span style={{ fontSize: "0.9em" }}>{ch.icon}</span>
                {ch.label}
                <span
                  style={{
                    background: activeChapter === ch.id ? ch.color + "30" : "rgba(255,255,255,0.06)",
                    borderRadius: "4px",
                    padding: "1px 5px",
                    fontSize: "0.75em",
                    color: activeChapter === ch.id ? ch.color : "rgba(255,255,255,0.35)",
                  }}
                >
                  {ch.topics.length}
                </span>
              </button>
            ))}
          </div>
        )}

        {search && (
          <div style={{ padding: "16px 0 8px", fontSize: "0.8em", color: "rgba(255,255,255,0.4)" }}>
            {filteredTopics.length} result{filteredTopics.length !== 1 ? "s" : ""} for "{search}"
          </div>
        )}

        {/* Topic Cards Grid */}
        {!openTopic && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
              gap: "14px",
              paddingBottom: "40px",
            }}
          >
            {filteredTopics.map((topic) => (
              <div
                key={topic.id}
                className="topic-card"
                onClick={() => setActiveTopic(topic.id)}
                style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: "12px",
                  padding: "18px",
                  cursor: "pointer",
                  transition: "all 0.2s",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: "2px",
                    background: `linear-gradient(90deg, ${topic.chapterColor}, transparent)`,
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    marginBottom: "10px",
                  }}
                >
                  <span
                    style={{
                      background: topic.chapterColor + "18",
                      color: topic.chapterColor,
                      border: `1px solid ${topic.chapterColor}30`,
                      borderRadius: "5px",
                      padding: "2px 8px",
                      fontSize: "0.7em",
                      fontWeight: 500,
                    }}
                  >
                    {topic.tag}
                  </span>
                  <span
                    style={{
                      fontSize: "0.7em",
                      color: "rgba(255,255,255,0.3)",
                      background: "rgba(255,255,255,0.05)",
                      borderRadius: "4px",
                      padding: "2px 7px",
                    }}
                  >
                    {topic.marks}m
                  </span>
                </div>
                <h3
                  style={{
                    margin: "0 0 8px",
                    fontSize: "0.95em",
                    fontWeight: 600,
                    color: "rgba(255,255,255,0.9)",
                    lineHeight: 1.35,
                  }}
                >
                  Q{topic.id}. {topic.title}
                </h3>
                <div
                  style={{
                    fontSize: "0.75em",
                    color: "rgba(255,255,255,0.3)",
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                  }}
                >
                  View answer <span style={{ fontSize: "0.9em" }}>→</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Topic Detail View */}
        {openTopic && (
          <div style={{ paddingBottom: "60px" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "16px 0",
              }}
            >
              <button
                className="close-btn"
                onClick={() => setActiveTopic(null)}
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  color: "rgba(255,255,255,0.6)",
                  cursor: "pointer",
                  padding: "6px 14px",
                  fontSize: "0.82em",
                  transition: "background 0.2s",
                  fontFamily: "inherit",
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                }}
              >
                ← Back
              </button>
              <div style={{ flex: 1, height: "1px", background: "rgba(255,255,255,0.06)" }} />
              <span style={{ fontSize: "0.75em", color: "rgba(255,255,255,0.3)" }}>
                Q{openTopic.id} · {openTopic.marks} marks
              </span>
            </div>

            <div
              style={{
                background: "rgba(255,255,255,0.025)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: "14px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  padding: "20px 24px",
                  borderBottom: "1px solid rgba(255,255,255,0.07)",
                  background: openTopicColor ? openTopicColor + "10" : undefined,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "12px",
                }}
              >
                <div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginBottom: "6px",
                    }}
                  >
                    <span
                      style={{
                        background: openTopicColor ? openTopicColor + "25" : "rgba(255,255,255,0.1)",
                        color: openTopicColor || "#fff",
                        border: `1px solid ${openTopicColor ? openTopicColor + "40" : "rgba(255,255,255,0.15)"}`,
                        borderRadius: "5px",
                        padding: "2px 8px",
                        fontSize: "0.72em",
                        fontWeight: 600,
                      }}
                    >
                      {openTopic.tag}
                    </span>
                  </div>
                  <h2 style={{ margin: 0, fontSize: "1.2em", fontWeight: 700, color: "#fff" }}>
                    Q{openTopic.id}. {openTopic.title}
                  </h2>
                </div>
                <div
                  style={{
                    background: openTopicColor ? openTopicColor + "20" : "rgba(255,255,255,0.05)",
                    border: `1px solid ${openTopicColor ? openTopicColor + "40" : "rgba(255,255,255,0.1)"}`,
                    borderRadius: "8px",
                    padding: "8px 14px",
                    textAlign: "center",
                    flexShrink: 0,
                  }}
                >
                  <div
                    style={{
                      fontSize: "1.4em",
                      fontWeight: 700,
                      color: openTopicColor || "#fff",
                    }}
                  >
                    {openTopic.marks}
                  </div>
                  <div
                    style={{
                      fontSize: "0.65em",
                      color: "rgba(255,255,255,0.4)",
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                    }}
                  >
                    marks
                  </div>
                </div>
              </div>

              <div style={{ padding: "24px" }}>{renderContent(openTopic.content)}</div>
            </div>

            {/* Prev / Next */}
            <div style={{ display: "flex", gap: "10px", marginTop: "16px" }}>
              {(() => {
                const allTopics = chapters.flatMap((c) => c.topics);
                const idx = allTopics.findIndex((t) => t.id === openTopic.id);
                const prev = idx > 0 ? allTopics[idx - 1] : null;
                const next = idx < allTopics.length - 1 ? allTopics[idx + 1] : null;
                return (
                  <>
                    {prev && (
                      <button
                        className="close-btn"
                        onClick={() => setActiveTopic(prev.id)}
                        style={{
                          background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          borderRadius: "8px",
                          color: "rgba(255,255,255,0.55)",
                          cursor: "pointer",
                          padding: "8px 16px",
                          fontSize: "0.8em",
                          transition: "background 0.2s",
                          fontFamily: "inherit",
                          flex: 1,
                          textAlign: "left",
                        }}
                      >
                        ← Q{prev.id}. {prev.title}
                      </button>
                    )}
                    {next && (
                      <button
                        className="close-btn"
                        onClick={() => setActiveTopic(next.id)}
                        style={{
                          background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.08)",
                          borderRadius: "8px",
                          color: "rgba(255,255,255,0.55)",
                          cursor: "pointer",
                          padding: "8px 16px",
                          fontSize: "0.8em",
                          transition: "background 0.2s",
                          fontFamily: "inherit",
                          flex: 1,
                          textAlign: "right",
                        }}
                      >
                        Q{next.id}. {next.title} →
                      </button>
                    )}
                  </>
                );
              })()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}