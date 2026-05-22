"use client";

import { useState } from "react";

const chapters = [
  {
    id: "core",
    label: "Core OOP",
    icon: "◈",
    color: "#00d4aa",
    topics: [
      {
        id: 1,
        title: "Class vs Object",
        marks: 5,
        tag: "Fundamental",
        content: `**Definition:**
A **class** is a user-defined blueprint or template that defines the structure (data members) and behavior (member functions) of objects. An **object** is an instance of a class — a concrete entity created from the blueprint.

**Syntax:**
\`\`\`cpp
class Student {          // CLASS - blueprint
    string name;
    int roll;
public:
    void setData(string n, int r) { name = n; roll = r; }
    void display() { cout << name << " " << roll; }
};

int main() {
    Student s1, s2;           // s1, s2 are OBJECTS
    s1.setData("Rahim", 101);
    s2.setData("Karim", 102);
    s1.display();  // Rahim 101
    s2.display();  // Karim 102
}
\`\`\`

**Real-world analogy:** Class = "Car" blueprint; Object = actual "Toyota", "Honda" cars.`,
      },
      {
        id: 2,
        title: "Procedural vs OOP",
        marks: 5,
        tag: "Comparison",
        content: `| # | Feature | Procedural | OOP |
|---|---------|-----------|-----|
| 1 | Division | Functions | Objects |
| 2 | Data Security | Low (global vars) | High (private/protected) |
| 3 | Approach | Top-down | Bottom-up |
| 4 | Reusability | Limited | High (inheritance) |
| 5 | Overloading | Not supported | Supported |
| 6 | Languages | C, Pascal | C++, Java, Python |

\`\`\`c
// Procedural (C) - data and functions separate
int balance = 1000;
void withdraw(int amt) { balance -= amt; } // Anyone can modify!
\`\`\`

\`\`\`cpp
// OOP (C++) - data and functions together
class Bank {
private:
    int balance = 1000;        // Hidden
public:
    void withdraw(int amt) {   // Controlled access
        if(amt <= balance) balance -= amt;
    }
};
\`\`\``,
      },
      {
        id: 3,
        title: "Encapsulation",
        marks: 5,
        tag: "Principle",
        content: `**Definition:** Encapsulation bundles data and methods into a single unit (class), restricting direct access to internal components.

**Key aspects:**
- **Data hiding** – making data members private
- **Controlled access** – public getter/setter methods
- **Validation** – checking data before modification
- **Abstraction** – hiding implementation details

**Real-life:** ATM Machine — you interact via interface; the ledger, balance, and security are encapsulated (hidden).

\`\`\`cpp
class BankAccount {
private:
    double balance;     // HIDDEN
    string password;
public:
    void deposit(double amount) {
        if (amount > 0) balance += amount;
    }
    bool withdraw(double amount, string pwd) {
        if (pwd == password && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    double getBalance() { return balance; }  // View only
};

int main() {
    BankAccount acc;
    // acc.balance = 5000;  // ERROR! Cannot access directly
    acc.deposit(5000);       // Correct way
}
\`\`\`

**Benefits:** Security, maintainability, modularity.`,
      },
      {
        id: 4,
        title: "Data Hiding",
        marks: 5,
        tag: "Principle",
        content: `**Definition:** Restricting direct access to an object's internal data members to prevent unauthorized modification.

**Why needed?**
- Prevents accidental data corruption
- Maintains data integrity and consistency
- Allows validation before modification
- Reduces system complexity

**How achieved — 3 methods:**

| Method | Explanation | Example |
|--------|-------------|---------|
| private specifier | Only accessible within same class | \`private: int age;\` |
| protected specifier | Accessible in derived classes | \`protected: int id;\` |
| Public getter/setter | Controlled read/write access | \`void setAge(int a) { if(a>0) age=a; }\` |

\`\`\`cpp
class Employee {
private:
    int salary;          // DATA HIDING
    string employeeId;
public:
    void setSalary(int s) {
        if (s >= 0) salary = s;   // Validation!
    }
    int getSalary() { return salary; }
    // No getter for employeeId - completely hidden!
};
\`\`\`

**Real-world analogy:** You cannot directly change exam marks in the university database — you submit a form (public method) that goes through validation.`,
      },
    ],
  },
  {
    id: "polymorphism",
    label: "Polymorphism",
    icon: "⟡",
    color: "#ff6b6b",
    topics: [
      {
        id: 5,
        title: "Polymorphism Overview",
        marks: 6,
        tag: "Core Concept",
        content: `**Definition:** Polymorphism (Greek: "many forms") allows entities to behave differently based on context.

**Compile-time Polymorphism (Static):**

| Aspect | Description |
|--------|-------------|
| Also called | Early binding, Static polymorphism |
| When resolved | During compilation |
| Achieved via | Function overloading, Operator overloading |
| Speed | Faster (no runtime overhead) |

\`\`\`cpp
class Calculator {
public:
    int add(int a, int b) { return a + b; }          // Version 1
    double add(double a, double b) { return a + b; } // Version 2
    int add(int a, int b, int c) { return a+b+c; }   // Version 3
};
\`\`\`

**Runtime Polymorphism (Dynamic):**

| Aspect | Description |
|--------|-------------|
| Also called | Late binding, Dynamic polymorphism |
| When resolved | During program execution |
| Achieved via | Virtual functions, Function overriding |
| Speed | Slightly slower (vtable lookup) |

\`\`\`cpp
class Animal {
public:
    virtual void sound() { cout << "Animal sound\\n"; }
};
class Dog : public Animal {
public:
    void sound() override { cout << "Dog barks\\n"; }
};
class Cat : public Animal {
public:
    void sound() override { cout << "Cat meows\\n"; }
};

int main() {
    Animal* ptr;
    Dog d; Cat c;
    ptr = &d; ptr->sound();  // "Dog barks" (RUNTIME decision)
    ptr = &c; ptr->sound();  // "Cat meows" (RUNTIME decision)
}
\`\`\``,
      },
      {
        id: 16,
        title: "Virtual Functions",
        marks: 6,
        tag: "⚡ BUET Favorite",
        content: `**Definition:** A virtual function uses the \`virtual\` keyword to enable runtime polymorphism — the correct function is called based on the actual object type, not the pointer type.

**Problem WITHOUT virtual:**
\`\`\`cpp
class Animal {
public:
    void speak() { cout << "Animal sound\\n"; }  // Non-virtual
};
class Dog : public Animal {
public:
    void speak() { cout << "Bark\\n"; }
};

Animal* ptr = new Dog();
ptr->speak();  // Output: "Animal sound" ❌ WRONG!
\`\`\`

**Solution WITH virtual:**
\`\`\`cpp
class Animal {
public:
    virtual void speak() { cout << "Animal sound\\n"; }
};
class Dog : public Animal {
public:
    void speak() override { cout << "Bark\\n"; }
};

Animal* ptr = new Dog();
ptr->speak();  // Output: "Bark" ✓ CORRECT!
\`\`\`

**How it works internally — vtable:**
\`\`\`
Animal object:  [vptr] → Animal_vtable → [&Animal::speak]
Dog object:     [vptr] → Dog_vtable    → [&Dog::speak]
\`\`\`

**Key Rules:**

| Rule | Explanation |
|------|-------------|
| Cannot be static | Static functions have no \`this\` pointer |
| Constructor cannot be virtual | Object doesn't exist yet |
| Destructor SHOULD be virtual | Ensures proper cleanup |
| Pure virtual (= 0) | Makes class abstract |`,
      },
      {
        id: 17,
        title: "Pure Virtual / Abstract Class",
        marks: 5,
        tag: "Important",
        content: `**Pure Virtual Function:** Declared with \`= 0\`, no implementation in base class.
\`\`\`cpp
virtual void functionName() = 0;  // Pure virtual
\`\`\`

**Abstract Class:** Contains at least one pure virtual function. Cannot be instantiated.

\`\`\`cpp
class Shape {                          // ABSTRACT CLASS
protected:
    string color;
public:
    virtual double area() = 0;         // Pure virtual
    virtual double perimeter() = 0;    // Pure virtual
    virtual void draw() {              // Regular virtual (has default)
        cout << "Drawing a " << color << " shape\\n";
    }
    virtual ~Shape() { }               // Virtual destructor!
};

class Circle : public Shape {          // CONCRETE CLASS
    double radius;
public:
    Circle(double r) : radius(r) { }
    double area() override { return 3.14159 * radius * radius; }
    double perimeter() override { return 2 * 3.14159 * radius; }
};

// Shape s;    // ERROR! Cannot instantiate abstract class
Shape* s = new Circle(5);  // OK — pointer to abstract is fine
\`\`\`

**When to use abstract classes:**

| Use Case | Example |
|----------|---------|
| Defining interfaces | All shapes must have area() |
| Providing common impl. | Default draw() method |
| Enforcing contracts | Every derived must implement |`,
      },
      {
        id: 18,
        title: "Static vs Dynamic Binding",
        marks: 5,
        tag: "Concept",
        content: `**Binding:** Connecting a function call to its definition.

| Feature | Static Binding | Dynamic Binding |
|---------|---------------|-----------------|
| Other names | Early binding | Late binding |
| Resolution time | Compile time | Runtime |
| Speed | Faster | Slower (vtable) |
| Basis | Pointer/reference TYPE | Actual OBJECT type |
| Functions | Non-virtual | Virtual |

\`\`\`cpp
class Parent {
public:
    void normalFunc() { cout << "Parent normal\\n"; }
    virtual void virtualFunc() { cout << "Parent virtual\\n"; }
};
class Child : public Parent {
public:
    void normalFunc() { cout << "Child normal\\n"; }
    void virtualFunc() override { cout << "Child virtual\\n"; }
};

int main() {
    Parent* ptr = new Child();
    ptr->normalFunc();   // STATIC  → "Parent normal" (pointer type)
    ptr->virtualFunc();  // DYNAMIC → "Child virtual" (object type)
}
\`\`\`

**Key exam point:** In C++, default is static binding. You must explicitly use \`virtual\` keyword for dynamic binding.`,
      },
    ],
  },
  {
    id: "constructors",
    label: "Constructors",
    icon: "⬡",
    color: "#ffd93d",
    topics: [
      {
        id: 6,
        title: "Constructors & Types",
        marks: 5,
        tag: "⚡ BUET Favorite",
        content: `**Definition:** A constructor is a special member function automatically invoked when an object is created. It has:
- Same name as the class
- No return type (not even void)
- Can be overloaded

**Types:**

\`\`\`cpp
class Student {
    string name; int roll; int* marks;
public:
    // 1. DEFAULT CONSTRUCTOR
    Student() {
        name = "Unknown"; roll = 0; marks = nullptr;
    }
    
    // 2. PARAMETERIZED CONSTRUCTOR
    Student(string n, int r) {
        name = n; roll = r;
        marks = new int[5];
    }
    
    // 3. COPY CONSTRUCTOR
    Student(const Student &s) {
        name = s.name; roll = s.roll;
        marks = new int[5];           // Deep copy!
        for(int i = 0; i < 5; i++)
            marks[i] = s.marks[i];
    }
    
    // 4. MOVE CONSTRUCTOR (C++11)
    Student(Student &&s) noexcept {
        name = move(s.name); roll = s.roll;
        marks = s.marks; s.marks = nullptr;
    }
    
    ~Student() { delete[] marks; }
};
\`\`\`

| Constructor | Called when |
|-------------|-------------|
| Default | \`Student s1;\` |
| Parameterized | \`Student s2("Ali", 10);\` |
| Copy | \`Student s3 = s1;\` or pass by value |
| Move | \`Student s4 = move(s1);\` |`,
      },
      {
        id: 7,
        title: "Copy Constructor (Deep vs Shallow)",
        marks: 6,
        tag: "Very Important",
        content: `**Definition:** Initializes a new object as a copy of an existing object.
\`\`\`cpp
ClassName(const ClassName &old_object);
\`\`\`

**When called (3 situations):**
1. Direct initialization: \`Sample s2 = s1;\`
2. Passing by value to function
3. Returning object from function

**Deep Copy vs Shallow Copy:**

| Aspect | Shallow Copy (Default) | Deep Copy (User-defined) |
|--------|----------------------|--------------------------|
| Copies | Memory addresses (pointers) | Actual data contents |
| Dynamic memory | Both point to same location | Each has own copy |
| Issue | Double deletion, data corruption | No such issue |

\`\`\`cpp
class String {
    char* data;
    int length;
public:
    String(const char* str = "") {
        length = strlen(str);
        data = new char[length + 1];
        strcpy(data, str);
    }
    
    // COPY CONSTRUCTOR with DEEP COPY
    String(const String &other) {
        length = other.length;
        data = new char[length + 1];  // NEW memory allocation
        strcpy(data, other.data);      // Copy actual data
    }
    
    ~String() { delete[] data; }
};

int main() {
    String s1("Hello");
    String s2(s1);          // Deep copy
    s1.modify("World");
    // s1 = "World", s2 = "Hello" (unaffected!)
}
\`\`\`

**Danger of default copy:**
\`\`\`cpp
class Dangerous {
    int* ptr;
public:
    Dangerous() { ptr = new int(10); }
    ~Dangerous() { delete ptr; }  // BOTH objects delete same memory → CRASH!
};
// Dangerous d1, d2 = d1;  // Double deletion!
\`\`\``,
      },
      {
        id: 8,
        title: "Destructor",
        marks: 5,
        tag: "Important",
        content: `**Definition:** Automatically invoked when an object goes out of scope or is deleted. Has:
- Same name as class with \`~\` prefix
- No parameters, no return type
- Cannot be overloaded (only one per class)

**Why needed:**

| Reason | Explanation |
|--------|-------------|
| Memory leak prevention | Deallocates dynamic memory (\`delete\`) |
| Resource cleanup | Closes files, network sockets |
| Release locks | Prevents deadlocks |
| Inheritance cleanup | Virtual destructor needed |

\`\`\`cpp
class FileManager {
    FILE* file;
    char* buffer;
public:
    FileManager(const char* filename) {
        file = fopen(filename, "r");
        buffer = new char[1024];
    }
    ~FileManager() {          // DESTRUCTOR - cleanup
        if (file) fclose(file);
        delete[] buffer;
        cout << "File closed, buffer freed\\n";
    }
};
// Destructor called automatically when FileManager goes out of scope
\`\`\`

**Virtual Destructor:**
\`\`\`cpp
class Base {
public:
    virtual ~Base() { }    // ALWAYS use virtual destructor in base!
};
// Without virtual: deleting derived through base ptr = incomplete cleanup
\`\`\`

**Call order:** Constructors: Base → Derived. Destructors: Derived → Base (reverse).`,
      },
      {
        id: 10,
        title: "Static Variable — Object Counter",
        marks: 5,
        tag: "Program",
        content: `**Concept:**
- **Static data member:** ONE copy shared by ALL objects
- **Static member function:** Called without object; only accesses static members

\`\`\`cpp
class ObjectCounter {
private:
    static int objectCount;  // Shared by all objects
public:
    ObjectCounter() {
        objectCount++;
        cout << "Created. Total: " << objectCount << endl;
    }
    ObjectCounter(const ObjectCounter&) {  // Don't forget copy!
        objectCount++;
        cout << "Copied. Total: " << objectCount << endl;
    }
    ~ObjectCounter() {
        objectCount--;
        cout << "Destroyed. Remaining: " << objectCount << endl;
    }
    // Static function — no object needed
    static int getCount() { return objectCount; }
};

// MUST define outside class!
int ObjectCounter::objectCount = 0;

int main() {
    cout << "Start: " << ObjectCounter::getCount() << endl;  // 0
    ObjectCounter o1;                     // Count: 1
    ObjectCounter o2;                     // Count: 2
    ObjectCounter o3 = o1;                // Copy → Count: 3
    {
        ObjectCounter o4;                 // Count: 4
    }  // o4 destroyed → Count: 3
    cout << "Final: " << ObjectCounter::getCount() << endl;  // 3
}
\`\`\`

| Rule | Explanation |
|------|-------------|
| Define outside class | \`int Counter::count = 0;\` mandatory |
| Shared across objects | All objects see same value |
| Copy constructor | Must also increment count! |
| Call without object | \`ClassName::function()\` |`,
      },
    ],
  },
  {
    id: "inheritance",
    label: "Inheritance",
    icon: "⟢",
    color: "#a78bfa",
    topics: [
      {
        id: 11,
        title: "Inheritance & Types",
        marks: 5,
        tag: "High Weightage",
        content: `**Definition:** A derived class (child) acquires properties from a base class (parent), enabling code reusability.

\`\`\`cpp
class DerivedClass : access_specifier BaseClass { };
\`\`\`

**5 Types:**

**1. Single** — one parent, one child
\`\`\`cpp
class Animal { public: void eat() { } };
class Dog : public Animal { public: void bark() { } };
\`\`\`

**2. Multilevel** — chain of inheritance
\`\`\`cpp
class Animal → class Mammal → class Dog
// Dog inherits from Mammal which inherits from Animal
\`\`\`

**3. Multiple** — inherits from 2+ parents (C++ only)
\`\`\`cpp
class Child : public Father, public Mother { };
\`\`\`

**4. Hierarchical** — one parent, multiple children
\`\`\`cpp
class Shape → Circle, Rectangle, Triangle
\`\`\`

**5. Hybrid** — combination (causes Diamond Problem)

**Access specifiers in inheritance:**

| Inheritance | public member → | protected member → | private member → |
|-------------|-----------------|-------------------|-----------------|
| public | public | protected | inaccessible |
| protected | protected | protected | inaccessible |
| private | private | private | inaccessible |`,
      },
      {
        id: 12,
        title: "Multiple Inheritance & Diamond Problem",
        marks: 6,
        tag: "Important",
        content: `**Multiple Inheritance:** A derived class inherits from more than one base class.

\`\`\`cpp
class Printer { public: void print() { } };
class Scanner { public: void scan() { } };
class AllInOne : public Printer, public Scanner {
    // Inherits both print() and scan()
};
\`\`\`

**Diamond Problem:**
\`\`\`
        Base
       /    \\
      A      B
       \\    /
        Child
\`\`\`

\`\`\`cpp
class Base { public: int value = 10; };
class A : public Base { };
class B : public Base { };
class Child : public A, public B { };

Child c;
// c.value = 20;    // ERROR! Ambiguous — which copy?
c.A::value = 20;    // Must specify path
c.B::value = 30;    // Different copy!
// Two different values for same logical attribute = inconsistency!
\`\`\`

**Solution: Virtual Inheritance**
\`\`\`cpp
class A : virtual public Base { };  // virtual keyword!
class B : virtual public Base { };  // virtual keyword!
class Child : public A, public B { };  // Only ONE Base copy now

Child c;
c.value = 20;       // Unambiguous! Works perfectly.
\`\`\`

| Aspect | Without Virtual | With Virtual |
|--------|----------------|--------------|
| Base copies | 2 copies | 1 copy |
| Ambiguity | Present | Resolved |
| Initialization | Each parent initializes | Most derived initializes Base |

**Important:** With virtual inheritance, the most derived class directly calls the base constructor.`,
      },
      {
        id: 14,
        title: "Method Overloading vs Overriding",
        marks: 5,
        tag: "Comparison",
        content: `| Feature | Overloading | Overriding |
|---------|-------------|------------|
| Definition | Same name, different params, SAME class | Same name + params, CHILD class |
| Binding | Compile-time | Runtime |
| Return type | Can differ | Must be same |
| Keyword | None needed | \`virtual\` + \`override\` |
| Constructor | Can be overloaded | Cannot be overridden |

**Overloading:**
\`\`\`cpp
class Calculator {
public:
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a+b+c; }
    // All decided at COMPILE TIME
};
\`\`\`

**Overriding:**
\`\`\`cpp
class Animal {
public:
    virtual void sound() { cout << "Animal sound\\n"; }
};
class Dog : public Animal {
public:
    void sound() override { cout << "Bark\\n"; }
    // Decided at RUNTIME based on actual object type
};
\`\`\`

**Method Hiding vs Overriding:**
\`\`\`cpp
Parent* p = new Child();
p->nonVirtual();   // "Parent" — HIDING (compile-time)
p->virtualFunc();  // "Child"  — OVERRIDING (runtime)
\`\`\`

**Memory Trick:**
- OverLOAD = same class, LOAD different parameters
- OverRIDE = RIDE over parent's method (different class)`,
      },
      {
        id: 13,
        title: "Java: No Multiple Inheritance (Interfaces)",
        marks: 5,
        tag: "Concept",
        content: `**Why Java rejects multiple inheritance:**
1. **Diamond Problem** — ambiguous method resolution
2. **Simplified design** — James Gosling called it a "design mistake" in C++
3. **Type safety** — fragile base class problems

\`\`\`java
// NOT allowed in Java:
class C extends A, B { }  // ERROR!
\`\`\`

**Java's solution: Interfaces**

\`\`\`java
interface Printable { void print(); }
interface Scannable { void scan(); }
interface Faxable { void fax(); }

// A class CAN implement multiple interfaces
class AllInOnePrinter implements Printable, Scannable, Faxable {
    public void print() { System.out.println("Printing"); }
    public void scan() { System.out.println("Scanning"); }
    public void fax() { System.out.println("Faxing"); }
}
\`\`\`

| Aspect | C++ Multiple Inheritance | Java Interface |
|--------|------------------------|----------------|
| State inheritance | Yes (data members) | No |
| Implementation | Yes (full methods) | Abstract (Java 8+: default) |
| Diamond problem | Exists | Not applicable |
| Constructor inherit | Yes | No |

**Resolving default method conflicts (Java 8+):**
\`\`\`java
interface A { default void test() { System.out.println("A"); } }
interface B { default void test() { System.out.println("B"); } }

class C implements A, B {
    public void test() {
        A.super.test();  // Must override to resolve
    }
}
\`\`\`

**Key takeaway:** Java achieves multiple *type* inheritance (not implementation) via interfaces, avoiding the diamond problem while retaining polymorphism.`,
      },
    ],
  },
  {
    id: "output",
    label: "Output Tracing",
    icon: "▷",
    color: "#f97316",
    topics: [
      {
        id: 19,
        title: "Static Members — Complete Example",
        marks: 5,
        tag: "Program",
        content: `\`\`\`cpp
class Bank {
private:
    string accountHolder;
    double balance;
    static double interestRate;   // Shared
    static int totalAccounts;     // Shared
public:
    Bank(string name, double bal) {
        accountHolder = name;
        balance = bal;
        totalAccounts++;
    }
    ~Bank() { totalAccounts--; }
    
    // Static function — no object needed, only accesses static members
    static void setInterestRate(double r) { interestRate = r; }
    static int getTotalAccounts() { return totalAccounts; }
    static void showStats() {
        cout << "Accounts: " << totalAccounts << endl;
        cout << "Rate: " << interestRate << "%" << endl;
    }
};

// MUST define outside class!
double Bank::interestRate = 5.5;
int Bank::totalAccounts = 0;

int main() {
    cout << Bank::getTotalAccounts() << endl;  // 0 (no objects yet)
    Bank::setInterestRate(6.75);
    
    Bank acc1("Rahim", 5000);   // totalAccounts = 1
    Bank acc2("Karim", 3000);   // totalAccounts = 2
    
    Bank::showStats();  // Accounts: 2, Rate: 6.75%
}
\`\`\`

| Feature | Static Member | Non-Static Member |
|---------|--------------|-------------------|
| Storage | One copy | Per object |
| Access | \`ClassName::member\` | \`obj.member\` |
| this pointer | None | Has it |
| Can be virtual | No | Yes |
| Lifetime | Entire program | Lifetime of object |`,
      },
      {
        id: 20,
        title: "Output Tracing: All Scenarios",
        marks: 6,
        tag: "⚡ Exam Common",
        content: `**Scenario 1 — Constructor/Destructor Order:**
\`\`\`cpp
class Base {
public:
    Base() { cout << "Base()\\n"; }
    ~Base() { cout << "~Base()\\n"; }
};
class Derived : public Base {
public:
    Derived() { cout << "Derived()\\n"; }
    ~Derived() { cout << "~Derived()\\n"; }
};
class MostDerived : public Derived {
public:
    MostDerived() { cout << "MostDerived()\\n"; }
    ~MostDerived() { cout << "~MostDerived()\\n"; }
};
int main() { MostDerived obj; }
\`\`\`
**Output:**
\`\`\`
Base()           ← Construction: top-down
Derived()
MostDerived()
~MostDerived()   ← Destruction: bottom-up (reverse)
~Derived()
~Base()
\`\`\`

**Scenario 2 — Virtual Functions:**
\`\`\`cpp
Animal* ptr = new Dog();
ptr->sound();     // VIRTUAL  → "Bark" (actual object type)
ptr->normalFunc();// NON-VIRT → "Animal" (pointer type)
\`\`\`

**Scenario 3 — Static Object Count:**

| Step | Action | Count | Why |
|------|--------|-------|-----|
| 1 | Start | 0 | No objects |
| 2 | \`Counter c1\` | 1 | Default ctor |
| 3 | \`Counter c2 = c1\` | 2 | Copy ctor |
| 4 | Block start, \`c3\` | 3 | Default ctor |
| 5 | Block end | 2 | c3 destroyed |
| 6 | \`func(c1)\` called | 3 | Copy to param |
| 7 | Return value copy | 4 | Copy |
| 8 | Param+return destroyed | 2 | Both destroyed |

**Quick Tracing Cheat Sheet:**

| Scenario | Rule |
|----------|------|
| Constructor order | Base → Derived → MostDerived |
| Destructor order | MostDerived → Derived → Base |
| Virtual function | Based on ACTUAL object type |
| Non-virtual function | Based on POINTER/REFERENCE type |
| Copy constructor | Pass by value, return by value, direct copy |
| Virtual destructor | Ensures proper cleanup in hierarchy |`,
      },
      {
        id: 9,
        title: "Constructor Overloading & Chaining",
        marks: 5,
        tag: "Program",
        content: `**Constructor Overloading:** Multiple constructors with different parameter lists.

\`\`\`cpp
class Rectangle {
    int length, width;
public:
    Rectangle() { length = width = 1; }           // Default
    Rectangle(int side) { length = width = side; } // Square
    Rectangle(int l, int w) { length=l; width=w; } // Rectangle
    int area() { return length * width; }
};

int main() {
    Rectangle r1;       // Default → 1x1,   area=1
    Rectangle r2(5);    // Square  → 5x5,   area=25
    Rectangle r3(4, 6); // Custom  → 4x6,   area=24
}
\`\`\`

**Constructor Chaining (C++11 Delegating Constructors):**
\`\`\`cpp
class Point {
    int x, y;
public:
    // Master constructor
    Point(int xVal, int yVal) : x(xVal), y(yVal) {
        cout << "Point(" << x << "," << y << ")\\n";
    }
    
    // Delegating constructors — chain to master
    Point() : Point(0, 0) { }          // Default (0,0)
    Point(int val) : Point(val, val) { } // Square point
};

int main() {
    Point p1;        // → Point(0, 0)
    Point p2(5);     // → Point(5, 5)
    Point p3(3, 7);  // → Point(3, 7) directly
}
\`\`\`

**Benefits of chaining:**
- Reduces code duplication
- Change one master constructor → all updated
- More consistent initialization

**Pre-C++11 approach (using init function):**
\`\`\`cpp
class Student {
    void init(string n, int r, double c) {
        name=n; roll=r; cgpa=c;         // Common init logic
    }
public:
    Student() { init("Unknown", 0, 0.0); }
    Student(string n) { init(n, 0, 0.0); }
    Student(string n, int r, double c) { init(n, r, c); }
};
\`\`\``,
      },
      {
        id: 15,
        title: "Access Specifiers in Inheritance",
        marks: 5,
        tag: "Reference",
        content: `**Three access specifiers:**

| Specifier | Within class | In derived class | Outside class |
|-----------|-------------|-----------------|---------------|
| public | ✓ | ✓ | ✓ |
| protected | ✓ | ✓ | ✗ |
| private | ✓ | ✗ | ✗ |

**Effect of inheritance type:**

\`\`\`cpp
class Base {
private:   int priv = 1;    // Private
protected: int prot = 2;    // Protected
public:    int pub  = 3;    // Public
};
\`\`\`

**public inheritance** → \`class Derived : public Base\`
\`\`\`cpp
// priv → still inaccessible in derived
// prot → remains protected
// pub  → remains public
\`\`\`

**protected inheritance** → \`class Derived : protected Base\`
\`\`\`cpp
// priv → inaccessible
// prot → remains protected
// pub  → becomes protected (downgraded!)
\`\`\`

**private inheritance** → \`class Derived : private Base\`
\`\`\`cpp
// priv → inaccessible
// prot → becomes private
// pub  → becomes private
// GrandChild cannot access any of them!
\`\`\`

**Summary table:**

| Base Member | public inherit | protected inherit | private inherit |
|-------------|---------------|-------------------|-----------------|
| public | public | protected | private |
| protected | protected | protected | private |
| private | inaccessible | inaccessible | inaccessible |

**Choosing inheritance type:**
- \`public\` → "is-a" relationship *(Dog is an Animal)*
- \`protected\` → Rarely used, hierarchical sharing
- \`private\` → "implemented-in-terms-of" *(Stack using Vector)*`,
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
      if (part.startsWith("`") && part.endsWith("`")) {
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
      if (part.startsWith("**") && part.endsWith("**")) {
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

    // Code block
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
                  .replace(
                    /(\/\/[^\n]*)/g,
                    '<span style="color:#6a9955">$1</span>'
                  )
                  .replace(
                    /\b(class|public|private|protected|virtual|override|static|const|int|double|string|void|return|new|delete|bool|char|float|nullptr|true|false|if|else|for|while|cout|cin|endl|using|namespace|include)\b/g,
                    '<span style="color:#569cd6">$1</span>'
                  )
                  .replace(
                    /"([^"]*)"/g,
                    '<span style="color:#ce9178">"$1"</span>'
                  ),
              }}
            />
          </pre>
        </div>
      );
      i++;
      continue;
    }

    // Table
    if (line.trim().startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const rows = tableLines.filter((l) => !l.match(/^\|[-|\s]+\|$/));
      elements.push(
        <div
          key={key()}
          style={{ margin: "12px 0", overflowX: "auto" }}
        >
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

    // Headings
    if (line.startsWith("**") && line.endsWith("**") && line.length > 4) {
      elements.push(
        <p
          key={key()}
          style={{
            fontWeight: 700,
            color: "#fff",
            margin: "14px 0 4px",
            fontSize: "0.92em",
          }}
        >
          {line.slice(2, -2)}
        </p>
      );
      i++;
      continue;
    }

    // List items
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
          <span style={{ color: "rgba(255,255,255,0.3)", flexShrink: 0 }}>
            ›
          </span>
          <span>{renderInline(line.replace(/^[-*]\s/, ""))}</span>
        </div>
      );
      i++;
      continue;
    }

    // Blank line
    if (line.trim() === "") {
      elements.push(<div key={key()} style={{ height: "6px" }} />);
      i++;
      continue;
    }

    // Regular paragraph
    elements.push(
      <p
        key={key()}
        style={{
          color: "rgba(255,255,255,0.75)",
          fontSize: "0.86em",
          lineHeight: 1.65,
          margin: "4px 0",
        }}
      >
        {renderInline(line)}
      </p>
    );
    i++;
  }

  return elements;
}

export default function OOPStudyGuide() {
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

  const openTopic = activeTopic !== null
    ? chapters.flatMap((c) => c.topics).find((t) => t.id === activeTopic)
    : null;
  const openTopicColor = activeTopic !== null
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
        .topic-card:hover { transform: translateY(-2px); }
        .nav-tab:hover { background: rgba(255,255,255,0.06) !important; }
        .close-btn:hover { background: rgba(255,255,255,0.1) !important; }
      `}</style>

      {/* Header */}
      <div
        style={{
          padding: "28px 32px 20px",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(255,255,255,0.02)",
          backdropFilter: "blur(10px)",
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
                  background: "linear-gradient(135deg, #00d4aa, #a78bfa)",
                  boxShadow: "0 0 10px #00d4aa60",
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
                CSE — Object Oriented Programming
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
              Complete Study Guide
            </h1>
          </div>

          {/* Search */}
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

          {/* Stats */}
          <div style={{ display: "flex", gap: "16px" }}>
            {[
              { label: "Topics", value: chapters.flatMap((c) => c.topics).length },
              { label: "Chapters", value: chapters.length },
            ].map((s) => (
              <div key={s.label} style={{ textAlign: "center" }}>
                <div style={{ fontSize: "1.3em", fontWeight: 700, color: "#00d4aa" }}>
                  {s.value}
                </div>
                <div style={{ fontSize: "0.65em", color: "rgba(255,255,255,0.35)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
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
                    activeChapter === ch.id
                      ? `${ch.color}20`
                      : "rgba(255,255,255,0.04)",
                  border: `1px solid ${
                    activeChapter === ch.id ? ch.color + "60" : "rgba(255,255,255,0.08)"
                  }`,
                  borderRadius: "8px",
                  padding: "8px 16px",
                  color:
                    activeChapter === ch.id ? ch.color : "rgba(255,255,255,0.55)",
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
                  View answer
                  <span style={{ fontSize: "0.9em" }}>→</span>
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
              <div
                style={{
                  flex: 1,
                  height: "1px",
                  background: "rgba(255,255,255,0.06)",
                }}
              />
              <span
                style={{
                  fontSize: "0.75em",
                  color: "rgba(255,255,255,0.3)",
                }}
              >
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
              {/* Topic header */}
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
                  <h2
                    style={{
                      margin: 0,
                      fontSize: "1.2em",
                      fontWeight: 700,
                      color: "#fff",
                    }}
                  >
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

              {/* Content */}
              <div style={{ padding: "24px" }}>
                {renderContent(openTopic.content)}
              </div>
            </div>

            {/* Prev / Next */}
            <div
              style={{
                display: "flex",
                gap: "10px",
                marginTop: "16px",
              }}
            >
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