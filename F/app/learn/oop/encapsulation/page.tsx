"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { MultiLanguageCode, CodeBlock } from "@/components/code-block";
import { EncapsulationVisualizer } from "@/components/visualizations/oop-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import {
  BookOpen,
  Lightbulb,
  AlertTriangle,
  Code,
  Target,
  Lock,
  Unlock,
  Shield,
} from "lucide-react";

const pythonCode = `# Encapsulation in Python

class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner           # Public attribute
        self._account_type = "Savings"  # Protected (convention)
        self.__balance = initial_balance  # Private (name mangling)
    
    # Public methods - the interface
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited \${amount}. New balance: \${self.__balance}"
        return "Invalid amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return f"Withdrew \${amount}. New balance: \${self.__balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):  # Getter
        return self.__balance
    
    # Using @property decorator (Pythonic way)
    @property
    def balance(self):
        return self.__balance
    
    @balance.setter
    def balance(self, value):
        if value >= 0:
            self.__balance = value
        else:
            raise ValueError("Balance cannot be negative")

# Usage
account = BankAccount("Alice", 1000)

# Public access works
print(account.owner)       # Alice
print(account.deposit(500))  # Deposited $500. New balance: $1500

# Direct access to private fails
# print(account.__balance)  # AttributeError!

# Use public methods instead
print(account.get_balance())  # 1500
print(account.balance)        # 1500 (using property)

# Attempting to set invalid balance
# account.balance = -100  # ValueError: Balance cannot be negative`;

const javascriptCode = `// Encapsulation in JavaScript (ES2022+)

class BankAccount {
    // Public field
    owner;
    
    // Private fields (use # prefix)
    #balance;
    #accountType;
    
    constructor(owner, initialBalance = 0) {
        this.owner = owner;
        this.#balance = initialBalance;
        this.#accountType = "Savings";
    }
    
    // Public methods - the interface
    deposit(amount) {
        if (amount > 0) {
            this.#balance += amount;
            return \`Deposited \$\${amount}. New balance: \$\${this.#balance}\`;
        }
        return "Invalid amount";
    }
    
    withdraw(amount) {
        if (amount > 0 && amount <= this.#balance) {
            this.#balance -= amount;
            return \`Withdrew \$\${amount}. New balance: \$\${this.#balance}\`;
        }
        return "Insufficient funds or invalid amount";
    }
    
    // Getter
    get balance() {
        return this.#balance;
    }
    
    // Setter with validation
    set balance(value) {
        if (value >= 0) {
            this.#balance = value;
        } else {
            throw new Error("Balance cannot be negative");
        }
    }
}

// Usage
const account = new BankAccount("Alice", 1000);

console.log(account.owner);      // Alice
console.log(account.deposit(500)); // Deposited $500...

// Direct access to private fails
// console.log(account.#balance);  // SyntaxError!

// Use getter instead
console.log(account.balance);    // 1500`;

const javaCode = `// Encapsulation in Java

public class BankAccount {
    // Private fields - hidden from outside
    private double balance;
    private String accountType;
    
    // Public field (not recommended, shown for comparison)
    public String owner;
    
    // Constructor
    public BankAccount(String owner, double initialBalance) {
        this.owner = owner;
        this.balance = initialBalance;
        this.accountType = "Savings";
    }
    
    // Public methods - controlled access
    public String deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return "Deposited $" + amount + ". New balance: $" + balance;
        }
        return "Invalid amount";
    }
    
    public String withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return "Withdrew $" + amount + ". New balance: $" + balance;
        }
        return "Insufficient funds or invalid amount";
    }
    
    // Getter
    public double getBalance() {
        return balance;
    }
    
    // Setter with validation
    public void setBalance(double value) {
        if (value >= 0) {
            balance = value;
        } else {
            throw new IllegalArgumentException("Balance cannot be negative");
        }
    }
}

// Usage
public class Main {
    public static void main(String[] args) {
        BankAccount account = new BankAccount("Alice", 1000);
        
        System.out.println(account.owner);       // Alice
        System.out.println(account.deposit(500)); // Deposited...
        
        // Direct access fails
        // System.out.println(account.balance);  // Error!
        
        // Use getter
        System.out.println(account.getBalance()); // 1500
    }
}`;

const cppCode = `// Encapsulation in C++
#include <iostream>
#include <string>
#include <stdexcept>
using namespace std;

class BankAccount {
private:
    // Private members - hidden from outside
    double balance;
    string accountType;

public:
    // Public member (not recommended, shown for comparison)
    string owner;
    
    // Constructor
    BankAccount(string owner, double initialBalance = 0) 
        : owner(owner), balance(initialBalance), accountType("Savings") {}
    
    // Public methods - controlled access
    string deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return "Deposited $" + to_string(amount) + 
                   ". New balance: $" + to_string(balance);
        }
        return "Invalid amount";
    }
    
    string withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return "Withdrew $" + to_string(amount) + 
                   ". New balance: $" + to_string(balance);
        }
        return "Insufficient funds or invalid amount";
    }
    
    // Getter
    double getBalance() const {
        return balance;
    }
    
    // Setter with validation
    void setBalance(double value) {
        if (value >= 0) {
            balance = value;
        } else {
            throw invalid_argument("Balance cannot be negative");
        }
    }
};

int main() {
    BankAccount account("Alice", 1000);
    
    cout << account.owner << endl;       // Alice
    cout << account.deposit(500) << endl; // Deposited...
    
    // Direct access fails
    // cout << account.balance;  // Error: private member
    
    // Use getter
    cout << account.getBalance() << endl; // 1500
    
    return 0;
}`;

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is encapsulation in OOP?",
    options: [
      "Inheriting from multiple classes",
      "Bundling data and methods together while restricting direct access to internal data",
      "Creating multiple objects from a class",
      "Having methods with the same name",
    ],
    correctAnswer: 1,
    explanation:
      "Encapsulation bundles data (attributes) and methods together in a class while hiding internal implementation details and providing controlled access through public methods.",
  },
  {
    id: 2,
    question: "What is the purpose of making attributes private?",
    options: [
      "To make the code run faster",
      "To prevent direct modification and ensure data integrity through validation",
      "To use less memory",
      "To allow inheritance",
    ],
    correctAnswer: 1,
    explanation:
      "Private attributes prevent direct access from outside the class, ensuring data can only be modified through validated methods, maintaining data integrity.",
  },
  {
    id: 3,
    question: "In Python, how do you indicate a private attribute?",
    options: [
      "Using the 'private' keyword",
      "Prefix with double underscore (__)",
      "Prefix with 'p_'",
      "Using @private decorator",
    ],
    correctAnswer: 1,
    explanation:
      "In Python, double underscore prefix (__variable) triggers name mangling, making the attribute harder to access from outside. Single underscore (_variable) is a convention for 'protected'.",
  },
  {
    id: 4,
    question: "What are getters and setters used for?",
    options: [
      "To create new objects",
      "To provide controlled access to private attributes",
      "To delete objects",
      "To inherit methods",
    ],
    correctAnswer: 1,
    explanation:
      "Getters retrieve private attribute values, and setters allow modifying them with validation. They provide controlled access while keeping the actual data private.",
  },
  {
    id: 5,
    question: "Which access modifier allows access only within the same class?",
    options: [
      "public",
      "protected",
      "private",
      "internal",
    ],
    correctAnswer: 2,
    explanation:
      "Private access modifier restricts access to within the same class only. Public allows access from anywhere, and protected allows access from the class and its subclasses.",
  },
  {
    id: 6,
    question: "Why is encapsulation important for large codebases?",
    options: [
      "It makes code compile faster",
      "It allows changing internal implementation without affecting code that uses the class",
      "It reduces the number of classes needed",
      "It eliminates the need for testing",
    ],
    correctAnswer: 1,
    explanation:
      "Encapsulation creates a clear interface. You can change how data is stored or validated internally without breaking code that uses the class, as long as the public interface stays the same.",
  },
];

export default function EncapsulationPage() {
  const result = getSubtopicBySlug("oop", "encapsulation");

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
            What is Encapsulation?
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">Encapsulation</strong> is the bundling of data 
            (attributes) and methods that operate on that data within a single unit (class), 
            while restricting direct access to some components.
          </p>
          <p>
            It's like a protective wrapper that prevents external code from directly accessing 
            internal data, instead providing controlled access through public methods.
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
                Think of an <strong className="text-foreground">ATM machine</strong>. You can't directly 
                access the cash inside - you must use the interface (card slot, PIN pad, buttons). 
                The ATM validates your request and controls how much you can withdraw. The internal 
                mechanisms are hidden (encapsulated) from you.
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
        <EncapsulationVisualizer />
      </section>

      {/* Access Modifiers */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">
          Access Modifiers
        </h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-card rounded-lg border-2 border-green-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Unlock className="h-5 w-5 text-green-500" />
              <h3 className="font-semibold text-foreground">Public</h3>
            </div>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Accessible from anywhere</li>
              <li>• No restrictions</li>
              <li>• Python: <code className="bg-muted px-1 rounded">name</code></li>
              <li>• Java/C++: <code className="bg-muted px-1 rounded">public</code></li>
            </ul>
          </div>
          <div className="p-4 bg-card rounded-lg border-2 border-yellow-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="h-5 w-5 text-yellow-500" />
              <h3 className="font-semibold text-foreground">Protected</h3>
            </div>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Class + subclasses</li>
              <li>• Inherited classes can access</li>
              <li>• Python: <code className="bg-muted px-1 rounded">_name</code></li>
              <li>• Java/C++: <code className="bg-muted px-1 rounded">protected</code></li>
            </ul>
          </div>
          <div className="p-4 bg-card rounded-lg border-2 border-red-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Lock className="h-5 w-5 text-red-500" />
              <h3 className="font-semibold text-foreground">Private</h3>
            </div>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>• Only within the class</li>
              <li>• Most restrictive</li>
              <li>• Python: <code className="bg-muted px-1 rounded">__name</code></li>
              <li>• Java/C++: <code className="bg-muted px-1 rounded">private</code></li>
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

      {/* Benefits */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-4 text-foreground">Benefits of Encapsulation</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Data Protection</h3>
            <p className="text-sm text-muted-foreground">
              Prevents accidental or malicious modification of data. Balance can't be set 
              to negative values if the setter validates it.
            </p>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Flexibility</h3>
            <p className="text-sm text-muted-foreground">
              Internal implementation can change without affecting external code. You could 
              change from storing cents to dollars internally.
            </p>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Easier Debugging</h3>
            <p className="text-sm text-muted-foreground">
              If data becomes invalid, you know it happened through a setter - easier to 
              find the bug than if any code could modify the data.
            </p>
          </div>
          <div className="p-4 bg-card rounded-lg border border-border">
            <h3 className="font-semibold text-foreground mb-2">Clear Interface</h3>
            <p className="text-sm text-muted-foreground">
              Public methods define a clear API. Users of the class know exactly 
              what operations are available and supported.
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
              Making everything public
            </h4>
            <p className="text-sm text-muted-foreground">
              Exposing all attributes as public defeats the purpose of encapsulation. Think about 
              what really needs to be accessible from outside.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Getters/setters without validation
            </h4>
            <p className="text-sm text-muted-foreground">
              If your setter just assigns the value without validation, you might as well 
              make the attribute public. Add meaningful validation logic.
            </p>
          </div>
          <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <h4 className="font-medium text-foreground mb-1">
              Returning mutable objects directly
            </h4>
            <p className="text-sm text-muted-foreground">
              If you return a list or object reference, the caller can modify it. Return 
              a copy instead: <code className="bg-muted px-1 rounded">return self.__items.copy()</code>
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
                "Why use getters and setters instead of public attributes?"
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-success font-bold">A:</span>
              <span className="text-foreground">
                "Getters/setters provide controlled access: you can add validation in setters 
                (e.g., age must be positive), add logging, compute derived values in getters, 
                or change internal representation later without breaking the API."
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
        <Quiz questions={quizQuestions} title="Encapsulation Quiz" />
      </section>
    </TopicContent>
  );
}
