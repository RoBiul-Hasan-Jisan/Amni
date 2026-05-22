"use client";

import * as React from "react";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock } from "@/components/code-block";
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
  Box,
  Link,
  Eye,
} from "lucide-react";

const quizQuestions: QuizQuestion[] = [
  {
    id: 1,
    question: "What is the difference between aggregation and composition?",
    options: [
      "They are the same",
      "Aggregation: weak 'has-a', Composition: strong 'has-a' with ownership",
      "Composition is for interfaces, aggregation for classes",
      "Aggregation requires inheritance",
    ],
    correctAnswer: 1,
    explanation: "Aggregation represents a weaker 'has-a' relationship where parts can exist independently. Composition is stronger where parts cannot exist without the whole.",
  },
  {
    id: 2,
    question: "What does a use case diagram show?",
    options: [
      "Class relationships",
      "System functionality from user perspective",
      "Object interactions over time",
      "State changes of an object",
    ],
    correctAnswer: 1,
    explanation: "Use case diagrams show system functionality from an external user's perspective, showing actors and use cases.",
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
            <span className="text-sm font-medium text-muted-foreground">Diagram & Explanation:</span>
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

export default function UMLAndOOADPage() {
  const result = getSubtopicBySlug("oop", "uml-ooad");

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
      <section className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-primary/10">
            <BookOpen className="h-5 w-5 text-primary" />
          </div>
          <h2 className="text-2xl font-bold text-foreground">
            UML & Object-Oriented Analysis and Design (OOAD)
          </h2>
        </div>
        <div className="space-y-4 text-muted-foreground">
          <p>
            <strong className="text-foreground">UML (Unified Modeling Language)</strong> is a standardized 
            visual language for modeling software systems. It provides various diagram types to represent 
            different aspects of a system.
          </p>
        </div>
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

        {/* Question 60: Library Management System Class Diagram */}
        <QuestionCard
          number={60}
          title="Library Management System Class Diagram"
          question="Draw a class diagram for a Library Management System."
          answer="A Library Management System class diagram includes classes: Library, Book, Member, Librarian, Transaction, Category. Relationships: Library has Books and Members (composition), Member borrows Books (association), Librarian manages Transactions (association), Book belongs to Category (aggregation). Includes attributes like title, author, ISBN, memberId, issueDate, returnDate, and methods like borrowBook(), returnBook(), searchBook()."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-3">Library Management System - Class Diagram</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`┌─────────────────────────┐    ┌─────────────────────────┐
│        Library          │    │         Book            │
├─────────────────────────┤    ├─────────────────────────┤
│ - name: String          │1  *│ - bookId: int           │
│ - address: String       │───▶│ - title: String         │
│ - phone: String         │    │ - author: String        │
├─────────────────────────┤    │ - isbn: String          │
│ + addBook()             │    │ - category: Category    │
│ + registerMember()      │    │ - isAvailable: boolean  │
│ + removeBook()          │    ├─────────────────────────┤
└─────────────────────────┘    │ + borrow()              │
                               │ + return()              │
                               │ + getDetails()          │
                               └───────────┬─────────────┘
                                           │
                               ┌───────────▼─────────────┐
                               │        Category         │
                               ├─────────────────────────┤
                               │ - categoryId: int       │
                               │ - name: String          │
                               │ - description: String   │
                               ├─────────────────────────┤
                               │ + addBook()             │
                               │ + removeBook()          │
                               └─────────────────────────┘

┌─────────────────────────┐    ┌─────────────────────────┐
│        Member           │    │      Transaction        │
├─────────────────────────┤    ├─────────────────────────┤
│ - memberId: int         │1  *│ - transactionId: int    │
│ - name: String          │───▶│ - issueDate: Date       │
│ - email: String         │    │ - dueDate: Date         │
│ - phone: String         │    │ - returnDate: Date      │
│ - membershipType: String│    │ - fine: double          │
├─────────────────────────┤    │ - status: String        │
│ + borrowBook()          │    ├─────────────────────────┤
│ + returnBook()          │    │ + calculateFine()       │
│ + payFine()             │    │ + issueBook()           │
│ + viewBorrowedBooks()   │    │ + returnBook()          │
└───────────┬─────────────┘    └───────────┬─────────────┘
            │                              │
            │ 1                           *│
            └──────────┬──────────────────┘
                       │
            ┌──────────▼──────────┐
            │     Librarian       │
            ├─────────────────────┤
            │ - staffId: int      │
            │ - name: String      │
            │ - department: String│
            ├─────────────────────┤
            │ + issueBook()       │
            │ + returnBook()      │
            │ + addMember()       │
            │ + generateReport()  │
            └─────────────────────┘

Relationships:
• Library (1) --- (1..*) Book (Composition)
• Library (1) --- (1..*) Member (Composition)
• Member (1) --- (0..*) Transaction (Association)
• Book (1) --- (0..*) Transaction (Association)
• Book (*) --- (1) Category (Aggregation)
• Librarian (1) --- (0..*) Transaction (Association)`}
            </pre>
            <div className="mt-3 text-sm text-muted-foreground">
              <p><strong>Key UML Notations:</strong></p>
              <ul className="list-disc list-inside">
                <li>◊ (diamond) - Composition (solid) / Aggregation (hollow)</li>
                <li>→ (arrow) - Association / Inheritance</li>
                <li>Numbers (1, *, 0..*) - Multiplicity</li>
                <li>- (minus) - Private members</li>
                <li>+ (plus) - Public methods</li>
              </ul>
            </div>
          </div>
        </QuestionCard>

        {/* Question 61: Online Shopping System Class Diagram */}
        <QuestionCard
          number={61}
          title="Online Shopping System Class Diagram"
          question="Draw a class diagram for an Online Shopping System."
          answer="Online Shopping System classes: User, Customer, Admin, Product, Category, ShoppingCart, Order, Payment, Address. Relationships: User has ShoppingCart (composition), Customer places Orders (association), Order contains Products (aggregation), Payment belongs to Order (composition). Includes attributes like product price, order status, payment method, and methods like addToCart(), checkout(), processPayment(), trackOrder()."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-3">Online Shopping System - Class Diagram</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`┌─────────────────────────┐
│          User           │
├─────────────────────────┤
│ - userId: int           │
│ - name: String          │
│ - email: String         │
│ - password: String      │
├─────────────────────────┤
│ + login()               │
│ + logout()              │
│ + register()            │
└───────────┬─────────────┘
            │
            │ inheritance
            ├──────────────────┐
            │                  │
┌───────────▼───────────┐ ┌────▼────────────────────┐
│       Customer        │ │         Admin           │
├───────────────────────┤ ├─────────────────────────┤
│ - customerId: int     │ │ - adminId: int          │
│ - loyaltyPoints: int  │ │ - role: String          │
├───────────────────────┤ ├─────────────────────────┤
│ + addToCart()         │ │ + addProduct()          │
│ + viewOrders()        │ │ + removeProduct()       │
│ + writeReview()       │ │ + manageUsers()         │
│ + trackOrder()        │ │ + generateSalesReport() │
└───────┬───────────────┘ └─────────────────────────┘
        │
        │ 1
        │
┌───────▼───────────────┐    ┌─────────────────────────┐
│     ShoppingCart      │    │         Product         │
├───────────────────────┤    ├─────────────────────────┤
│ - cartId: int         │1  *│ - productId: int        │
│ - creationDate: Date  │───▶│ - name: String          │
│ - totalAmount: double │    │ - description: String   │
├───────────────────────┤    │ - price: double         │
│ + addItem()           │    │ - stock: int            │
│ + removeItem()        │    │ - imageUrl: String      │
│ + updateQuantity()    │    ├─────────────────────────┤
│ + calculateTotal()    │    │ + getDetails()          │
│ + checkout()          │    │ + updateStock()         │
└───────┬───────────────┘    │ + applyDiscount()       │
        │                    └───────────┬─────────────┘
        │ 1                              │
        │                                │ * (aggregation)
┌───────▼───────────────┐    ┌───────────▼─────────────┐
│        Order          │    │        Category         │
├───────────────────────┤    ├─────────────────────────┤
│ - orderId: int        │    │ - categoryId: int       │
│ - orderDate: Date     │    │ - name: String          │
│ - status: String      │    │ - parentCategory: Cat   │
│ - shippingAddress:    │    ├─────────────────────────┤
├───────────────────────┤    │ + getProducts()         │
│ + calculateTotal()    │    │ + addSubCategory()      │
│ + cancelOrder()       │    └─────────────────────────┘
│ + updateStatus()      │
│ + trackOrder()        │
└───────┬───────────────┘
        │
        │ 1
        │
┌───────▼───────────────┐    ┌─────────────────────────┐
│       Payment         │    │        Review           │
├───────────────────────┤    ├─────────────────────────┤
│ - paymentId: int      │    │ - reviewId: int         │
│ - amount: double      │    │ - rating: int           │
│ - paymentDate: Date   │    │ - comment: String       │
│ - method: String      │    │ - date: Date            │
│ - status: String      │    ├─────────────────────────┤
├───────────────────────┤    │ + addReview()           │
│ + processPayment()    │    │ + editReview()          │
│ + refund()            │    │ + deleteReview()        │
│ + verifyPayment()     │    └─────────────────────────┘
└───────────────────────┘

Relationships:
• User (1) --- (1) ShoppingCart (Composition)
• User (1) --- (0..*) Order (Association)
• Customer (1) --- (0..*) Review (Association)
• Order (1) --- (1) Payment (Composition)
• Order (*) --- (*) Product (Many-to-Many via OrderItem)
• Product (*) --- (1) Category (Aggregation)`}
            </pre>
          </div>
        </QuestionCard>

        {/* Question 62: Use Case Diagram */}
        <QuestionCard
          number={62}
          title="Use Case Diagram"
          question="What is a use case diagram? Explain the relationship between actor and use case."
          answer="A use case diagram shows system functionality from an external user's perspective. Actors represent users or external systems interacting with the system. Use cases represent specific functionalities. Relationships: Association (between actor and use case), Include (one use case calls another), Extend (optional behavior), Generalization (actor inheritance)."
          marks={5}
          icon={<Eye className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-3">Use Case Diagram - Online Banking System</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`                          ┌─────────────────────────────────────────────────┐
                          │            Online Banking System                │
                          │                                                  │
    ┌─────────────┐       │    ┌──────────────┐        ┌──────────────┐    │
    │             │       │    │              │        │              │    │
    │  Customer   │───────┼───▶│  Login       │        │  Check       │    │
    │             │       │    │              │        │  Balance     │    │
    └─────────────┘       │    └──────────────┘        └──────┬───────┘    │
           │              │                                   │            │
           │              │                          ┌────────▼────────┐   │
           │              │                          │                 │   │
           │              │    ┌──────────────┐      │  View           │   │
           │              │    │              │      │  Transactions   │   │
           ├──────────────┼───▶│  Transfer    │      │                 │   │
           │              │    │  Funds       │      └─────────────────┘   │
           │              │    └──────┬───────┘                             │
           │              │           │                                     │
           │              │           │ <<include>>                         │
           │              │           │                                     │
           │              │    ┌──────▼───────┐                             │
           │              │    │              │        ┌──────────────┐    │
           │              │    │  Verify      │        │              │    │
           │              │    │  Account     │        │  Pay Bills   │    │
           │              │    │              │        │              │    │
           │              │    └──────────────┘        └──────────────┘    │
           │              │                                                  │
    ┌─────────────┐       │    ┌──────────────┐        ┌──────────────┐    │
    │             │       │    │              │        │              │    │
    │   Admin     │───────┼───▶│  Manage       │        │  Generate    │    │
    │             │       │    │  Users       │        │  Reports     │    │
    └─────────────┘       │    └──────────────┘        └──────────────┘    │
                          │                                                  │
                          │    ┌──────────────┐        ┌──────────────┐    │
                          │    │              │        │              │    │
                          │    │  Manage       │        │  Audit       │    │
                          │    │  System       │        │  Logs        │    │
                          │    │              │        │              │    │
                          │    └──────────────┘        └──────────────┘    │
                          │                                                  │
                          └─────────────────────────────────────────────────┘

Relationship Types:

1. Association (Actor ↔ Use Case): Solid line connecting actor to use case
2. Include (<<include>>): One use case always includes another
   Example: Transfer Funds includes Verify Account
3. Extend (<<extend>>): Optional extension of use case behavior
4. Generalization (Actor inheritance): Child actor inherits from parent actor

Key Elements:
• Actor: Stick figure (Customer, Admin) - external entity
• Use Case: Oval (Login, Transfer Funds) - system functionality
• System Boundary: Rectangle - system scope`}
            </pre>
          </div>
        </QuestionCard>

        {/* Question 63: Sequence Diagram */}
        <QuestionCard
          number={63}
          title="Sequence Diagram"
          question="What is a sequence diagram? Draw one for a login process."
          answer="A sequence diagram shows object interactions arranged in time sequence. It displays the flow of messages between objects. For a login process: User enters credentials, LoginController validates, UserService checks database, returns success/failure. Sequence diagrams show lifelines (vertical dashed lines), activation bars, and messages (arrows)."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-3">Sequence Diagram - User Login Process</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`User        LoginView      LoginController    UserService      Database
 │              │                  │                │               │
 │ 1. enter     │                  │                │               │
 │─────────────▶│                  │                │               │
 │  credentials │                  │                │               │
 │              │                  │                │               │
 │              │ 2. validate      │                │               │
 │              │   credentials    │                │               │
 │              │─────────────────▶│                │               │
 │              │                  │                │               │
 │              │                  │ 3. authenticate│               │
 │              │                  │   (username,   │               │
 │              │                  │    password)   │               │
 │              │                  │───────────────▶│               │
 │              │                  │                │               │
 │              │                  │                │ 4. queryUser()│
 │              │                  │                │──────────────▶│
 │              │                  │                │               │
 │              │                  │                │ 5. user data  │
 │              │                  │                │◁──────────────│
 │              │                  │                │               │
 │              │                  │ 6. validation  │               │
 │              │                  │   result       │               │
 │              │                  │◁───────────────│               │
 │              │                  │                │               │
 │              │ 7. login         │                │               │
 │              │   response       │                │               │
 │              │◁─────────────────│                │               │
 │              │                  │                │               │
 │ 8. show      │                  │                │               │
 │   dashboard  │                  │                │               │
 │◁─────────────│                  │                │               │
 │              │                  │                │               │

Alternative flow (Invalid credentials):
 │              │                  │                │               │
 │              │                  │ 3. authenticate│               │
 │              │                  │───────────────▶│               │
 │              │                  │                │               │
 │              │                  │ 6. invalid     │               │
 │              │                  │   credentials  │               │
 │              │                  │◁───────────────│               │
 │              │                  │                │               │
 │              │ 7. error message │                │               │
 │              │◁─────────────────│                │               │
 │ 8. show      │                  │                │               │
 │   error      │                  │                │               │
 │◁─────────────│                  │                │               │

Legend:
▶─ Solid arrow   = Synchronous message
◁─ Dashed arrow  = Return message
│  Vertical line = Lifeline
┌┐ Rectangles     = Activation bars`}
            </pre>
          </div>
        </QuestionCard>

        {/* Question 64: State Diagram */}
        <QuestionCard
          number={64}
          title="State Diagram"
          question="What is a state diagram? Draw one for a traffic light or a vending machine."
          answer="A state diagram shows the states of an object and transitions between states based on events. Traffic light states: Red, Red-Yellow, Green, Yellow. Transitions triggered by timer events. Vending machine states: Idle, CoinInserted, ProductSelected, Dispensing, OutOfStock. Transitions: insert coin, select product, dispense, refill."
          marks={5}
          icon={<Box className="h-3 w-3" />}
        >
        
            <h4 className="font-medium text-foreground mb-3">State Diagram - Traffic Light System</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`                              ┌─────────────────────────────────────────┐
                              │           Traffic Light FSM             │
                              └─────────────────────────────────────────┘

                              Timer: 30 sec
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
              ┌──────────┐      Timer: 3 sec      ┌──────────┐
              │          │                        │          │
              │   RED    │───────────────────────▶│RED-YELLOW│
              │          │                        │          │
              └──────────┘                        └────┬─────┘
                   │                                    │
                   │ Timer: 30 sec                      │ Timer: 3 sec
                   │                                    │
                   │                                    ▼
              ┌──────────┐                        ┌──────────┐
              │          │                        │          │
              │  GREEN   │◁───────────────────────│  YELLOW  │
              │          │                        │          │
              └──────────┘                        └──────────┘
                   ▲                                    │
                   │                                    │
                   └────────────────────────────────────┘
                              Timer: 30 sec

States:      Events (Triggers):
• Red        • Timer expires
• Red-Yellow • Emergency override
• Green      • Power failure
• Yellow

Entry/Exit Actions:
• Entry/Red: Stop all traffic
• Entry/Green: Go signal
• Entry/Yellow: Warning, prepare to stop`}
            </pre>
            <div className="mt-4">
              <h4 className="font-medium text-foreground mb-2">State Diagram - Vending Machine</h4>
              <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`                              ┌─────────────────────────────────────────┐
                              │          Vending Machine FSM            │
                              └─────────────────────────────────────────┘

                    insert coin
                    (amount < price)
              ┌─────────────────────────────────────┐
              │                                     │
              ▼                                     │
        ┌──────────┐        select product         │
        │          │        (valid & in stock)     │
        │   IDLE   │──────────────────────────────▶│
        │          │                               │
        └────┬─────┘                               │
             │                                     │
             │ insert coin                          │
             │ (amount >= price)                   ▼
             │                              ┌──────────────┐
             │                              │   PRODUCT    │
             │                              │   SELECTED   │
             │                              │              │
             │                              └──────┬───────┘
             │                                     │
             │                                     │ dispense
             │                                     │ product
             │                                     │
             │                              ┌──────▼───────┐
             │                              │              │
             │                              │  DISPENSING  │
             │                              │              │
             │                              └──────┬───────┘
             │                                     │
             │                                     │ product
             │                                     │ taken
             │                                     │
             │                              ┌──────▼───────┐
             │                              │              │
             │                              │    IDLE      │
             │                              │              │
             │                              └──────────────┘
             │
             │ cancel/refund
             ▼
        ┌──────────┐
        │          │
        │ REFUND   │
        │          │
        └────┬─────┘
             │
             │ refund
             │ complete
             ▼
        ┌──────────┐
        │          │
        │   IDLE   │
        │          │
        └──────────┘

States:                    Transitions:
• Idle                     • insert coin
• Coin Inserted            • select product
• Product Selected         • dispense
• Dispensing               • product taken
• Out of Stock             • cancel / refund
• Refund                   • refill`}
            </pre>
          </div>
        </QuestionCard>

        {/* Question 65: Aggregation vs Composition */}
        <QuestionCard
          number={65}
          title="Aggregation vs Composition in UML"
          question="What is the difference between aggregation and composition? Show in UML."
          answer="Aggregation (hollow diamond) represents a 'has-a' relationship where parts can exist independently of the whole (weak ownership). Composition (solid diamond) represents a stronger 'has-a' where parts cannot exist without the whole (strong ownership, lifecycle dependency). Example: Car has Engine (composition - engine dies with car), Department has Professor (aggregation - professor can exist without department)."
          marks={5}
          icon={<Link className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg">
            <h4 className="font-medium text-foreground mb-2">Aggregation (Hollow Diamond)</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded">
{`┌────────────────┐          ┌────────────────┐
│   Department   │◇─────────│   Professor    │
├────────────────┤          ├────────────────┤
│ - name: String │          │ - name: String │
│ - deptId: int  │          │ - profId: int  │
├────────────────┤          │ - specialization│
│ + addProfessor()│          ├────────────────┤
│ + removeProfessor()│       │ + teach()      │
└────────────────┘          │ + research()   │
                            └────────────────┘

• Weak "has-a" relationship
• Parts can exist independently
• Professor can work in multiple departments
• Professor survives department deletion`}
            </pre>
            <h4 className="font-medium text-foreground mb-2 mt-4">Composition (Solid Diamond)</h4>
            <pre className="text-xs font-mono bg-background p-3 rounded">
{`┌────────────────┐          ┌────────────────┐
│      Car       │●─────────│     Engine     │
├────────────────┤          ├────────────────┤
│ - model: String│          │ - horsepower:  │
│ - year: int    │          │   int          │
│ - color: String│          │ - cylinders:   │
├────────────────┤          │   int          │
│ + start()      │          ├────────────────┤
│ + stop()       │          │ + start()      │
│ + drive()      │          │ + stop()       │
└────────────────┘          └────────────────┘

• Strong "has-a" relationship
• Parts cannot exist independently
• Engine is created with Car, destroyed with Car
• Engine cannot be shared between Cars`}
            </pre>
            <div className="mt-3 text-sm">
              <DifferenceTable
                title="Aggregation vs Composition Comparison"
                headers={["Feature", "Aggregation", "Composition"]}
                rows={[
                  ["UML Symbol", "Hollow diamond ◇", "Solid diamond ●"],
                  ["Relationship Strength", "Weak", "Strong"],
                  ["Lifecycle", "Parts can outlive whole", "Parts die with whole"],
                  ["Ownership", "Shared ownership", "Exclusive ownership"],
                  ["Example", "Department - Professor", "Car - Engine"],
                  ["Multiplicity", "Usually many-to-many", "Usually one-to-one"],
                ]}
              />
            </div>
          </div>
        </QuestionCard>

        {/* Question 66: Generalization vs Specialization */}
        <QuestionCard
          number={66}
          title="Generalization vs Specialization"
          question="What is the difference between generalization and specialization?"
          answer="Generalization is the process of extracting common features from multiple classes into a superclass (bottom-up). Specialization is creating subclasses from a superclass by adding specific features (top-down). Generalization increases abstraction; specialization increases detail. In UML, both are shown with hollow triangle arrows pointing to the superclass."
          marks={5}
          icon={<GitBranch className="h-3 w-3" />}
        >
          <div className="p-4 bg-muted/30 rounded-lg">
            <pre className="text-xs font-mono bg-background p-3 rounded overflow-x-auto">
{`                              ┌─────────────────────────────────────┐
                              │        Generalization (Bottom-Up)    │
                              │   Extracting common features upward  │
                              └─────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │         Vehicle         │  ← Superclass (Generalization)
                    │  (common features)      │
                    ├─────────────────────────┤
                    │ - brand: String         │
                    │ - model: String         │
                    │ + start()               │
                    │ + stop()                │
                    └───────────┬─────────────┘
                                ▲
                                │ (Generalization)
                    ┌───────────┴───────────┐
                    │                       │
          ┌─────────▼─────────┐   ┌─────────▼─────────┐
          │       Car         │   │      Motorcycle   │
          │ (specific to Car) │   │ (specific to Bike)│
          ├───────────────────┤   ├───────────────────┤
          │ - doorCount: int  │   │ - hasSidecar: bool│
          │ - transmission:   │   │ - handlebarType:  │
          │   String          │   │   String          │
          ├───────────────────┤   ├───────────────────┤
          │ + openTrunk()     │   │ + wheelie()       │
          └───────────────────┘   └───────────────────┘

                              ┌─────────────────────────────────────┐
                              │        Specialization (Top-Down)    │
                              │   Adding specific features downward │
                              └─────────────────────────────────────┘

Comparison:
┌─────────────────┬─────────────────────┬─────────────────────┐
│    Feature      │   Generalization    │   Specialization    │
├─────────────────┼─────────────────────┼─────────────────────┘
│ Direction       │ Bottom-up           │ Top-down            │
│ Process         │ Extract common      │ Add specific        │
│                 │ features upward     │ features downward   │
│ Abstraction     │ Increases           │ Decreases           │
│ Detail          │ Decreases           │ Increases           │
│ Reusability     │ Improves            │ Improves clarity    │
│ Relationship    │ Parent ← Child      │ Parent → Child      │
│ UML Arrow       │ Hollow triangle     │ Hollow triangle     │
│                 │ pointing up         │ pointing up         │
└─────────────────┴─────────────────────┴─────────────────────┘

Example - Animal Hierarchy:
┌─────────────────────────────────────────────────────────────────┐
│                        Animal                                    │
│                   (Generalization)                               │
│              ▲                  ▲                               │
│              │                  │                               │
│         ┌────┴────┐        ┌────┴────┐                          │
│         │ Mammal  │        │  Bird   │                          │
│         └────┬────┘        └────┬────┘                          │
│              │                  │                               │
│         ┌────┴────┐        ┌────┴────┐                          │
│         │  Dog    │        │ Sparrow │                          │
│         │  Cat    │        │ Penguin │                          │
│         └─────────┘        └─────────┘                          │
│                   (Specialization)                               │
└─────────────────────────────────────────────────────────────────┘`}
            </pre>
          </div>
        </QuestionCard>

        {/* Summary */}
        <div className="mt-10 p-6 bg-primary/5 rounded-lg border border-primary/20">
          <div className="flex items-center gap-3 mb-4">
            <Target className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-bold text-foreground">Quick Revision Summary</h3>
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q60-61:</span>
              <span className="text-muted-foreground ml-1">Class diagrams show structure, attributes, methods, relationships</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q62:</span>
              <span className="text-muted-foreground ml-1">Use case diagrams show actors and system functionality</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q63:</span>
              <span className="text-muted-foreground ml-1">Sequence diagrams show object interactions over time</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q64:</span>
              <span className="text-muted-foreground ml-1">State diagrams show object states and transitions</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q65:</span>
              <span className="text-muted-foreground ml-1">Aggregation (◇) weak; Composition (●) strong ownership</span>
            </div>
            <div className="p-2 bg-background rounded-lg text-sm">
              <span className="font-bold text-primary">Q66:</span>
              <span className="text-muted-foreground ml-1">Generalization (bottom-up); Specialization (top-down)</span>
            </div>
          </div>
        </div>
      </section>

      <Quiz questions={quizQuestions} title="UML & OOAD Quiz" />
    </TopicContent>
  );
}

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