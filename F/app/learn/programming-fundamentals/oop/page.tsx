import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function OopPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "oop");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Object-Oriented Programming
          </h1>
          <p className="text-muted-foreground text-lg">
            Everything in Python is an object. Integers, strings, functions, and even classes themselves are objects. Object-oriented programming (OOP) takes advantage of this by building programs around objects that combine data and behavior.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              OOP in Python is more hands-on than theoretical. The syntax is minimal, but the details matter: class attributes behave differently from instance attributes, <code className="bg-muted px-1.5 py-0.5 rounded">__repr__</code> is not the same as <code className="bg-muted px-1.5 py-0.5 rounded">__str__</code>, and defining <code className="bg-muted px-1.5 py-0.5 rounded">__eq__</code> silently removes hashability.
            </p>
          </div>
        </div>

        {/* What is OOP */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What is OOP?
          </h2>
          <p className="text-muted-foreground mb-4">
            Object-oriented programming organizes code around <strong className="text-foreground">objects</strong> instead of procedures. An object bundles data (<strong className="text-foreground">attributes</strong>) and behavior (<strong className="text-foreground">methods</strong>) into a single unit.
          </p>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Concept</th>
                  <th className="text-left p-3 font-semibold text-foreground">Meaning</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">Encapsulation</td><td className="p-3">Bundling data and behavior together and controlling access</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">Inheritance</td><td className="p-3">One class reusing or extending another</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">Polymorphism</td><td className="p-3">Different objects responding to the same interface</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">Abstraction</td><td className="p-3">Hiding implementation details behind a clean interface</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Classes and Objects */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Classes and Objects
          </h2>
          <p className="text-muted-foreground mb-4">
            A <strong className="text-foreground">class</strong> is a blueprint. An <strong className="text-foreground">object</strong> (also called an <strong className="text-foreground">instance</strong>) is a concrete thing built from that blueprint.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Dog:
    pass

rex = Dog()                # creates an instance of Dog
print(type(rex))           # <class '__main__.Dog'>
print(isinstance(rex, Dog))  # True`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <code className="bg-muted px-1.5 py-0.5 rounded">__new__</code> vs <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code>: <code className="bg-muted px-1.5 py-0.5 rounded">__new__</code> allocates memory and returns the new instance. <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code> initializes it. In everyday code, only <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code> is ever defined.
              </p>
            </div>
          </div>
        </section>

        {/* __init__ and self */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            __init__ and self
          </h2>
          <p className="text-muted-foreground mb-4">
            <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code> is the <strong className="text-foreground">initializer</strong>. It runs automatically after a new instance is created. <code className="bg-muted px-1.5 py-0.5 rounded">self</code> is the first parameter of every instance method — Python automatically passes the instance as <code className="bg-muted px-1.5 py-0.5 rounded">self</code>.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Dog:
    def __init__(self, name, age):
        self.name = name   # instance attribute
        self.age = age

rex = Dog("Rex", 4)
print(rex.name)   # Rex
print(rex.age)    # 4`}
            </pre>
          </div>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code> must return <code className="bg-muted px-1.5 py-0.5 rounded">None</code>. Returning any other value raises <code className="bg-muted px-1.5 py-0.5 rounded">TypeError</code>.
              </p>
            </div>
          </div>
        </section>

        {/* Instance vs Class Attributes */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Instance Attributes vs Class Attributes
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Type</th>
                  <th className="text-left p-3 font-semibold text-foreground">Defined</th>
                  <th className="text-left p-3 font-semibold text-foreground">Shared?</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">Instance attribute</td><td className="p-3">Inside methods on <code className="bg-muted px-1.5 py-0.5 rounded">self</code></td><td className="p-3">No — each instance has its own</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold text-foreground">Class attribute</td><td className="p-3">Directly in the class body</td><td className="p-3">Yes — all instances share</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Counter:
    total = 0        # class attribute; shared

    def __init__(self, value):
        self.value = value   # instance attribute; unique
        Counter.total += 1

a = Counter(2)
b = Counter(4)
print(Counter.total)   # 2
print(a.value)         # 2
print(b.value)         # 4

# Shadowing - instance assignment creates new attribute
class Config:
    limit = 10

c = Config()
c.limit = 20          # creates instance attribute
print(Config.limit)   # 10
print(c.limit)        # 20`}
            </pre>
          </div>
        </section>

        {/* Instance Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Instance Methods
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def scale(self, factor):
        self.width *= factor
        self.height *= factor

r = Rectangle(4, 6)
print(r.area())   # 24
r.scale(2)
print(r.area())   # 96`}
            </pre>
          </div>
        </section>

        {/* Class Methods and Static Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Class Methods and Static Methods
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Decorator</th>
                  <th className="text-left p-3 font-semibold text-foreground">First parameter</th>
                  <th className="text-left p-3 font-semibold text-foreground">Typical use</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">(none)</td><td className="p-3"><code className="bg-muted px-1.5 py-0.5 rounded">self</code></td><td className="p-3">Most methods</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">@classmethod</td><td className="p-3"><code className="bg-muted px-1.5 py-0.5 rounded">cls</code></td><td className="p-3">Alternative constructors, class-level state</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">@staticmethod</td><td className="p-3">(none)</td><td className="p-3">Utility functions grouped with the class</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Temperature:
    unit = "Celsius"

    def __init__(self, value):
        self.value = value

    @classmethod
    def from_fahrenheit(cls, f):
        celsius = (f - 32) * 5 / 9
        return cls(celsius)

    @staticmethod
    def is_freezing(value):
        return value <= 0

t = Temperature.from_fahrenheit(68)   # 68°F = 20°C
print(t.value)                        # 20.0
print(Temperature.is_freezing(0))     # True`}
            </pre>
          </div>
        </section>

        {/* __str__ and __repr__ */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            __str__ and __repr__
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Called by</th>
                  <th className="text-left p-3 font-semibold text-foreground">Purpose</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">__repr__</td><td className="p-3"><code className="bg-muted px-1.5 py-0.5 rounded">repr()</code>, REPL, containers</td><td className="p-3">Unambiguous developer view</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">__str__</td><td className="p-3"><code className="bg-muted px-1.5 py-0.5 rounded">str()</code>, <code className="bg-muted px-1.5 py-0.5 rounded">print()</code>, f-strings</td><td className="p-3">Readable human-facing string</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x!r}, {self.y!r})"

    def __str__(self):
        return f"({self.x}, {self.y})"

p = Point(2, 4)
print(repr(p))   # Point(2, 4)
print(str(p))    # (2, 4)
print(p)         # (2, 4) -- print() uses __str__
print([p])       # [Point(2, 4)] -- containers use __repr__`}
            </pre>
          </div>
        </section>

        {/* Dunder Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Dunder (Magic) Methods
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Box:
    def __init__(self, volume):
        self.volume = volume

    def __eq__(self, other):
        return self.volume == other.volume

    def __lt__(self, other):
        return self.volume < other.volume

a = Box(8)
b = Box(16)
print(a == a)          # True
print(a < b)           # True
print(sorted([b, a]))  # sorts using __lt__`}
            </pre>
          </div>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <strong>Important:</strong> Defining <code className="bg-muted px-1.5 py-0.5 rounded">__eq__</code> without <code className="bg-muted px-1.5 py-0.5 rounded">__hash__</code> makes instances unhashable. They cannot be used as dict keys or in sets.
              </p>
            </div>
          </div>
        </section>

        {/* @property */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            @property
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Circle:
    def __init__(self, radius):
        self._radius = 0
        self.radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def diameter(self):
        return self._radius * 2

c = Circle(4)
print(c.radius)    # 4
print(c.diameter)  # 8
c.radius = 6
print(c.diameter)  # 12
# c.diameter = 20 # AttributeError: can't set attribute`}
            </pre>
          </div>
        </section>

        {/* __slots__ */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            __slots__
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(2, 4)
print(p.x)    # 2
# p.z = 6    # AttributeError -- only x and y allowed
# p.__dict__ # AttributeError -- no __dict__ when __slots__ defined`}
            </pre>
          </div>
        </section>

        {/* Inheritance */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Inheritance
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"

class Cat(Animal):
    def speak(self):
        return f"{self.name} meows"

d = Dog("Rex")
c = Cat("Whiskers")
print(d.speak())   # Rex barks
print(c.speak())   # Whiskers meows
print(d.name)      # Rex -- inherited from Animal

# Using super()
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)   # calls Animal.__init__
        self.breed = breed`}
            </pre>
          </div>
        </section>

        {/* Multiple Inheritance and MRO */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Multiple Inheritance and MRO
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class Flyable:
    def fly(self):
        return "flying"

class Swimmable:
    def swim(self):
        return "swimming"

class Duck(Flyable, Swimmable):
    pass

d = Duck()
print(d.fly())    # flying
print(d.swim())   # swimming

# Method Resolution Order
class A:
    def hello(self):
        return "A"

class B(A):
    def hello(self):
        return "B"

class C(A):
    def hello(self):
        return "C"

class D(B, C):
    pass

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
print(D().hello())   # B -- first class in MRO that defines hello`}
            </pre>
          </div>
        </section>

        {/* Polymorphism */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Polymorphism
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Method overriding
class Shape:
    def area(self):
        return 0

class Square(Shape):
    def __init__(self, side):
        self.side = side
    def area(self):
        return self.side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        import math
        return math.pi * self.radius ** 2

shapes = [Square(4), Circle(2), Square(6)]
for s in shapes:
    print(s.area())   # each calls its own version

# Duck typing
class FileWriter:
    def write(self, data):
        print(f"writing to file: {data}")

class NetworkSender:
    def write(self, data):
        print(f"sending over network: {data}")

def send(output, data):
    output.write(data)   # any object with .write() works

send(FileWriter(), "hello")
send(NetworkSender(), "hello")`}
            </pre>
          </div>
        </section>

        {/* Abstract Base Classes */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Abstract Base Classes
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def describe(self):
        return f"area={self.area():.2f}, perimeter={self.perimeter():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

r = Rectangle(4, 6)
print(r.area())        # 24
print(r.perimeter())   # 20
print(r.describe())    # area=24.00, perimeter=20.00`}
            </pre>
          </div>
        </section>

        {/* Protocol */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Protocol (Structural Subtyping)
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from typing import Protocol, runtime_checkable

@runtime_checkable
class Writable(Protocol):
    def write(self, data: str) -> None: ...

class FileWriter:
    def write(self, data: str) -> None:
        print(f"file: {data}")

def send(output: Writable, data: str) -> None:
    output.write(data)

send(FileWriter(), "hello")   # file: hello
print(isinstance(FileWriter(), Writable))   # True`}
            </pre>
          </div>
        </section>

        {/* Tricky Points */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Name mangling is not true privacy</p>
                  <p className="text-muted-foreground text-sm"><code className="bg-muted px-1.5 py-0.5 rounded">__attr</code> becomes <code className="bg-muted px-1.5 py-0.5 rounded">_ClassName__attr</code>. The mangled name is always accessible.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Mutable class attributes are shared</p>
                  <p className="text-muted-foreground text-sm">A list or dict defined as a class attribute is shared by all instances. Fix by defining mutable attributes in <code className="bg-muted px-1.5 py-0.5 rounded">__init__</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Defining __eq__ removes hashability</p>
                  <p className="text-muted-foreground text-sm">Python automatically sets <code className="bg-muted px-1.5 py-0.5 rounded">__hash__ = None</code>. Define <code className="bg-muted px-1.5 py-0.5 rounded">__hash__</code> explicitly if hashability is needed.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Interview Questions
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between a class and an object?</p>
                  <p className="text-muted-foreground">A class is a blueprint defining what attributes and methods its instances will have. An object is a concrete instance created from that blueprint.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between <code className="bg-muted px-1.5 py-0.5 rounded">__str__</code> and <code className="bg-muted px-1.5 py-0.5 rounded">__repr__</code>?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">__repr__</code> is for developers (unambiguous, ideally <code className="bg-muted px-1.5 py-0.5 rounded">eval(repr(obj)) == obj</code>). <code className="bg-muted px-1.5 py-0.5 rounded">__str__</code> is for users (readable). <code className="bg-muted px-1.5 py-0.5 rounded">print()</code> uses <code className="bg-muted px-1.5 py-0.5 rounded">__str__</code>; containers use <code className="bg-muted px-1.5 py-0.5 rounded">__repr__</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the Method Resolution Order (MRO)?</p>
                  <p className="text-muted-foreground">The MRO is the order Python uses to look up attributes in a class hierarchy. Use <code className="bg-muted px-1.5 py-0.5 rounded">ClassName.__mro__</code> to inspect it. Python uses the C3 linearization algorithm.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

       
      </div>
    </TopicContent>
  );
}