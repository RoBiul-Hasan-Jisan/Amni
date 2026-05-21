import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function AdvancedPythonPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "advanced-python");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Advanced Python
          </h1>
          <p className="text-muted-foreground text-lg">
            Master Python's advanced features: iterators, generators, decorators, context managers, concurrency, asyncio, metaclasses, descriptors, type hints, testing, profiling, and packaging.
          </p>
        </div>

        {/* ==================== ITERATORS & GENERATORS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Iterators & Generators
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Iterators</h3>
          <p className="text-muted-foreground mb-2">
            An iterator is an object that implements <code className="bg-muted px-1.5 py-0.5 rounded">__iter__()</code> and <code className="bg-muted px-1.5 py-0.5 rounded">__next__()</code>.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class CountUpTo:
    def __init__(self, max):
        self.max = max
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max:
            raise StopIteration
        result = self.current
        self.current += 1
        return result

for num in CountUpTo(5):
    print(num)  # 0, 1, 2, 3, 4`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Generators</h3>
          <p className="text-muted-foreground mb-2">
            Generators use <code className="bg-muted px-1.5 py-0.5 rounded">yield</code> to produce values lazily.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def count_up_to(max):
    n = 0
    while n < max:
        yield n
        n += 1

for num in count_up_to(5):
    print(num)  # 0, 1, 2, 3, 4

# Generator expression
squares = (x * x for x in range(10))
print(next(squares))  # 0
print(next(squares))  # 1
print(list(squares))  # [4, 9, 16, 25, 36, 49, 64, 81]

# yield from - delegate to sub-generator
def chain(*iterables):
    for it in iterables:
        yield from it

list(chain([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Generators are memory-efficient because they produce values on demand, not all at once.
              </p>
            </div>
          </div>
        </section>

        {/* ==================== DECORATORS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Decorators
          </h2>
          <p className="text-muted-foreground mb-2">
            Decorators wrap functions to add behavior without modifying their code.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import functools
import time

# Basic decorator
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # slow_function took 1.0002s

# Decorator with arguments
def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}")

greet("Alice")  # Prints 3 times

# Class decorator
def add_method(cls):
    def new_method(self):
        return "Added by decorator"
    cls.new_method = new_method
    return cls

@add_method
class MyClass:
    pass

obj = MyClass()
print(obj.new_method())  # Added by decorator`}
            </pre>
          </div>
        </section>

        {/* ==================== CONTEXT MANAGERS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Context Managers
          </h2>
          <p className="text-muted-foreground mb-2">
            Context managers manage resources with <code className="bg-muted px-1.5 py-0.5 rounded">with</code> statements.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Class-based context manager
class ManagedFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False  # Propagate exceptions

with ManagedFile("test.txt", "w") as f:
    f.write("Hello, World!")

# Generator-based context manager
from contextlib import contextmanager

@contextmanager
def managed_file(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()

with managed_file("test.txt", "r") as f:
    content = f.read()
    print(content)

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())`}
            </pre>
          </div>
        </section>

        {/* ==================== MODULES & PACKAGES ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Modules & Packages
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Importing modules
import math
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np  # Third-party

# Creating a module (mymodule.py)
# def greet(name):
#     return f"Hello, {name}"

# Using the module
# import mymodule
# print(mymodule.greet("Alice"))

# Package structure
# mypackage/
#     __init__.py
#     module1.py
#     module2.py
#     subpackage/
#         __init__.py
#         module3.py

# __name__ == "__main__" guard
if __name__ == "__main__":
    # Code runs only when script executed directly
    print("This script is run directly")

# Import from parent directory
import sys
sys.path.append("..")
from parent_module import something`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                The <code className="bg-muted px-1.5 py-0.5 rounded">__name__ == "__main__"</code> guard prevents code from running when the module is imported.
              </p>
            </div>
          </div>
        </section>

        {/* ==================== CONCURRENCY (Threading & Multiprocessing) ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Concurrency (Threading & Multiprocessing)
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Threading (I/O-bound tasks)</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import threading
import time

def worker(name, delay):
    print(f"Thread {name} starting")
    time.sleep(delay)
    print(f"Thread {name} finished")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i, 2))
    threads.append(t)
    t.start()

for t in threads:
    t.join()  # Wait for all threads to complete

# Thread synchronization
lock = threading.Lock()
counter = 0

def increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)  # 400000`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Multiprocessing (CPU-bound tasks)</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import multiprocessing
import time

def cpu_intensive(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    numbers = [10_000_000] * 4
    
    # Sequential
    start = time.time()
    results = [cpu_intensive(n) for n in numbers]
    print(f"Sequential: {time.time() - start:.2f}s")
    
    # Parallel
    start = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(cpu_intensive, numbers)
    print(f"Parallel: {time.time() - start:.2f}s")`}
            </pre>
          </div>
        </section>

        {/* ==================== ASYNCIO ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Asyncio (Asynchronous Programming)
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import asyncio

async def fetch_data(delay, name):
    print(f"Fetching {name}...")
    await asyncio.sleep(delay)
    print(f"{name} complete")
    return f"Data from {name}"

async def main():
    # Run tasks concurrently
    tasks = [
        fetch_data(2, "API 1"),
        fetch_data(1, "API 2"),
        fetch_data(3, "API 3"),
    ]
    results = await asyncio.gather(*tasks)
    print(results)

# Run async function
asyncio.run(main())

# Async context manager
async def read_file():
    async with aiofiles.open("data.txt", "r") as f:
        content = await f.read()
        return content

# Async generator
async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_range(5):
        print(num)

asyncio.run(main())`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Use asyncio for I/O-bound tasks with many concurrent connections. Use multiprocessing for CPU-bound tasks.
              </p>
            </div>
          </div>
        </section>

        {/* ==================== METACLASSES ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Metaclasses
          </h2>
          <p className="text-muted-foreground mb-2">
            Metaclasses are classes of classes. They control class creation and behavior.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Simple metaclass
class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        # Add attribute to all classes
        dct['created_by'] = 'Meta'
        return super().__new__(cls, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        print(f"Instantiating {cls.__name__}")
        return super().__call__(*args, **kwargs)

class MyClass(metaclass=Meta):
    pass

obj = MyClass()
print(MyClass.created_by)  # Meta

# Singleton metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connected = True

db1 = Database()
db2 = Database()
print(db1 is db2)  # True`}
            </pre>
          </div>
        </section>

        {/* ==================== DESCRIPTORS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Descriptors
          </h2>
          <p className="text-muted-foreground mb-2">
            Descriptors control attribute access through <code className="bg-muted px-1.5 py-0.5 rounded">__get__</code>, <code className="bg-muted px-1.5 py-0.5 rounded">__set__</code>, and <code className="bg-muted px-1.5 py-0.5 rounded">__delete__</code>.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class PositiveNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        obj.__dict__[self.name] = value

class Person:
    age = PositiveNumber()
    score = PositiveNumber()

    def __init__(self, age, score):
        self.age = age
        self.score = score

p = Person(25, 95)
print(p.age)    # 25
# p.age = -5    # ValueError: age must be positive

# Property vs Descriptor
# @property is implemented using descriptors!
class Property:
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset:
            self.fset(obj, value)`}
            </pre>
          </div>
        </section>

        {/* ==================== TYPE HINTS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Type Hints
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from typing import List, Dict, Optional, Union, Tuple, Callable, Any

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}"

# Collections
def process_items(items: List[int]) -> int:
    return sum(items)

# Optional (can be None)
def find_user(id: int) -> Optional[Dict[str, str]]:
    if id == 1:
        return {"name": "Alice"}
    return None

# Union (multiple types)
def stringify(value: Union[int, float, str]) -> str:
    return str(value)

# Type aliases
Vector = List[float]
Matrix = List[Vector]

def scale(vector: Vector, factor: float) -> Vector:
    return [v * factor for v in vector]

# Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Type variables
from typing import TypeVar
T = TypeVar('T')

def first(items: List[T]) -> T:
    return items[0]

# Protocol (structural typing)
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable) -> None:
    obj.draw()

# Runtime type checking with isinstance
def process(data: Any) -> None:
    if isinstance(data, list):
        print(f"List of {len(data)} items")
    elif isinstance(data, dict):
        print(f"Dict with keys {list(data.keys())}")`}
            </pre>
          </div>
        </section>

        {/* ==================== TESTING ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Testing (pytest & unittest)
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">unittest (Built-in)</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import unittest

def add(a, b):
    return a + b

class TestMath(unittest.TestCase):
    def setUp(self):
        # Runs before each test
        self.data = [1, 2, 3]

    def tearDown(self):
        # Runs after each test
        self.data.clear()

    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertNotEqual(add(2, 3), 6)

    def test_raises(self):
        with self.assertRaises(TypeError):
            add("2", 3)

if __name__ == "__main__":
    unittest.main()`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">pytest (Third-party, more concise)</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# test_math.py
import pytest

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(2, 3) != 6

def test_raises():
    with pytest.raises(TypeError):
        add("2", 3)

# Fixtures
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

# Parametrized tests
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add_param(a, b, expected):
    assert add(a, b) == expected

# Mocking
from unittest.mock import Mock

def test_mock():
    mock = Mock()
    mock.method.return_value = 42
    assert mock.method() == 42

# Run: pytest test_math.py -v`}
            </pre>
          </div>
        </section>

        {/* ==================== PROFILING ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Profiling & Optimization
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import cProfile
import pstats
import time

def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

def fast_function():
    return sum(i ** 2 for i in range(1000000))

# Basic profiling
cProfile.run('slow_function()')

# Profile with output file
cProfile.run('slow_function()', 'profile.stats')

# Analyze profile results
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Line profiler (requires line_profiler)
# @profile
# def my_function():
#     ...

# Time profiling
start = time.perf_counter()
result = slow_function()
end = time.perf_counter()
print(f"Time: {end - start:.4f}s")

# Memory profiling (requires memory_profiler)
# from memory_profiler import profile
# @profile
# def memory_intensive():
#     return [i for i in range(1000000)]

# Optimization tips
# - Use local variables for speed
# - List comprehensions over loops
# - Use sets for membership testing
# - Cache function results with @lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`}
            </pre>
          </div>
        </section>

        {/* ==================== PACKAGING ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Packaging (pyproject.toml, setuptools)
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# pyproject.toml (modern approach)
"""
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "My awesome package"
authors = [{name = "Your Name", email = "you@example.com"}]
dependencies = [
    "requests>=2.25.0",
    "numpy>=1.19.0",
]

[project.optional-dependencies]
dev = ["pytest>=6.0", "black", "mypy"]

[project.scripts]
mycli = "mypackage.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
"""

# Package structure
# mypackage/
#     src/
#         mypackage/
#             __init__.py
#             core.py
#             cli.py
#     tests/
#         test_core.py
#     pyproject.toml
#     README.md
#     LICENSE

# Installing package in development mode
# pip install -e .

# Building distribution
# python -m build
# pip install dist/mypackage-0.1.0-py3-none-any.whl

# Uploading to PyPI
# twine upload dist/*

# __init__.py - package initialization
"""
__version__ = "0.1.0"
__author__ = "Your Name"

from .core import main_function

__all__ = ["main_function"]
"""`}
            </pre>
          </div>
        </section>

        {/* ==================== TRICKY POINTS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Generators can only be consumed once</p>
                  <p className="text-muted-foreground text-sm">After exhausting a generator, it cannot be reused. Create a new generator instance to iterate again.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Threading vs Multiprocessing</p>
                  <p className="text-muted-foreground text-sm">Threads are limited by GIL for CPU-bound tasks. Use multiprocessing for CPU-intensive work; use threading for I/O-bound work.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Type hints are not enforced at runtime</p>
                  <p className="text-muted-foreground text-sm">Type hints are for static type checkers like mypy. Python ignores them at runtime.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================== INTERVIEW QUESTIONS ==================== */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Interview Questions
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between a generator and an iterator?</p>
                  <p className="text-muted-foreground">All generators are iterators, but not all iterators are generators. Generators are created with <code className="bg-muted px-1.5 py-0.5 rounded">yield</code> and are a concise way to create iterators.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What problem do decorators solve?</p>
                  <p className="text-muted-foreground">Decorators enable the Open-Closed Principle: you can add behavior (logging, timing, caching) without modifying the original function's code.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is a metaclass and when would you use it?</p>
                  <p className="text-muted-foreground">A metaclass is a class that creates classes. Use it for framework-level concerns: automatic registration, ORM models, or enforcing coding standards across many classes.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the GIL and how does it affect concurrency?</p>
                  <p className="text-muted-foreground">The Global Interpreter Lock prevents multiple threads from executing Python bytecode simultaneously. For CPU-bound tasks, use multiprocessing. For I/O-bound tasks, threading or asyncio works fine.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

    
      </div>
    </TopicContent>
  );
}