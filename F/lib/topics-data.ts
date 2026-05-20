export interface Topic {
  id: string;
  title: string;
  description: string;
  icon: string;
  subtopics: Subtopic[];
}

export interface Subtopic {
  id: string;
  title: string;
  slug: string;
}

export const topics: Topic[] = [
  {
    id: "data-structures",
    title: "Data Structures & Algorithms",
    description: "Learn fundamental data structures and algorithms with visualizations",
    icon: "layers",
    subtopics: [
      { id: "arrays", title: "Arrays", slug: "arrays" },
      { id: "linked-lists", title: "Linked Lists", slug: "linked-lists" },
      { id: "stacks", title: "Stacks", slug: "stacks" },
      { id: "queues", title: "Queues", slug: "queues" },
      { id: "trees", title: "Trees", slug: "trees" },
      { id: "graphs", title: "Graphs", slug: "graphs" },
      { id: "bfsandbfs", title: "BFS and DFS", slug: "bfsandbfs" },
      { id: "cycles", title: "Cycles", slug: "cycles" },
      { id: "shortestpath", title: "Shortest Path", slug: "shortestpath" },
      { id: "spanningtree", title: "Minimum Spanning Tree", slug: "spanningtree" },
      { id: "topologicalsorting", title: "Topological Sorting", slug: "topologicalsorting" },
      
      { id: "hash-tables", title: "Hash Tables", slug: "hash-tables" },
      { id: "sorting", title: "Sorting Algorithms", slug: "sorting" },
      { id: "searching", title: "Searching Algorithms", slug: "searching" },
    ],
  },
  {
    id: "operating-systems",
    title: "Operating Systems",
    description: "Understanding OS concepts from processes to memory management",
    icon: "cpu",
    subtopics: [
      { id: "process-thread", title: "Process vs Thread", slug: "process-thread" },
      { id: "cpu-scheduling", title: "CPU Scheduling", slug: "cpu-scheduling" },
      { id: "deadlocks", title: "Deadlocks", slug: "deadlocks" },
      { id: "memory-management", title: "Memory Management", slug: "memory-management" },
      { id: "virtual-memory", title: "Virtual Memory", slug: "virtual-memory" },
      { id: "file-systems", title: "File Systems", slug: "file-systems" },
    ],
  },
  {
    id: "programming-fundamentals",
    title: "Programming Fundamentals",
    description: "Core programming concepts every developer should know",
    icon: "code",
"subtopics": [
    { "id": "hello-world", "title": "Hello, World! & Environment Setup", "slug": "hello-world" },
    { "id": "variables", "title": "Variables & Data Types (int, float, str, bool)", "slug": "variables" },
    { "id": "basic-io", "title": "Basic I/O: print(), input(), f-strings", "slug": "basic-io" },
    { "id": "control-flow", "title": "Control Flow: if, elif, else, match/case", "slug": "control-flow" },
    { "id": "loops", "title": "Loops: for, while, break, continue", "slug": "loops" },
    
    { "id": "match-pattern", "title": "Match-pattern", "slug": "match-pattern" },
    { "id": "collections", "title": "Collections: list, tuple, dict, set", "slug": "collections" },

    { "id": "string", "title": "String ", "slug": "string" },
    { "id": "tuple", "title": "Tuple", "slug": "tuple" },
    { "id": "list", "title": "Lists ", "slug": "list" },
    { "id": "dict", "title": "Dictionaries", "slug": "dict" },
    { "id": "set", "title": " set", "slug": "set" },
    { "id": "functions", "title": "Functions: def, return, parameters, scope", "slug": "functions" },
    { "id": "lambda", "title": "Lambda Functions & map/filter/reduce", "slug": "lambda" },
    { "id": "error-handling", "title": "Exception Handling: try, except, finally", "slug": "error-handling" },
    { "id": "file-handling", "title": "File I/O: open, read, write, with context", "slug": "file-handling" },
    { "id": "modules", "title": "Modules & Packages: import, pip, __name__", "slug": "modules" },
    { "id": "recursion", "title": "Recursion in Python", "slug": "recursion" },
    { "id": "oop", "title": "Object-Oriented Programming: classes, inheritance, dunder methods", "slug": "oop" },
    { "id": "iterators-generators", "title": "Iterators & Generators: yield, next()", "slug": "iterators-generators" },
    { "id": "decorators", "title": "Decorators", "slug": "decorators" },
    { "id": "context-managers", "title": "Context Managers: with statement, __enter__/__exit__", "slug": "context-managers" },
    { "id": "concurrency", "title": "Concurrency: threading, multiprocessing", "slug": "concurrency" },
    { "id": "asyncio", "title": "Asynchronous Programming: async, await, asyncio", "slug": "asyncio" },
    { "id": "metaclasses", "title": "Metaclasses", "slug": "metaclasses" },
    { "id": "descriptors", "title": "Descriptors & property()", "slug": "descriptors" },
    { "id": "type-hints", "title": "Type Hints: typing module, mypy", "slug": "type-hints" },
    { "id": "testing", "title": "Testing: pytest, unittest, mocking", "slug": "testing" },
    { "id": "profiling", "title": "Profiling & Optimization: cProfile, line_profiler", "slug": "profiling" },
    { "id": "packaging", "title": "Packaging: pyproject.toml, setuptools, twine", "slug": "packaging" },
    { "id": "c-extensions", "title": "C Extensions: ctypes, Cython, CFFI", "slug": "c-extensions" }
  ]
  },
  {
    id: "oop",
    title: "Object-Oriented Programming",
    description: "Master OOP principles with practical examples",
    icon: "box",
    subtopics: [
      { id: "classes", title: "Classes & Objects", slug: "classes" },
      { id: "inheritance", title: "Inheritance", slug: "inheritance" },
      { id: "polymorphism", title: "Polymorphism", slug: "polymorphism" },
      { id: "encapsulation", title: "Encapsulation", slug: "encapsulation" },
      { id: "abstraction", title: "Abstraction", slug: "abstraction" },
    ],
  },
  {
    id: "dbms",
    title: "Database Management Systems",
    description: "Learn database design, SQL, and DBMS concepts",
    icon: "database",
    subtopics: [
      { id: "sql-basics", title: "SQL Basics", slug: "sql-basics" },
      { id: "normalization", title: "Normalization", slug: "normalization" },
      { id: "transactions", title: "Transactions", slug: "transactions" },
      { id: "indexing", title: "Indexing", slug: "indexing" },
      { id: "nosql", title: "NoSQL Databases", slug: "nosql" },
    ],
  },
  {
    id: "networks",
    title: "Computer Networks",
    description: "Understand networking from protocols to architecture",
    icon: "globe",
    subtopics: [
      { id: "osi-model", title: "OSI Model", slug: "osi-model" },
      { id: "tcp-ip", title: "TCP/IP", slug: "tcp-ip" },
      { id: "http", title: "HTTP & HTTPS", slug: "http" },
      { id: "dns", title: "DNS", slug: "dns" },
      { id: "routing", title: "Routing", slug: "routing" },
    ],
  },
  {
    id: "web-development",
    title: "Web Development",
    description: "Modern web development with HTML, CSS, and JavaScript",
    icon: "layout",
    subtopics: [
      { id: "html", title: "HTML Fundamentals", slug: "html" },
      { id: "css", title: "CSS Styling", slug: "css" },
      { id: "javascript", title: "JavaScript", slug: "javascript" },
      { id: "react", title: "React Basics", slug: "react" },
      { id: "apis", title: "REST APIs", slug: "apis" },
    ],
  },
  {
    id: "software-engineering",
    title: "Software Engineering",
    description: "Software development practices and methodologies",
    icon: "settings",
    subtopics: [
      { id: "sdlc", title: "SDLC Models", slug: "sdlc" },
      { id: "design-patterns", title: "Design Patterns", slug: "design-patterns" },
      { id: "testing", title: "Software Testing", slug: "testing" },
      { id: "agile", title: "Agile & Scrum", slug: "agile" },
    ],
  },
  {
    id: "cyber-security",
    title: "Cyber Security",
    description: "Security concepts and best practices",
    icon: "shield",
    subtopics: [
      { id: "encryption", title: "Encryption", slug: "encryption" },
      { id: "authentication", title: "Authentication", slug: "authentication" },
      { id: "vulnerabilities", title: "Common Vulnerabilities", slug: "vulnerabilities" },
      { id: "network-security", title: "Network Security", slug: "network-security" },
    ],
  },
  {
    id: "machine-learning",
    title: "Machine Learning Basics",
    description: "Introduction to ML concepts and algorithms",
    icon: "brain",
    subtopics: [
      { id: "ml-intro", title: "Introduction to ML", slug: "ml-intro" },
      { id: "supervised", title: "Supervised Learning", slug: "supervised" },
      { id: "unsupervised", title: "Unsupervised Learning", slug: "unsupervised" },
      { id: "neural-networks", title: "Neural Networks", slug: "neural-networks" },
    ],
  },
  {
    id: "deep-learning",
    title: "Deep Learning",
    description: "Advanced neural networks and deep learning architectures",
    icon: "sparkles",
    subtopics: [
      { id: "dl-intro", title: "Introduction to Deep Learning", slug: "dl-intro" },
      { id: "ann", title: "Artificial Neural Networks (ANN)", slug: "ann" },
      { id: "cnn", title: "Convolutional Neural Networks (CNN)", slug: "cnn" },
      { id: "rnn", title: "Recurrent Neural Networks (RNN)", slug: "rnn" },
      { id: "lstm", title: "LSTM & GRU Networks", slug: "lstm" },
      { id: "transformers", title: "Transformers & Attention Mechanism", slug: "transformers" },
      { id: "activation-functions", title: "Activation Functions", slug: "activation-functions" },
      { id: "optimizers", title: "Optimizers (Adam, SGD, RMSprop)", slug: "optimizers" },
      { id: "backpropagation", title: "Backpropagation", slug: "backpropagation" },
      { id: "regularization", title: "Regularization (Dropout, BatchNorm)", slug: "regularization" },
      { id: "transfer-learning", title: "Transfer Learning", slug: "transfer-learning" },
      { id: "gan", title: "Generative Adversarial Networks (GANs)", slug: "gan" },
      { id: "autoencoders", title: "Autoencoders", slug: "autoencoders" },
      { id: "reinforcement", title: "Deep Reinforcement Learning", slug: "reinforcement" },
      { id: "nlp", title: "NLP with Deep Learning", slug: "nlp" },
      { id: "cv", title: "Computer Vision", slug: "cv" },
      { id: "frameworks", title: "Frameworks (TensorFlow, PyTorch, Keras)", slug: "frameworks" },
      { id: "model-deployment", title: "Model Deployment", slug: "model-deployment" },

    
  
],
  },
];

export function getTopicBySlug(slug: string): Topic | undefined {
  return topics.find((t) => t.id === slug);
}

export function getSubtopicBySlug(
  topicSlug: string,
  subtopicSlug: string
): { topic: Topic; subtopic: Subtopic } | undefined {
  const topic = getTopicBySlug(topicSlug);
  if (!topic) return undefined;
  const subtopic = topic.subtopics.find((s) => s.slug === subtopicSlug);
  if (!subtopic) return undefined;
  return { topic, subtopic };
}
