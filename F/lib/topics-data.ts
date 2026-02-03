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
    subtopics: [
      { id: "variables", title: "Variables & Data Types", slug: "variables" },
      { id: "control-flow", title: "Control Flow", slug: "control-flow" },
      { id: "functions", title: "Functions", slug: "functions" },
      { id: "recursion", title: "Recursion", slug: "recursion" },
      { id: "pointers", title: "Pointers & References", slug: "pointers" },
    ],
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
