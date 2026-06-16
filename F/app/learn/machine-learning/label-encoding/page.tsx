import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { 
  AlertCircle, 
  CheckCircle2, 
  Clock, 
  Lightbulb,
  Brain,
  Target,
  TrendingUp,
  BarChart3,
  Layers,
  GitBranch,
  Activity,
  Shield,
  Sparkles,
  Database,
  Cpu,
  LineChart,
  PieChart,
  GraduationCap,
  Network,
  ScatterChart,
  TreePine,
  Grid,
  Zap,
  Orbit,
  Cog,
  Filter,
  Sliders,
  Wand2,
  Eye,
  FileSearch,
  Table,
  ChartBar,
  ChartScatter,
  ChartLine,
  Palette,
  Square,
  Circle,
  Hammer,
  Wrench,
  Construction,
  Ruler,
  Gauge,
  ArrowUpRight,
  Sigma,
  Calculator,
  
  Minus,
  Plus,
  Equal,
  X,
  
  ArrowRight,
  Scale,
  Maximize,
  Minimize,
  Move,
  
  AlignCenter,
  AlignLeft,
  AlignRight,
  Binary,
  Hash,
  Type,
  List,
  Grid3x3,
  Tag
} from "lucide-react";

export default function LabelEncodingPage() {
  const result = getSubtopicBySlug("machine-learning", "label-encoding");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-basic",
      label: "Basic Label Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create sample data with categorical columns
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small', 'Large'],
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Price': [10, 20, 30, 15, 25, 35, 12, 22, 32]
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. BASIC LABEL ENCODING")
print("=" * 50)

# Create LabelEncoder instance
le = LabelEncoder()

# Apply to a single column
df['Color_Encoded'] = le.fit_transform(df['Color'])

print("Color mapping:")
for original, encoded in zip(df['Color'].unique(), le.transform(df['Color'].unique())):
    print(f"  {original} → {encoded}")

print("\\nData with encoded column:")
print(df[['Color', 'Color_Encoded']].head(10))

print("\\n" + "=" * 50)
print("3. ENCODING MULTIPLE COLUMNS")
print("=" * 50)

# Apply to multiple columns
df_encoded = df.copy()
for col in ['Color', 'Size', 'Product']:
    le = LabelEncoder()
    df_encoded[col + '_Encoded'] = le.fit_transform(df[col])
    
    # Show mapping for each column
    print(f"\\n{col} Mapping:")
    for original, encoded in zip(df[col].unique(), le.transform(df[col].unique())):
        print(f"  {original} → {encoded}")

print("\\n" + "=" * 50)
print("4. ENCODED DATA")
print("=" * 50)
print(df_encoded)`,
    },
    {
      language: "python-inverse",
      label: "Inverse Transformation",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create sample data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small', 'Large']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

# Encode the data
le = LabelEncoder()
df['Color_Encoded'] = le.fit_transform(df['Color'])
df['Size_Encoded'] = le.fit_transform(df['Size'])

print("\\n" + "=" * 50)
print("2. ENCODED DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("3. INVERSE TRANSFORMATION")
print("=" * 50)

# Decode back to original values
df['Color_Decoded'] = le.inverse_transform(df['Color_Encoded'])

# For Size, we need a new encoder since we used the same 'le' instance
le_size = LabelEncoder()
le_size.fit(df['Size'])
df['Size_Decoded'] = le_size.inverse_transform(df['Size_Encoded'])

print("Decoded data:")
print(df[['Color', 'Color_Encoded', 'Color_Decoded', 'Size', 'Size_Encoded', 'Size_Decoded']])

print("\\n" + "=" * 50)
print("4. VERIFICATION")
print("=" * 50)
print("Original Color matches Decoded Color:", (df['Color'] == df['Color_Decoded']).all())
print("Original Size matches Decoded Size:", (df['Size'] == df['Size_Decoded']).all())`,
    },
    {
      language: "python-ordinal",
      label: "Label Encoding for Ordinal Data",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create ordinal data (has natural order)
data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large', 'Medium', 'Extra Large'],
    'Experience': ['Junior', 'Mid', 'Senior', 'Lead', 'Mid', 'Senior', 'Junior', 'Lead']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL ORDINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. LABEL ENCODING ORDINAL DATA")
print("=" * 50)

print("Note: Label Encoding preserves the order if categories are in proper sequence")
print("However, it may not preserve the exact spacing between categories\\n")

# Encode ordinal data
le_education = LabelEncoder()
le_size = LabelEncoder()
le_experience = LabelEncoder()

df['Education_Encoded'] = le_education.fit_transform(df['Education'])
df['Size_Encoded'] = le_size.fit_transform(df['Size'])
df['Experience_Encoded'] = le_experience.fit_transform(df['Experience'])

print("Encoded data:")
print(df)

print("\\n" + "=" * 50)
print("3. MAPPING VERIFICATION")
print("=" * 50)

# Show the mapping for each column
education_mapping = dict(zip(le_education.classes_, le_education.transform(le_education.classes_)))
size_mapping = dict(zip(le_size.classes_, le_size.transform(le_size.classes_)))
experience_mapping = dict(zip(le_experience.classes_, le_experience.transform(le_experience.classes_)))

print("Education mapping (preserves order):")
for key, value in sorted(education_mapping.items(), key=lambda x: x[1]):
    print(f"  {key} → {value}")

print("\\nSize mapping (preserves order):")
for key, value in sorted(size_mapping.items(), key=lambda x: x[1]):
    print(f"  {key} → {value}")

print("\\nExperience mapping (preserves order):")
for key, value in sorted(experience_mapping.items(), key=lambda x: x[1]):
    print(f"  {key} → {value}")

print("\\n" + "=" * 50)
print("4. IMPORTANT NOTE")
print("=" * 50)
print("Label Encoding assigns integers based on alphabetical order by default!")
print("To preserve custom order, use OrdinalEncoder with specified categories.")`,
    },
    {
      language: "python-mapping",
      label: "Manual Label Encoding",
      code: `import pandas as pd
import numpy as np

# Create sample data
data = {
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'London', 'Paris', 'New York', 'Tokyo'],
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'IT', 'HR', 'Marketing', 'IT'],
    'Experience': ['Junior', 'Mid', 'Senior', 'Lead', 'Mid', 'Senior', 'Junior', 'Lead']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. MANUAL LABEL ENCODING (Using Dictionary)")
print("=" * 50)

# Create mapping dictionaries
city_mapping = {
    'New York': 0,
    'London': 1,
    'Paris': 2,
    'Tokyo': 3
}

department_mapping = {
    'Sales': 0,
    'HR': 1,
    'IT': 2,
    'Marketing': 3
}

experience_mapping = {
    'Junior': 0,
    'Mid': 1,
    'Senior': 2,
    'Lead': 3
}

print("City mapping:", city_mapping)
print("Department mapping:", department_mapping)
print("Experience mapping:", experience_mapping)

# Apply mapping
df['City_Encoded'] = df['City'].map(city_mapping)
df['Department_Encoded'] = df['Department'].map(department_mapping)
df['Experience_Encoded'] = df['Experience'].map(experience_mapping)

print("\\n" + "=" * 50)
print("3. ENCODED DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("4. REVERSE MAPPING")
print("=" * 50)

# Create reverse mapping
reverse_city = {v: k for k, v in city_mapping.items()}
reverse_department = {v: k for k, v in department_mapping.items()}
reverse_experience = {v: k for k, v in experience_mapping.items()}

print("Reverse city mapping:", reverse_city)

# Apply reverse mapping
df['City_Decoded'] = df['City_Encoded'].map(reverse_city)
df['Department_Decoded'] = df['Department_Encoded'].map(reverse_department)
df['Experience_Decoded'] = df['Experience_Encoded'].map(reverse_experience)

print("\\n" + "=" * 50)
print("5. DECODED DATA")
print("=" * 50)
print(df[['City', 'City_Encoded', 'City_Decoded', 'Department', 'Department_Encoded', 'Department_Decoded']])

print("\\n" + "=" * 50)
print("6. COMPARISON: SKLEARN VS MANUAL")
print("=" * 50)
print("Manual encoding gives you more control over the mapping values")
print("It's useful when you need specific integer assignments")`,
    },
    {
      language: "python-pitfalls",
      label: "Common Pitfalls & Solutions",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create sample data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Quality': ['Good', 'Bad', 'Excellent', 'Good', 'Excellent', 'Bad', 'Good', 'Bad', 'Excellent']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. PITFALL: FALSE ORDINAL RELATIONSHIP")
print("=" * 50)

# Encode with LabelEncoder
le = LabelEncoder()
df['Color_Encoded'] = le.fit_transform(df['Color'])

print("Color encoding (alphabetical order):")
for color, encoded in zip(df['Color'].unique(), le.transform(df['Color'].unique())):
    print(f"  {color} → {encoded}")

print("\\nIssue: LabelEncoder assigns numbers alphabetically (Blue→0, Green→1, Red→2)")
print("This creates a false ordinal relationship where Blue < Green < Red")
print("This can mislead models that interpret numerical order\\n")

print("\\n" + "=" * 50)
print("3. PITFALL: SOLUTIONS")
print("=" * 50)

print("\\nSolution 1: Use One-Hot Encoding for nominal data")
print("  pd.get_dummies(df, columns=['Color'])")

print("\\nSolution 2: Use OrdinalEncoder for ordered categories")
print("  from sklearn.preprocessing import OrdinalEncoder")
print("  encoder = OrdinalEncoder(categories=[['Bad', 'Good', 'Excellent']])")

print("\\n" + "=" * 50)
print("4. PITFALL: UNKNOWN CATEGORIES IN TEST DATA")
print("=" * 50)

# Simulate train-test split
train_data = df.iloc[:6]  # First 6 rows
test_data = df.iloc[6:]   # Last 3 rows

le_train = LabelEncoder()
train_data['Color_Encoded'] = le_train.fit_transform(train_data['Color'])

print("Training data color mapping:")
for color, encoded in zip(train_data['Color'].unique(), le_train.transform(train_data['Color'].unique())):
    print(f"  {color} → {encoded}")

print("\\nTest data:")
print(test_data['Color'])

print("\\nIssue: If test data has a category not in training data, fit_transform will fail")
print("Solution: Always fit on training data, then transform test data")

print("\\n" + "=" * 50)
print("5. BEST PRACTICE: FIT ON TRAINING ONLY")
print("=" * 50)

# Proper way
le = LabelEncoder()
le.fit(train_data['Color'])  # Fit on training data only
train_data['Color_Encoded'] = le.transform(train_data['Color'])
test_data['Color_Encoded'] = le.transform(test_data['Color'])  # Transform test data

print("Train data:")
print(train_data[['Color', 'Color_Encoded']])
print("\\nTest data:")
print(test_data[['Color', 'Color_Encoded']])`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Label Encoding?",
      options: [
        "Creating binary columns for each category",
        "Assigning a unique integer to each category",
        "Replacing categories with their frequency",
        "Using target values to encode categories"
      ],
      correctAnswer: 1,
      explanation: "Label Encoding assigns a unique integer to each category (e.g., Red=0, Blue=1, Green=2).",
    },
    {
      id: 2,
      question: "What is the main problem with Label Encoding for nominal data?",
      options: [
        "It's too slow",
        "It creates a false ordinal relationship between categories",
        "It removes the data",
        "It's not supported in scikit-learn"
      ],
      correctAnswer: 1,
      explanation: "Label Encoding creates an artificial ordering (0 < 1 < 2), which can mislead models into thinking there's an ordinal relationship.",
    },
    {
      id: 3,
      question: "What is the default order used by LabelEncoder in scikit-learn?",
      options: [
        "Random order",
        "Alphabetical order",
        "Frequency order",
        "Input order"
      ],
      correctAnswer: 1,
      explanation: "LabelEncoder in scikit-learn assigns integers in alphabetical order by default.",
    },
    {
      id: 4,
      question: "Which scikit-learn class is used for Label Encoding?",
      options: [
        "OneHotEncoder",
        "LabelEncoder",
        "OrdinalEncoder",
        "TargetEncoder"
      ],
      correctAnswer: 1,
      explanation: "LabelEncoder is the scikit-learn class specifically designed for label encoding.",
    },
    {
      id: 5,
      question: "What does the inverse_transform() method do in LabelEncoder?",
      options: [
        "Encodes the data again",
        "Converts encoded values back to original categories",
        "Scales the data",
        "Removes the encoding"
      ],
      correctAnswer: 1,
      explanation: "inverse_transform() converts encoded integer values back to their original categorical labels.",
    },
    {
      id: 6,
      question: "When should you use Label Encoding?",
      options: [
        "For nominal data with many categories",
        "For ordinal data where order matters",
        "For binary data only",
        "For continuous data"
      ],
      correctAnswer: 1,
      explanation: "Label Encoding is best suited for ordinal data where the order of categories is meaningful.",
    },
    {
      id: 7,
      question: "What is the problem with using Label Encoding for high cardinality features?",
      options: [
        "It creates too many columns",
        "It assigns arbitrary numbers with no meaning",
        "It's computationally expensive",
        "It doesn't work with high cardinality"
      ],
      correctAnswer: 1,
      explanation: "Label Encoding assigns arbitrary numbers that may not reflect the true relationships between categories.",
    },
    {
      id: 8,
      question: "What should you do if you have a category in test data that wasn't in training data?",
      options: [
        "Fit the encoder on test data",
        "Fit the encoder on combined data",
        "Handle unseen categories before encoding",
        "Skip that row"
      ],
      correctAnswer: 2,
      explanation: "You should handle unseen categories before encoding, such as creating an 'unknown' category or using handle_unknown='ignore'.",
    },
    {
      id: 9,
      question: "What is the difference between Label Encoding and Ordinal Encoding?",
      options: [
        "They are the same",
        "Label Encoding assigns arbitrary integers, Ordinal Encoding respects custom order",
        "Ordinal Encoding is for classification",
        "Label Encoding is for regression"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding explicitly allows you to specify the order of categories, while Label Encoding assigns arbitrary integers.",
    },
    {
      id: 10,
      question: "Which encoding method is memory efficient for high cardinality features?",
      options: [
        "One-Hot Encoding",
        "Label Encoding",
        "Binary Encoding",
        "Frequency Encoding"
      ],
      correctAnswer: 1,
      explanation: "Label Encoding is memory efficient because it creates only one column regardless of the number of categories.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
     

        {/* 1. What is Label Encoding? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Hash className="h-5 w-5 text-primary" />
            1. What is Label Encoding?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Label Encoding is a technique that <span className="font-semibold text-foreground">converts categorical variables into numerical format</span> by assigning a unique integer to each category.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-lg">Red → 0, Blue → 1, Green → 2</p>
                <p className="text-xs text-muted-foreground mt-2">Each category gets a unique integer</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Characteristics</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Assigns integers from 0 to n-1</li>
                    <li>Single column output</li>
                    <li>Memory efficient</li>
                    <li>Works with any number of categories</li>
                    <li>Fast and simple</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. When to Use Label Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            2. When to Use Label Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Best Use Cases
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Ordinal Data</span> - Categories with natural order</li>
                <li>• <span className="font-medium text-foreground">Tree-based Models</span> - Decision Trees, Random Forest</li>
                <li>• <span className="font-medium text-foreground">High Cardinality</span> - Many categories</li>
                <li>• <span className="font-medium text-foreground">Memory Constraints</span> - Single column output</li>
                <li>• <span className="font-medium text-foreground">Simple Problems</span> - Quick and dirty encoding</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                When to Avoid
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Nominal Data</span> - No natural order</li>
                <li>• <span className="font-medium text-foreground">Linear Models</span> - SVM, Logistic Regression</li>
                <li>• <span className="font-medium text-foreground">Distance-based</span> - KNN, K-Means</li>
                <li>• <span className="font-medium text-foreground">When order matters</span> - Use OrdinalEncoder instead</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3. How Label Encoding Works */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            3. How Label Encoding Works
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 1</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Find unique categories</p>
                  <p className="text-xs text-muted-foreground mt-1">[Red, Blue, Green]</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 2</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Assign integers</p>
                  <p className="text-xs text-muted-foreground mt-1">Red → 0, Blue → 1, Green → 2</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 3</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Transform data</p>
                  <p className="text-xs text-muted-foreground mt-1">[Red, Blue, Green] → [0, 1, 2]</p>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-muted/50 p-3 rounded-lg">
              <p className="text-xs text-muted-foreground text-center">
                <span className="font-medium text-foreground">Note:</span> By default, LabelEncoder in scikit-learn assigns integers in alphabetical order
              </p>
            </div>
          </div>
        </section>

        {/* 4. Label Encoding in Scikit-learn */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Cog className="h-5 w-5 text-primary" />
            4. Label Encoding in Scikit-learn
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Basic Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary">from sklearn.preprocessing import LabelEncoder</p>
                <p className="font-mono text-xs text-primary mt-2">le = LabelEncoder()</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = le.fit_transform(df['category'])</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Key Methods</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium">fit()</span>: Learn mapping</li>
                    <li><span className="font-medium">transform()</span>: Apply mapping</li>
                    <li><span className="font-medium">fit_transform()</span>: Fit and transform</li>
                    <li><span className="font-medium">inverse_transform()</span>: Decode</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Important Methods</h4>
              <div className="space-y-3">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">le.classes_</p>
                  <p className="text-xs text-muted-foreground">Shows all unique categories</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">le.inverse_transform(encoded_array)</p>
                  <p className="text-xs text-muted-foreground">Convert back to original categories</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">len(le.classes_)</p>
                  <p className="text-xs text-muted-foreground">Number of unique categories</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Advantages and Disadvantages */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            5. Advantages and Disadvantages
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Advantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Simple</span> - Easy to understand and implement</li>
                <li>• <span className="font-medium text-foreground">Memory Efficient</span> - Only one column</li>
                <li>• <span className="font-medium text-foreground">Fast</span> - O(n) time complexity</li>
                <li>• <span className="font-medium text-foreground">No Dimensionality Increase</span> - Handles high cardinality</li>
                <li>• <span className="font-medium text-foreground">Works with Any Model</span> - Tree-based models especially</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                Disadvantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">False Ordinal Relationship</span> - Creates artificial ordering</li>
                <li>• <span className="font-medium text-foreground">Not for Nominal Data</span> - Can mislead models</li>
                <li>• <span className="font-medium text-foreground">Arbitrary Assignment</span> - Integers have no meaning</li>
                <li>• <span className="font-medium text-foreground">Not Scale-Invariant</span> - Sensitive to the mapping</li>
                <li>• <span className="font-medium text-foreground">Unseen Categories</span> - Cannot handle new categories</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 6. The Problem with Nominal Data */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            6. The Problem with Nominal Data
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">The Issue</h4>
                <p className="text-sm text-muted-foreground">
                  Label Encoding creates an artificial numerical order that doesn't exist in nominal data.
                </p>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <p className="font-mono text-xs">Red → 0, Blue → 1, Green → 2</p>
                  <p className="text-xs text-muted-foreground mt-1">This implies Red &lt; Blue &lt; Green, which is meaningless</p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Impact on Models</h4>
                <div className="grid md:grid-cols-2 gap-3">
                  <div className="bg-destructive/10 p-3 rounded">
                    <p className="font-medium text-foreground text-sm">Linear Models</p>
                    <p className="text-xs text-muted-foreground">
                      Will treat the numeric values as meaningful, leading to incorrect predictions
                    </p>
                  </div>
                  <div className="bg-destructive/10 p-3 rounded">
                    <p className="font-medium text-foreground text-sm">Distance-based Models</p>
                    <p className="text-xs text-muted-foreground">
                      Will treat different categories as being at different distances (0 vs 1 vs 2)
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Solution</h4>
                <div className="bg-primary/10 p-3 rounded">
                  <p className="text-sm text-muted-foreground">
                    Use <span className="font-medium text-foreground">One-Hot Encoding</span> for nominal data or 
                    <span className="font-medium text-foreground"> Ordinal Encoding</span> when order is defined.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Manual Label Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Wand2 className="h-5 w-5 text-primary" />
            7. Manual Label Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Using Dictionary Mapping</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Create mapping</p>
                <p className="font-mono text-xs text-primary mt-1">
  {`mapping = {'Red': 0, 'Blue': 1, 'Green': 2}`}
</p>
                <p className="font-mono text-xs text-primary mt-2"># Apply mapping</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['color'].map(mapping)</p>
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                <p> Full control over integer assignments</p>
                <p> Can specify custom order</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Using Category Codes</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Convert to category type</p>
                <p className="font-mono text-xs text-primary mt-1">df['color'] = df['color'].astype('category')</p>
                <p className="font-mono text-xs text-primary mt-2"># Extract codes</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['color'].cat.codes</p>
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                <p> Built-in pandas functionality</p>
                <p> Memory efficient</p>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Handling Unseen Categories */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            8. Handling Unseen Categories
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">The Problem</h4>
            <p className="text-sm text-muted-foreground mb-3">
              When test data contains categories not seen during training, LabelEncoder will throw an error.
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-destructive/10 p-3 rounded">
                <p className="font-medium text-foreground text-sm"> Wrong Approach</p>
                <div className="bg-muted p-2 rounded mt-1">
                  <p className="font-mono text-xs">le.fit_transform(df_train['category'])</p>
                  <p className="font-mono text-xs">le.transform(df_test['category'])</p>
                  <p className="text-xs text-destructive mt-1">✗ Fails if test has unseen categories</p>
                </div>
              </div>

              <div className="bg-green-50 p-3 rounded dark:bg-green-950">
                <p className="font-medium text-foreground text-sm"> Best Practice</p>
                <div className="bg-muted p-2 rounded mt-1">
                  <p className="font-mono text-xs"># Fit on training data</p>
                  <p className="font-mono text-xs">le.fit(df_train['category'])</p>
                  <p className="font-mono text-xs">df_train['encoded'] = le.transform(df_train['category'])</p>
                  <p className="font-mono text-xs"># Handle unseen categories</p>
                  <p className="font-mono text-xs">df_test['encoded'] = df_test['category'].map(</p>
                  <p className="font-mono text-xs">    lambda x: le.transform([x])[0] if x in le.classes_ else -1</p>
                  <p className="font-mono text-xs">)</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Common Pitfalls */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            9. Common Pitfalls
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">False Ordinal Relationships</h4>
                  <p className="text-sm text-muted-foreground">
                    Using Label Encoding for nominal data creates artificial ordering that misleads models.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Fitting LabelEncoder on combined train and test data can cause data leakage.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Unseen Categories</h4>
                  <p className="text-sm text-muted-foreground">
                    Test data with new categories will cause errors during transformation.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Loss of Category Identity</h4>
                  <p className="text-sm text-muted-foreground">
                    The numeric values lose the categorical meaning and context.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 10. Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            10. Best Practices
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Fit Only on Training Data</h4>
                <p className="text-sm text-muted-foreground">Always fit LabelEncoder on training data only, then transform test data</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Know Your Data</h4>
                <p className="text-sm text-muted-foreground">Understand if your data is nominal or ordinal before choosing encoding</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Handle Unseen Categories</h4>
                <p className="text-sm text-muted-foreground">Create an 'unknown' category or use manual mapping with fallback</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Save the Encoder</h4>
                <p className="text-sm text-muted-foreground">Save your fitted LabelEncoder for future predictions</p>
              </div>
            </div>
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Code Examples
          </h2>
          <p className="text-muted-foreground mb-4">
            See label encoding in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Label Encoding Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}