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
  Tag,
  Columns,
  SquareSplitVertical,
  SquareSplitHorizontal,
  ArrowUp,
  ArrowDown,
  SortAsc,
  SortDesc,
  ListOrdered,
  Crosshair,

  ToggleLeft,
  ToggleRight,

  Power,
  CircleDot,
  CircleOff
} from "lucide-react";

export default function BinaryEncodingPage() {
  const result = getSubtopicBySlug("machine-learning", "binary-encoding");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-basic",
      label: "Basic Binary Encoding",
      code: `import pandas as pd
import numpy as np

# Create sample data with binary categories
data = {
    'Is_Member': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female'],
    'Has_Insurance': ['True', 'False', 'True', 'True', 'False', 'True', 'False', 'True'],
    'Status': ['Active', 'Inactive', 'Active', 'Inactive', 'Active', 'Active', 'Inactive', 'Active']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. BINARY ENCODING - METHOD 1 (Map)")
print("=" * 50)

# Method 1: Using map with dictionary
mapping = {
    'Yes': 1,
    'No': 0,
    'Male': 1,
    'Female': 0,
    'True': 1,
    'False': 0,
    'Active': 1,
    'Inactive': 0
}

df['Is_Member_Encoded'] = df['Is_Member'].map(mapping)
df['Gender_Encoded'] = df['Gender'].map(mapping)
df['Has_Insurance_Encoded'] = df['Has_Insurance'].map(mapping)
df['Status_Encoded'] = df['Status'].map(mapping)

print("Encoded data:")
print(df)

print("\\n" + "=" * 50)
print("3. BINARY ENCODING - METHOD 2 (Boolean to int)")
print("=" * 50)

# Method 2: Convert boolean to integer
df['Is_Member_Bool'] = (df['Is_Member'] == 'Yes').astype(int)
df['Gender_Bool'] = (df['Gender'] == 'Male').astype(int)
df['Has_Insurance_Bool'] = (df['Has_Insurance'] == 'True').astype(int)
df['Status_Bool'] = (df['Status'] == 'Active').astype(int)

print("Boolean encoding:")
print(df[['Is_Member', 'Is_Member_Bool', 'Gender', 'Gender_Bool', 'Has_Insurance', 'Has_Insurance_Bool']])

print("\\n" + "=" * 50)
print("4. VERIFICATION")
print("=" * 50)
print("Is_Member mapping:", dict(zip(df['Is_Member'].unique(), 
                                    df['Is_Member'].unique().map(lambda x: 1 if x == 'Yes' else 0))))
print("Gender mapping:", dict(zip(df['Gender'].unique(), 
                                  df['Gender'].unique().map(lambda x: 1 if x == 'Male' else 0))))`,
    },
    {
      language: "python-sklearn",
      label: "Binary Encoding with Scikit-learn",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create sample data
data = {
    'Is_Student': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Has_Car': ['True', 'False', 'True', 'True', 'False', 'True', 'False', 'True'],
    'Is_Employed': ['Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. USING LABEL ENCODER")
print("=" * 50)

# Using LabelEncoder for binary features
le = LabelEncoder()
df_encoded = df.copy()

for col in ['Is_Student', 'Has_Car', 'Is_Employed', 'Gender']:
    df_encoded[col + '_Encoded'] = le.fit_transform(df[col])
    print(f"\\n{col} mapping:")
    for original, encoded in zip(df[col].unique(), le.transform(df[col].unique())):
        print(f"  {original} → {encoded}")

print("\\nEncoded data:")
print(df_encoded)

print("\\n" + "=" * 50)
print("3. USING ONE-HOT ENCODER (BINARY OUTPUT)")
print("=" * 50)

# OneHotEncoder with drop='first' to get binary columns
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False), ['Is_Student', 'Has_Car', 'Is_Employed', 'Gender'])
    ],
    remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(df)
feature_names = preprocessor.get_feature_names_out()
df_onehot = pd.DataFrame(X_encoded, columns=feature_names)

print("One-Hot Encoded:")
print(df_onehot)

print("\\n" + "=" * 50)
print("4. COMPARISON")
print("=" * 50)
print("LabelEncoder: Single column, 0/1 values")
print("OneHotEncoder: Two columns, 0/1 values (one dropped)")`,
    },
    {
      language: "python-multiple",
      label: "Multiple Binary Columns",
      code: `import pandas as pd
import numpy as np

# Create sample data with multiple binary categories
data = {
    'ID': [1, 2, 3, 4, 5],
    'Features': [
        ['A', 'B', 'C'],
        ['A', 'D'],
        ['B', 'E'],
        ['A', 'B', 'C', 'D'],
        ['C', 'E']
    ]
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. CREATE BINARY COLUMNS")
print("=" * 50)

# Get all unique features
all_features = set()
for features in df['Features']:
    all_features.update(features)
all_features = sorted(list(all_features))

print("Unique features:", all_features)

# Create binary columns
for feature in all_features:
    df[feature] = df['Features'].apply(lambda x: 1 if feature in x else 0)

print("\\nData with binary columns:")
print(df)

print("\\n" + "=" * 50)
print("3. BINARY ENCODING SUMMARY")
print("=" * 50)
print("Number of original columns:", df.shape[1])
print("Number of binary columns created:", len(all_features))
print("\\nFeature presence counts:")
for feature in all_features:
    count = df[feature].sum()
    print(f"  {feature}: {count} rows ({count/len(df)*100:.1f}%)")`,
    },
    {
      language: "python-custom",
      label: "Custom Binary Encoding",
      code: `import pandas as pd
import numpy as np

# Create sample data
data = {
    'User': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
    'Subscription': ['Premium', 'Basic', 'Premium', 'Free', 'Premium'],
    'Email_Verified': [True, False, True, True, False],
    'Phone_Verified': [True, True, False, True, False],
    'Preferences': ['Web', 'Mobile', 'Both', 'Web', 'Mobile']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. CUSTOM BINARY ENCODING")
print("=" * 50)

# Custom binary encoding with descriptive mappings
def encode_binary_column(df, col, true_value, encoded_col_name=None):
    if encoded_col_name is None:
        encoded_col_name = col + '_Encoded'
    df[encoded_col_name] = (df[col] == true_value).astype(int)
    return df

# Apply custom encoding
encode_binary_column(df, 'Subscription', 'Premium', 'Is_Premium')
encode_binary_column(df, 'Email_Verified', True, 'Email_Verified_Encoded')
encode_binary_column(df, 'Phone_Verified', True, 'Phone_Verified_Encoded')

# For multi-category preferences, create binary indicators
df['Is_Web'] = df['Preferences'].apply(lambda x: 1 if x in ['Web', 'Both'] else 0)
df['Is_Mobile'] = df['Preferences'].apply(lambda x: 1 if x in ['Mobile', 'Both'] else 0)

print("Encoded data:")
print(df)

print("\\n" + "=" * 50)
print("3. ENCODING MAPPINGS")
print("=" * 50)
print("Subscription → Premium: 1, Other: 0")
print("Email_Verified → True: 1, False: 0")
print("Phone_Verified → True: 1, False: 0")
print("Preferences → Web/Mobile indicators")

print("\\n" + "=" * 50)
print("4. SUMMARY STATISTICS")
print("=" * 50)
binary_cols = ['Is_Premium', 'Email_Verified_Encoded', 'Phone_Verified_Encoded', 'Is_Web', 'Is_Mobile']
for col in binary_cols:
    print(f"{col}: {df[col].sum()} out of {len(df)} ({df[col].mean()*100:.1f}%)")`,
    },
    {
      language: "python-pandas",
      label: "Advanced Pandas Binary Encoding",
      code: `import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 100

data = {
    'Is_Member': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
    'Has_Paid': np.random.choice([1, 0], n_samples, p=[0.7, 0.3]),
    'Active_User': np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
    'Risk_Tolerance': np.random.choice(['High', 'Medium', 'Low'], n_samples, p=[0.3, 0.4, 0.3])
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df.head())
print("\\nData types:")
print(df.dtypes)

print("\\n" + "=" * 50)
print("2. USING .astype() FOR BOOLEAN CONVERSION")
print("=" * 50)

# Convert string binary to boolean then to int
df['Is_Member_Bool'] = df['Is_Member'].map({'Yes': True, 'No': False})
df['Is_Member_Encoded'] = df['Is_Member_Bool'].astype(int)

# Direct conversion for existing booleans
df['Active_User_Encoded'] = df['Active_User'].astype(int)

# For numeric binary (1/0 already)
df['Has_Paid_Encoded'] = df['Has_Paid']

print("Encoded data:")
print(df[['Is_Member', 'Is_Member_Encoded', 'Active_User', 'Active_User_Encoded', 'Has_Paid', 'Has_Paid_Encoded']].head())

print("\\n" + "=" * 50)
print("3. ONE-HOT ENCODING FOR MULTI-CLASS BINARY INDICATORS")
print("=" * 50)

# Create binary indicators for Risk_Tolerance
df_risk = pd.get_dummies(df['Risk_Tolerance'], prefix='Risk')
df = pd.concat([df, df_risk], axis=1)

print("Risk indicators:")
print(df_risk.head())

print("\\n" + "=" * 50)
print("4. SUMMARY")
print("=" * 50)
binary_columns = ['Is_Member_Encoded', 'Has_Paid_Encoded', 'Active_User_Encoded', 'Risk_High', 'Risk_Low', 'Risk_Medium']
for col in binary_columns:
    if col in df.columns:
        print(f"{col}: {df[col].sum():.0f} / {len(df)} ({df[col].mean()*100:.1f}%)")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Binary Encoding?",
      options: [
        "Encoding categories with multiple values",
        "Converting binary categories (0/1, Yes/No) to numeric values",
        "Creating binary columns for each category",
        "Assigning random numbers to categories"
      ],
      correctAnswer: 1,
      explanation: "Binary Encoding converts categories that have two possible values (e.g., Yes/No, True/False) to numeric 0/1 values.",
    },
    {
      id: 2,
      question: "Which values are typically used in Binary Encoding?",
      options: [
        "0 and 1",
        "-1 and 1",
        "1 and 2",
        "0 and 100"
      ],
      correctAnswer: 0,
      explanation: "Binary Encoding typically uses 0 and 1 to represent the two categories.",
    },
    {
      id: 3,
      question: "What is the difference between Binary Encoding and One-Hot Encoding?",
      options: [
        "They are the same",
        "Binary Encoding creates one column, One-Hot creates multiple columns",
        "Binary Encoding creates multiple columns, One-Hot creates one column",
        "Binary Encoding is for numeric data"
      ],
      correctAnswer: 1,
      explanation: "Binary Encoding creates a single column with 0/1 values, while One-Hot Encoding creates multiple binary columns.",
    },
    {
      id: 4,
      question: "How do you convert 'Yes'/'No' values to binary?",
      options: [
        "Using astype('int')",
        "Mapping 'Yes' to 1 and 'No' to 0",
        "Using LabelEncoder",
        "All of the above"
      ],
      correctAnswer: 3,
      explanation: "You can use mapping, boolean conversion with astype(int), or LabelEncoder to convert 'Yes'/'No' to 0/1.",
    },
    {
      id: 5,
      question: "What is the advantage of Binary Encoding?",
      options: [
        "It creates many columns",
        "It's simple and memory efficient",
        "It preserves order",
        "It handles high cardinality"
      ],
      correctAnswer: 1,
      explanation: "Binary Encoding is simple and memory efficient because it creates only one column.",
    },
    {
      id: 6,
      question: "When should you use Binary Encoding?",
      options: [
        "For ordinal data",
        "For binary categorical variables (2 categories)",
        "For high cardinality features",
        "For continuous data"
      ],
      correctAnswer: 1,
      explanation: "Binary Encoding is specifically designed for variables that have exactly two categories.",
    },
    {
      id: 7,
      question: "What is the output of converting a boolean column with astype(int) in pandas?",
      options: [
        "String values",
        "0 for False, 1 for True",
        "1 for False, 0 for True",
        "Error"
      ],
      correctAnswer: 1,
      explanation: "astype(int) on a boolean column converts False to 0 and True to 1.",
    },
    {
      id: 8,
      question: "How do you handle multi-class variables with Binary Encoding?",
      options: [
        "You can't",
        "Create multiple binary indicator columns",
        "Use Label Encoding instead",
        "Use One-Hot Encoding"
      ],
      correctAnswer: 1,
      explanation: "For multi-class variables, you create multiple binary indicator columns (e.g., one per category).",
    },
    {
      id: 9,
      question: "Which algorithm works well with Binary Encoding?",
      options: [
        "Only tree-based models",
        "All algorithms",
        "Only linear models",
        "Only neural networks"
      ],
      correctAnswer: 1,
      explanation: "Binary Encoding produces numeric 0/1 values that work well with all machine learning algorithms.",
    },
    {
      id: 10,
      question: "What is the difference between Binary Encoding and Label Encoding for two categories?",
      options: [
        "They are the same",
        "Binary Encoding uses 0/1, Label Encoding can use any numbers",
        "Binary Encoding uses 1/2, Label Encoding uses 0/1",
        "Label Encoding only works with strings"
      ],
      correctAnswer: 1,
      explanation: "For two categories, Binary Encoding specifically uses 0 and 1, while Label Encoding could assign any numbers.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
      

        {/* 1. What is Binary Encoding? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Binary className="h-5 w-5 text-primary" />
            1. What is Binary Encoding?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Binary Encoding is the simplest form of categorical encoding that <span className="font-semibold text-foreground">converts binary categories</span> (variables with exactly two possible values) into <span className="font-semibold text-foreground">numeric 0 and 1 values</span>.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center gap-8">
                  <div className="text-center">
                    <p className="font-mono">Yes</p>
                    <p className="text-xs text-primary font-bold">→ 1</p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono">No</p>
                    <p className="text-xs text-primary font-bold">→ 0</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Binary categories → 0/1 values</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Characteristics</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Only two possible values</li>
                    <li>Single column output</li>
                    <li>Most memory efficient</li>
                    <li>Works with all algorithms</li>
                    <li>Simplest encoding method</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Types of Binary Data */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            2. Types of Binary Data
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">String Binary</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs">'Yes' / 'No'</p>
                  <p className="font-mono text-xs">'True' / 'False'</p>
                  <p className="font-mono text-xs">'Male' / 'Female'</p>
                  <p className="font-mono text-xs">'Active' / 'Inactive'</p>
                </div>
                <p className="text-muted-foreground">Need mapping or boolean conversion</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Boolean Binary</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs">True / False</p>
                  <p className="font-mono text-xs">1 / 0</p>
                </div>
                <p className="text-muted-foreground">Direct conversion using astype(int)</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Numeric Binary</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs">1 / 0</p>
                  <p className="font-mono text-xs">-1 / 1</p>
                </div>
                <p className="text-muted-foreground">Already numeric, may need standardization</p>
              </div>
            </div>
          </div>
        </section>

        {/* 3. How Binary Encoding Works */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            3. How Binary Encoding Works
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 1</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Identify categories</p>
                  <p className="text-xs text-muted-foreground mt-1">[Yes, No]</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 2</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Assign values</p>
                  <p className="text-xs text-muted-foreground mt-1">Yes → 1, No → 0</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 3</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Transform</p>
                  <p className="text-xs text-muted-foreground mt-1">[Yes, No, Yes] → [1, 0, 1]</p>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-muted/50 p-3 rounded-lg">
              <p className="text-xs text-muted-foreground text-center">
                <span className="font-medium text-foreground">Note:</span> Binary Encoding is equivalent to Label Encoding with exactly 2 categories
              </p>
            </div>
          </div>
        </section>

        {/* 4. Binary Encoding Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Cog className="h-5 w-5 text-primary" />
            4. Binary Encoding Methods
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Method 1: Mapping</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Dictionary mapping</p>
               <p className="font-mono text-xs text-primary mt-1">
  {`mapping = {'Yes': 1, 'No': 0}`}
</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['column'].map(mapping)</p>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                <p> Full control over mapping</p>
                <p> Easy to understand</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Method 2: Boolean Conversion</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Direct conversion</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = (df['column'] == 'Yes').astype(int)</p>
                <p className="font-mono text-xs text-primary mt-2"># For boolean columns</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['boolean_col'].astype(int)</p>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                <p> Simple one-liner</p>
                <p> Works with any condition</p>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mt-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Method 3: Scikit-learn</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary">from sklearn.preprocessing import LabelEncoder</p>
                <p className="font-mono text-xs text-primary mt-1">le = LabelEncoder()</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = le.fit_transform(df['column'])</p>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                <p> Works with all categories</p>
                <p> Handles unseen categories</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Method 4: Pandas Category</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Convert to category</p>
                <p className="font-mono text-xs text-primary mt-1">df['column'] = df['column'].astype('category')</p>
                <p className="font-mono text-xs text-primary mt-2"># Extract codes</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['column'].cat.codes</p>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                <p> Memory efficient</p>
                <p> Works with large datasets</p>
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
                <li>• <span className="font-medium text-foreground">Simple</span> - Very easy to understand and implement</li>
                <li>• <span className="font-medium text-foreground">Memory Efficient</span> - Only one column</li>
                <li>• <span className="font-medium text-foreground">Works with All Models</span> - Universal compatibility</li>
                <li>• <span className="font-medium text-foreground">Fast</span> - Quick transformation</li>
                <li>• <span className="font-medium text-foreground">Interpretable</span> - Clear meaning (1=Yes, 0=No)</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                Disadvantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Only for Binary</span> - Cannot handle more than 2 categories</li>
                <li>• <span className="font-medium text-foreground">Assumes 0/1 Meaning</span> - May not be appropriate for some domains</li>
                <li>• <span className="font-medium text-foreground">No Order Information</span> - 0 and 1 imply order</li>
                <li>• <span className="font-medium text-foreground">Limited Use</span> - Not applicable to multi-category features</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 6. Binary vs Other Encodings */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            6. Binary Encoding vs Other Methods
          </h2>

          <div className="bg-card border border-border rounded-lg p-4 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-2">Method</th>
                  <th className="text-left p-2">Number of Columns</th>
                  <th className="text-left p-2">Memory Usage</th>
                  <th className="text-left p-2">Works With</th>
                  <th className="text-left p-2">Order</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Binary Encoding</td>
                  <td className="text-muted-foreground">1</td>
                  <td className="text-muted-foreground">Lowest</td>
                  <td className="text-muted-foreground">Binary only</td>
                  <td className="text-muted-foreground">2 values</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Label Encoding</td>
                  <td className="text-muted-foreground">1</td>
                  <td className="text-muted-foreground">Low</td>
                  <td className="text-muted-foreground">Any cardinality</td>
                  <td className="text-muted-foreground">Alphabetical</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">One-Hot Encoding</td>
                  <td className="text-muted-foreground">N</td>
                  <td className="text-muted-foreground">High</td>
                  <td className="text-muted-foreground">Any cardinality</td>
                  <td className="text-muted-foreground">None</td>
                </tr>
                <tr>
                  <td className="p-2 font-medium text-primary">Target Encoding</td>
                  <td className="text-muted-foreground">1</td>
                  <td className="text-muted-foreground">Low</td>
                  <td className="text-muted-foreground">Any cardinality</td>
                  <td className="text-muted-foreground">Target-based</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 7. Creating Multiple Binary Indicators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Grid className="h-5 w-5 text-primary" />
            7. Creating Multiple Binary Indicators
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">When you have more than 2 categories</h4>
                <p className="text-sm text-muted-foreground">
                  You can create multiple binary indicator columns for each category (like One-Hot Encoding):
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-muted p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Example</p>
                  <p className="text-xs text-muted-foreground">Category: ['A', 'B', 'C']</p>
                  <p className="text-xs text-muted-foreground mt-2">→ Is_A: [1, 0, 0]</p>
                  <p className="text-xs text-muted-foreground">→ Is_B: [0, 1, 0]</p>
                  <p className="text-xs text-muted-foreground">→ Is_C: [0, 0, 1]</p>
                </div>

                <div className="bg-primary/5 p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Implementation</p>
                  <div className="bg-background p-2 rounded mt-1">
                    <p className="font-mono text-xs"># Using get_dummies</p>
                    <p className="font-mono text-xs">df_binary = pd.get_dummies(df, columns=['category'])</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Common Pitfalls */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            8. Common Pitfalls
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Using on Multi-Category Data</h4>
                  <p className="text-sm text-muted-foreground">
                    Binary Encoding only works with exactly 2 categories. For more, use One-Hot or other methods.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Incorrect Mapping</h4>
                  <p className="text-sm text-muted-foreground">
                    Mapping 'Yes' to 0 and 'No' to 1 can be confusing. Keep it consistent (typically 1=True/Yes).
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Assuming Order</h4>
                 <p className="text-sm text-muted-foreground">
  Binary encoding implies 0 &lt; 1, which may not be meaningful for some binary variables.
</p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Type Issues</h4>
                  <p className="text-sm text-muted-foreground">
                    Mixing string and boolean representations can cause errors. Ensure consistent types.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            9. Best Practices
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Consistent Mapping</h4>
                <p className="text-sm text-muted-foreground">Always map 1 to the positive/true category and 0 to the negative/false</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Check Data Types</h4>
                <p className="text-sm text-muted-foreground">Ensure your data is clean and consistently typed before encoding</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Document Your Encoding</h4>
                <p className="text-sm text-muted-foreground">Keep track of what 0 and 1 represent for interpretability</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Built-in Methods</h4>
                <p className="text-sm text-muted-foreground">Leverage pandas and sklearn functions for clean, efficient code</p>
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
            See binary encoding in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Binary Encoding Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}