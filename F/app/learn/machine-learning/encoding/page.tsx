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
  Grid3x3
} from "lucide-react";

export default function EncodingPage() {
  const result = getSubtopicBySlug("machine-learning", "encoding");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-label",
      label: "Label Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create sample data
data = {
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small'],
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'Price': [10, 20, 30, 15, 25, 35, 12, 22]
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. LABEL ENCODING")
print("=" * 50)

# Label Encoding for each categorical column
le = LabelEncoder()
df_encoded = df.copy()

for col in ['Size', 'Color', 'Product']:
    df_encoded[col + '_Encoded'] = le.fit_transform(df[col])
    # Show mapping for each column
    unique_values = df[col].unique()
    encoded_values = le.transform(unique_values)
    print(f"\\n{col} Mapping:")
    for original, encoded in zip(unique_values, encoded_values):
        print(f"  {original} → {encoded}")

print("\\n" + "=" * 50)
print("3. ENCODED DATA")
print("=" * 50)
print(df_encoded)

# Note about LabelEncoder on multiple columns
print("\\n" + "=" * 50)
print("4. IMPORTANT NOTE")
print("=" * 50)
print("LabelEncoder only works with single column at a time.")
print("Use OrdinalEncoder for multiple columns:")`,
    },
    {
      language: "python-onehot",
      label: "One-Hot Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create sample data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small', 'Large'],
    'Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)
print("\\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].unique()}")

print("\\n" + "=" * 50)
print("2. ONE-HOT ENCODING (PANDAS)")
print("=" * 50)

# Using pandas get_dummies
df_onehot = pd.get_dummies(df, columns=['Color', 'Size', 'Category'])
print("Encoded data shape:", df_onehot.shape)
print("Encoded columns:", df_onehot.columns.tolist())
print("\\nFirst 5 rows:")
print(df_onehot.head())

print("\\n" + "=" * 50)
print("3. ONE-HOT ENCODING (SKLEARN)")
print("=" * 50)

# Using sklearn OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = encoder.fit_transform(df[['Color', 'Size', 'Category']])

# Get feature names
feature_names = encoder.get_feature_names_out(['Color', 'Size', 'Category'])
df_sklearn = pd.DataFrame(encoded_array, columns=feature_names)

print("Encoded data shape:", df_sklearn.shape)
print("Encoded columns:", df_sklearn.columns.tolist())
print("\\nFirst 5 rows:")
print(df_sklearn.head())

print("\\n" + "=" * 50)
print("4. DROPPING FIRST CATEGORY (Avoid Multicollinearity)")
print("=" * 50)

df_drop_first = pd.get_dummies(df, columns=['Color', 'Size', 'Category'], drop_first=True)
print("Encoded data shape:", df_drop_first.shape)
print("Encoded columns:", df_drop_first.columns.tolist())
print("\\nFirst 5 rows:")
print(df_drop_first.head())`,
    },
    {
      language: "python-ordinal",
      label: "Ordinal Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Create sample data
data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large', 'Medium', 'Extra Large'],
    'Experience': ['Junior', 'Mid', 'Senior', 'Lead', 'Mid', 'Senior', 'Junior', 'Lead']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. DEFINING ORDINAL ORDER")
print("=" * 50)

# Define the order of categories (ordered from lowest to highest)
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
size_order = ['Small', 'Medium', 'Large', 'Extra Large']
experience_order = ['Junior', 'Mid', 'Senior', 'Lead']

print("Education order:", education_order)
print("Size order:", size_order)
print("Experience order:", experience_order)

print("\\n" + "=" * 50)
print("3. ORDINAL ENCODING")
print("=" * 50)

# Using sklearn OrdinalEncoder with categories order
encoder = OrdinalEncoder(
    categories=[education_order, size_order, experience_order]
)
df_encoded = pd.DataFrame(
    encoder.fit_transform(df),
    columns=df.columns
)

print("Encoded data:")
print(df_encoded)

print("\\n" + "=" * 50)
print("4. MAPPING VERIFICATION")
print("=" * 50)

print("\\nEducation mapping:")
for i, level in enumerate(education_order):
    print(f"  {level} → {i}")

print("\\nSize mapping:")
for i, level in enumerate(size_order):
    print(f"  {level} → {i}")

print("\\nExperience mapping:")
for i, level in enumerate(experience_order):
    print(f"  {level} → {i}")

print("\\n" + "=" * 50)
print("5. REVERSE MAPPING (Decoding)")
print("=" * 50)

df_decoded = pd.DataFrame(
    encoder.inverse_transform(df_encoded.values),
    columns=df.columns
)
print("Decoded data (should match original):")
print(df_decoded.head())`,
    },
    {
      language: "python-target",
      label: "Target Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Create sample data
np.random.seed(42)
n_samples = 500

data = {
    'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
    'Product': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
    'Feature1': np.random.randn(n_samples),
    'Feature2': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

# Create target with some relationship to categories
category_effects = {'A': 0.8, 'B': 0.5, 'C': 0.2, 'D': -0.3, 'E': -0.6}
product_effects = {'X': 0.4, 'Y': 0.1, 'Z': -0.2, 'W': -0.5}
df['Target'] = (
    df['Category'].map(category_effects) + 
    df['Product'].map(product_effects) + 
    np.random.randn(n_samples) * 0.3
)
df['Target_Binary'] = (df['Target'] > 0).astype(int)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print("First 5 rows:")
print(df.head())
print("\\nTarget statistics:")
print(df['Target'].describe())

print("\\n" + "=" * 50)
print("2. TARGET ENCODING (Mean Encoding)")
print("=" * 50)

# Calculate mean target for each category
category_means = df.groupby('Category')['Target'].mean()
product_means = df.groupby('Product')['Target'].mean()

print("\\nCategory means:")
print(category_means)

print("\\nProduct means:")
print(product_means)

# Apply target encoding
df['Category_TargetEncoded'] = df['Category'].map(category_means)
df['Product_TargetEncoded'] = df['Product'].map(product_means)

print("\\n" + "=" * 50)
print("3. ENCODED DATA")
print("=" * 50)
print("First 5 rows:")
print(df[['Category', 'Category_TargetEncoded', 'Product', 'Product_TargetEncoded']].head())

print("\\n" + "=" * 50)
print("4. TARGET ENCODING WITH SMOOTHING")
print("=" * 50)

# Calculate smoothing
global_mean = df['Target'].mean()
smoothing_factor = 10

def smooth_target_encoding(df, col, target, global_mean, smoothing):
    means = df.groupby(col)[target].agg(['mean', 'count'])
    means['smoothed'] = (means['mean'] * means['count'] + global_mean * smoothing) / (means['count'] + smoothing)
    return means['smoothed']

smoothed_category = smooth_target_encoding(df, 'Category', 'Target', global_mean, smoothing_factor)
print("Smoothed category encodings:")
print(smoothed_category)

print("\\n" + "=" * 50)
print("5. AVOID OVERFITTING - K-FOLD TARGET ENCODING")
print("=" * 50)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
df['Category_TargetEncoded_CV'] = np.nan

for train_idx, val_idx in kf.split(df):
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    # Calculate encoding on training fold
    fold_means = train_data.groupby('Category')['Target'].mean()
    
    # Apply to validation fold
    df.loc[val_idx, 'Category_TargetEncoded_CV'] = val_data['Category'].map(fold_means)

print("Cross-validated target encoding:")
print(df[['Category', 'Target', 'Category_TargetEncoded_CV']].head(10))`,
    },
    {
      language: "python-frequency",
      label: "Frequency & Binary Encoding",
      code: `import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'Department': np.random.choice(
        ['IT', 'HR', 'Finance', 'Marketing', 'Sales', 'Operations', 'R&D'], 
        n_samples,
        p=[0.3, 0.15, 0.2, 0.1, 0.1, 0.1, 0.05]
    ),
    'Product': np.random.choice(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        n_samples,
        p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    ),
    'Feature1': np.random.randn(n_samples),
    'Feature2': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print("First 5 rows:")
print(df.head())
print("\\nUnique values:")
for col in ['Department', 'Product']:
    print(f"  {col}: {df[col].nunique()} unique values")

print("\\n" + "=" * 50)
print("2. FREQUENCY ENCODING")
print("=" * 50)

# Calculate frequencies
department_freq = df['Department'].value_counts(normalize=True)
product_freq = df['Product'].value_counts(normalize=True)

print("\\nDepartment frequencies:")
print(department_freq)

print("\\nProduct frequencies:")
print(product_freq)

# Apply frequency encoding
df['Department_Freq'] = df['Department'].map(department_freq)
df['Product_Freq'] = df['Product'].map(product_freq)

print("\\nEncoded data:")
print(df[['Department', 'Department_Freq', 'Product', 'Product_Freq']].head(10))

print("\\n" + "=" * 50)
print("3. BINARY ENCODING")
print("=" * 50)

# Binary encoding for 2-value categories
df['Is_Senior'] = np.random.choice(['Yes', 'No'], n_samples)
df['Has_Experience'] = np.random.choice(['True', 'False'], n_samples)

# Binary encoding
df['Is_Senior_Binary'] = (df['Is_Senior'] == 'Yes').astype(int)
df['Has_Experience_Binary'] = (df['Has_Experience'] == 'True').astype(int)

print("Binary encoding:")
print(df[['Is_Senior', 'Is_Senior_Binary', 'Has_Experience', 'Has_Experience_Binary']].head(10))

print("\\n" + "=" * 50)
print("4. IMPACT ON DATA")
print("=" * 50)

# Show impact of frequency encoding
print("Correlation with target (if target existed):")
print("Department frequency encoding captures category importance")

print("\\n" + "=" * 50)
print("5. MEMORY COMPARISON")
print("=" * 50)

original_memory = df[['Department', 'Product']].memory_usage(deep=True).sum()
encoded_memory = df[['Department_Freq', 'Product_Freq']].memory_usage(deep=True).sum()

print(f"Original memory: {original_memory / 1024:.2f} KB")
print(f"Encoded memory: {encoded_memory / 1024:.2f} KB")
print(f"Memory reduction: {(1 - encoded_memory / original_memory) * 100:.2f}%")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is categorical encoding in machine learning?",
      options: [
        "Converting numerical data to categorical",
        "Converting categorical data to numerical format",
        "Removing categorical data",
        "Scaling categorical data"
      ],
      correctAnswer: 1,
      explanation: "Categorical encoding is the process of converting categorical variables into numerical format that machine learning algorithms can work with.",
    },
    {
      id: 2,
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
      id: 3,
      question: "What is the problem with Label Encoding for nominal data?",
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
      id: 4,
      question: "What is One-Hot Encoding?",
      options: [
        "Assigning integers to categories",
        "Creating binary columns for each category",
        "Using target means for encoding",
        "Assigning frequencies to categories"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding creates a binary column for each category, with a 1 indicating the presence of that category.",
    },
    {
      id: 5,
      question: "What is the disadvantage of One-Hot Encoding?",
      options: [
        "It creates too many columns for high-cardinality features",
        "It's not supported in Python",
        "It removes data",
        "It's computationally expensive"
      ],
      correctAnswer: 0,
      explanation: "One-Hot Encoding creates one column per category, which can lead to high dimensionality with many categories.",
    },
    {
      id: 6,
      question: "What is the difference between Label Encoding and Ordinal Encoding?",
      options: [
        "They are the same",
        "Label Encoding assigns arbitrary integers, Ordinal Encoding respects order",
        "Ordinal Encoding is for classification",
        "Label Encoding is for regression"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding explicitly respects the order of categories (e.g., Small=0, Medium=1, Large=2), while Label Encoding assigns arbitrary integers.",
    },
    {
      id: 7,
      question: "What is Target Encoding?",
      options: [
        "Encoding based on the target variable's mean per category",
        "Encoding using the target's frequency",
        "Encoding using binary values",
        "Encoding using random numbers"
      ],
      correctAnswer: 0,
      explanation: "Target Encoding replaces categories with the mean (or other statistic) of the target variable for that category.",
    },
    {
      id: 8,
      question: "What is the risk of Target Encoding?",
      options: [
        "It's too slow",
        "It can cause overfitting",
        "It doesn't work with numeric data",
        "It's not supported in Python"
      ],
      correctAnswer: 1,
      explanation: "Target Encoding can cause overfitting because it uses information from the target variable. Use cross-validation or smoothing to mitigate.",
    },
    {
      id: 9,
      question: "What is Frequency Encoding?",
      options: [
        "Encoding based on category frequency",
        "Encoding using binary values",
        "Encoding using target means",
        "Encoding using random values"
      ],
      correctAnswer: 0,
      explanation: "Frequency Encoding replaces categories with their frequency (or proportion) in the dataset.",
    },
    {
      id: 10,
      question: "When should you use One-Hot Encoding vs Target Encoding?",
      options: [
        "One-Hot for high cardinality, Target for low cardinality",
        "One-Hot for low cardinality, Target for high cardinality",
        "They are interchangeable",
        "Use One-Hot for regression, Target for classification"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding works well for low-cardinality features, while Target Encoding is better for high-cardinality features to avoid creating too many columns.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
      

        {/* 1. What is Categorical Encoding? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Binary className="h-5 w-5 text-primary" />
            1. What is Categorical Encoding?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Categorical encoding is the process of converting <span className="font-semibold text-foreground">categorical (non-numeric) variables</span> into numerical format that machine learning algorithms can work with.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Categories</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">Encoding</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">Numbers</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Making categorical data machine-readable</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Types of Categorical Data</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium text-foreground">Nominal</span>: No order (e.g., Colors, Countries)</li>
                    <li><span className="font-medium text-foreground">Ordinal</span>: Has order (e.g., Small, Medium, Large)</li>
                    <li><span className="font-medium text-foreground">Binary</span>: Two categories (e.g., Yes/No)</li>
                    <li><span className="font-medium text-foreground">High Cardinality</span>: Many categories (e.g., ZIP codes)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Label Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Hash className="h-5 w-5 text-primary" />
            2. Label Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Definition</h4>
              <p className="text-sm text-muted-foreground">
                Assigns a unique integer to each category. Simple and memory-efficient.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center mt-3">
                <p className="font-mono text-sm">Red → 0, Blue → 1, Green → 2</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Simple and fast</li>
                    <li>Memory efficient</li>
                    <li>Works with many algorithms</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Creates false ordinal relationship</li>
                    <li>Can mislead models</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary">from sklearn.preprocessing import LabelEncoder</p>
                <p className="font-mono text-xs text-primary mt-2">le = LabelEncoder()</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = le.fit_transform(df['category'])</p>
              </div>
              <div className="mt-3">
                <h5 className="font-medium text-foreground text-sm">When to use</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Ordinal data (order matters)</li>
                  <li>Tree-based models (can handle)</li>
                  <li>When you have many categories</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 3. One-Hot Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Grid3x3 className="h-5 w-5 text-primary" />
            3. One-Hot Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Definition</h4>
              <p className="text-sm text-muted-foreground">
                Creates binary (0/1) columns for each category. One column per category.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center mt-3">
                <p className="font-mono text-xs">Red: [1,0,0] | Blue: [0,1,0] | Green: [0,0,1]</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>No false ordinal relationships</li>
                    <li>Works with all algorithms</li>
                    <li>Simple interpretation</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>High dimensionality</li>
                    <li>Memory intensive</li>
                    <li>Can cause multicollinearity</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Using pandas</p>
                <p className="font-mono text-xs text-primary mt-1">df_onehot = pd.get_dummies(df, columns=['category'])</p>
                <p className="font-mono text-xs text-primary mt-2"># Using sklearn</p>
                <p className="font-mono text-xs text-primary mt-1">from sklearn.preprocessing import OneHotEncoder</p>
                <p className="font-mono text-xs text-primary mt-1">encoder = OneHotEncoder()</p>
                <p className="font-mono text-xs text-primary mt-1">encoded = encoder.fit_transform(df[['category']])</p>
              </div>
              <div className="mt-3">
                <h5 className="font-medium text-foreground text-sm">When to use</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Nominal data (no order)</li>
                  <li>Few categories (&lt; 20)</li>
                  <li>Linear models, SVM, KNN</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-4 bg-muted/50 rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">Tip: Drop First Category</h4>
            <p className="text-sm text-muted-foreground">
              Use <code className="bg-background px-1 rounded">drop_first=True</code> to avoid multicollinearity by dropping one category column.
            </p>
            <div className="bg-card p-2 rounded mt-2">
              <code className="text-xs">df_onehot = pd.get_dummies(df, columns=['category'], drop_first=True)</code>
            </div>
          </div>
        </section>

        {/* 4. Ordinal Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <List className="h-5 w-5 text-primary" />
            4. Ordinal Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Definition</h4>
              <p className="text-sm text-muted-foreground">
                Similar to Label Encoding but respects the natural order of categories.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center mt-3">
                <p className="font-mono text-xs">Small → 0, Medium → 1, Large → 2</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Preserves order information</li>
                    <li>Single column</li>
                    <li>Memory efficient</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Assumes equal spacing between categories</li>
                    <li>May not work for all algorithms</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary">from sklearn.preprocessing import OrdinalEncoder</p>
                <p className="font-mono text-xs text-primary mt-2">encoder = OrdinalEncoder(</p>
                <p className="font-mono text-xs text-primary ml-4">categories=[['Small', 'Medium', 'Large']]</p>
                <p className="font-mono text-xs text-primary">)</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = encoder.fit_transform(df[['size']])</p>
              </div>
              <div className="mt-3">
                <h5 className="font-medium text-foreground text-sm">When to use</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Ordinal data with clear order</li>
                  <li>Tree-based models</li>
                  <li>When order matters</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Target Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            5. Target Encoding (Mean Encoding)
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Definition</h4>
              <p className="text-sm text-muted-foreground">
                Replaces categories with the mean (or other statistic) of the target variable for that category.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center mt-3">
                <p className="font-mono text-xs">Category A (mean target=0.8) → 0.8</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Handles high cardinality</li>
                    <li>Captures category-target relationship</li>
                    <li>Single column</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Risk of overfitting</li>
                    <li>Data leakage if not careful</li>
                    <li>Requires target variable</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Simple mean encoding</p>
                <p className="font-mono text-xs text-primary mt-1">means = df.groupby('category')['target'].mean()</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['category'].map(means)</p>
              </div>
              <div className="mt-3">
                <h5 className="font-medium text-foreground text-sm">Avoid Overfitting</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Use K-Fold cross-validation</li>
                  <li>Apply smoothing</li>
                  <li>Use out-of-fold encoding</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Frequency Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            6. Frequency Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Definition</h4>
              <p className="text-sm text-muted-foreground">
                Replaces categories with their frequency or proportion in the dataset.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center mt-3">
                <p className="font-mono text-xs">Category A (frequency=0.3) → 0.3</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Simple and fast</li>
                    <li>No risk of overfitting</li>
                    <li>Handles high cardinality</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Loses category identity</li>
                    <li>Assumes frequency is predictive</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Frequency encoding</p>
                <p className="font-mono text-xs text-primary mt-1">freq = df['category'].value_counts(normalize=True)</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = df['category'].map(freq)</p>
              </div>
              <div className="mt-3">
                <h5 className="font-medium text-foreground text-sm">When to use</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>High cardinality features</li>
                  <li>When frequency indicates importance</li>
                  <li>Tree-based models</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Binary Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Binary className="h-5 w-5 text-primary" />
            7. Binary Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Definition</h4>
              <p className="text-sm text-muted-foreground">
                Converts binary categories (Yes/No, True/False) to 0/1 values.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center mt-3">
                <p className="font-mono text-xs">Yes → 1, No → 0</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Very simple</li>
                    <li>Works with all algorithms</li>
                    <li>Memory efficient</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Only works with 2 categories</li>
                    <li>Assumes 0/1 relationship</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Simple mapping</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = (df['category'] == 'Yes').astype(int)</p>
                <p className="font-mono text-xs text-primary mt-2"># Using map</p>
           <pre className="font-mono text-xs text-primary mt-1">
  {`df['encoded'] = df['category'].map({'Yes': 1, 'No': 0})`}
</pre>
              </div>
              <div className="mt-3">
                <h5 className="font-medium text-foreground text-sm">When to use</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Binary classification</li>
                  <li>Yes/No, True/False, Male/Female</li>
                  <li>Any two-category variable</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Encoding Comparison */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Grid className="h-5 w-5 text-primary" />
            8. Encoding Methods Comparison
          </h2>

          <div className="bg-card border border-border rounded-lg p-4 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-2">Method</th>
                  <th className="text-left p-2">Output</th>
                  <th className="text-left p-2">Best For</th>
                  <th className="text-left p-2">When to Avoid</th>
                  <th className="text-left p-2">Memory</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Label Encoding</td>
                  <td className="text-muted-foreground">Single integer</td>
                  <td className="text-muted-foreground">Ordinal data</td>
                  <td className="text-muted-foreground">Nominal data</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">One-Hot Encoding</td>
                  <td className="text-muted-foreground">Multiple binary columns</td>
                  <td className="text-muted-foreground">Nominal data (&lt;20 categories)</td>
                  <td className="text-muted-foreground">High cardinality</td>
                  <td className="text-muted-foreground">High</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Ordinal Encoding</td>
                  <td className="text-muted-foreground">Single integer</td>
                  <td className="text-muted-foreground">Ordered categories</td>
                  <td className="text-muted-foreground">No order</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Target Encoding</td>
                  <td className="text-muted-foreground">Single numeric</td>
                  <td className="text-muted-foreground">High cardinality</td>
                  <td className="text-muted-foreground">Small datasets</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Frequency Encoding</td>
                  <td className="text-muted-foreground">Single numeric</td>
                  <td className="text-muted-foreground">High cardinality</td>
                  <td className="text-muted-foreground">When frequency isn't predictive</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
                <tr>
                  <td className="p-2 font-medium text-primary">Binary Encoding</td>
                  <td className="text-muted-foreground">0 or 1</td>
                  <td className="text-muted-foreground">Binary categories</td>
                  <td className="text-muted-foreground">More than 2 categories</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
              </tbody>
            </table>
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
                    Using Label Encoding for nominal data creates artificial ordering (0 &lt; 1 &lt; 2) that misleads models.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Using target encoding without cross-validation can leak information from target to features.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">High Dimensionality</h4>
                  <p className="text-sm text-muted-foreground">
                    One-Hot Encoding creates too many columns for high-cardinality features, causing memory issues.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Unseen Categories</h4>
                  <p className="text-sm text-muted-foreground">
                    New categories in test data that weren't seen during training can cause errors.
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
                <h4 className="font-semibold text-foreground mb-1">Understand Your Data</h4>
                <p className="text-sm text-muted-foreground">Know if categories are nominal, ordinal, or binary before choosing encoding</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Consider Model Requirements</h4>
                <p className="text-sm text-muted-foreground">Tree-based models handle label encoding well, linear models need one-hot</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Avoid Data Leakage</h4>
                <p className="text-sm text-muted-foreground">Fit encoders on training data only, use cross-validation for target encoding</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Handle Unseen Categories</h4>
                <p className="text-sm text-muted-foreground">Use <code className="bg-background px-1 rounded">handle_unknown='ignore'</code> or create an 'unknown' category</p>
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
            See encoding techniques in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Categorical Encoding Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}