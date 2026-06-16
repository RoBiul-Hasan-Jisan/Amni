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
  SquareSplitHorizontal
} from "lucide-react";

export default function OneHotEncodingPage() {
  const result = getSubtopicBySlug("machine-learning", "one-hot");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-basic",
      label: "Basic One-Hot Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Create sample data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small', 'Large'],
    'Price': [10, 20, 30, 15, 25, 35, 12, 22, 32]
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. ONE-HOT ENCODING (PANDAS)")
print("=" * 50)

# Using pandas get_dummies
df_onehot = pd.get_dummies(df, columns=['Color', 'Size'])
print("Encoded data shape:", df_onehot.shape)
print("\\nEncoded columns:")
for col in df_onehot.columns:
    print(f"  {col}")
print("\\nFirst 5 rows:")
print(df_onehot.head())

print("\\n" + "=" * 50)
print("3. ONE-HOT ENCODING WITH DROP FIRST")
print("=" * 50)

# Drop first category to avoid multicollinearity
df_drop_first = pd.get_dummies(df, columns=['Color', 'Size'], drop_first=True)
print("Shape:", df_drop_first.shape)
print("\\nColumns:", df_drop_first.columns.tolist())
print("\\nFirst 5 rows:")
print(df_drop_first.head())`,
    },
    {
      language: "python-sklearn",
      label: "One-Hot Encoding (Scikit-learn)",
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

print("\\n" + "=" * 50)
print("2. ONE-HOT ENCODING (SKLEARN)")
print("=" * 50)

# Create encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform
encoded_array = encoder.fit_transform(df[['Color', 'Size', 'Category']])

# Get feature names
feature_names = encoder.get_feature_names_out(['Color', 'Size', 'Category'])
df_encoded = pd.DataFrame(encoded_array, columns=feature_names)

print("Encoded data shape:", df_encoded.shape)
print("\\nEncoded columns:")
for col in df_encoded.columns:
    print(f"  {col}")
print("\\nFirst 5 rows:")
print(df_encoded.head())

print("\\n" + "=" * 50)
print("3. ENCODER ATTRIBUTES")
print("=" * 50)
print("Categories:", encoder.categories_)
print("Feature names:", encoder.get_feature_names_out())`,
    },
    {
      language: "python-column",
      label: "ColumnTransformer for Mixed Data",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create mixed data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small', 'Large'],
    'Price': [10, 20, 30, 15, 25, 35, 12, 22, 32],
    'Quantity': [1, 2, 3, 4, 5, 6, 7, 8, 9]
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. COLUMN TRANSFORMER")
print("=" * 50)

# Define preprocessing for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False), ['Color', 'Size']),
        ('numeric', StandardScaler(), ['Price', 'Quantity'])
    ],
    remainder='passthrough'
)

# Fit and transform
X_transformed = preprocessor.fit_transform(df)

# Get feature names
feature_names = (
    preprocessor.named_transformers_['onehot'].get_feature_names_out(['Color', 'Size']).tolist() +
    ['Price', 'Quantity']
)

df_transformed = pd.DataFrame(X_transformed, columns=feature_names)

print("Transformed data:")
print(df_transformed)

print("\\n" + "=" * 50)
print("3. USING IN A PIPELINE")
print("=" * 50)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', None)  # Would add your model here
])

print("Pipeline created successfully!")
print("Transformers:", pipeline.named_steps['preprocessor'].transformers_)`,
    },
    {
      language: "python-sparse",
      label: "Sparse Matrix & Memory",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys

# Create data with high cardinality
np.random.seed(42)
n_samples = 1000
n_categories = 50

# Create high cardinality feature
categories = [f'Category_{i}' for i in range(n_categories)]
data = {
    'HighCardinality': np.random.choice(categories, n_samples),
    'LowCardinality': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'Value': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. DATA WITH HIGH CARDINALITY")
print("=" * 50)
print(f"Total samples: {len(df)}")
print(f"Unique categories in HighCardinality: {df['HighCardinality'].nunique()}")
print(f"Unique categories in LowCardinality: {df['LowCardinality'].nunique()}")

print("\\n" + "=" * 50)
print("2. DENSE VS SPARSE MATRIX")
print("=" * 50)

# Dense matrix (default)
encoder_dense = OneHotEncoder(sparse_output=False)
encoded_dense = encoder_dense.fit_transform(df[['HighCardinality']])
memory_dense = encoded_dense.nbytes / (1024 * 1024)  # MB

print(f"Dense matrix shape: {encoded_dense.shape}")
print(f"Dense matrix memory: {memory_dense:.2f} MB")

# Sparse matrix
encoder_sparse = OneHotEncoder(sparse_output=True)
encoded_sparse = encoder_sparse.fit_transform(df[['HighCardinality']])
memory_sparse = encoded_sparse.data.nbytes / (1024 * 1024)  # MB

print(f"Sparse matrix shape: {encoded_sparse.shape}")
print(f"Sparse matrix memory: {memory_sparse:.2f} MB")
print(f"Memory savings: {(1 - memory_sparse/memory_dense) * 100:.2f}%")

print("\\n" + "=" * 50)
print("3. COMPARISON FOR DIFFERENT CARDINALITIES")
print("=" * 50)

cardinalities = [5, 10, 20, 50, 100]
results = []

for n_cat in cardinalities:
    cats = [f'Cat_{i}' for i in range(n_cat)]
    data_small = {'Category': np.random.choice(cats, 1000)}
    df_small = pd.DataFrame(data_small)
    
    # Dense
    enc_dense = OneHotEncoder(sparse_output=False)
    dense = enc_dense.fit_transform(df_small[['Category']])
    mem_dense = dense.nbytes / 1024
    
    # Sparse
    enc_sparse = OneHotEncoder(sparse_output=True)
    sparse = enc_sparse.fit_transform(df_small[['Category']])
    mem_sparse = sparse.data.nbytes / 1024
    
    results.append({
        'Categories': n_cat,
        'Dense (KB)': mem_dense,
        'Sparse (KB)': mem_sparse,
        'Savings %': (1 - mem_sparse/mem_dense) * 100
    })

print(pd.DataFrame(results).to_string(index=False))`,
    },
    {
      language: "python-handle",
      label: "Handling Unknown Categories",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Create training and test data
train_data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium']
}
train_df = pd.DataFrame(train_data)

test_data = {
    'Color': ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Blue'],  # Yellow and Purple are new
    'Size': ['Small', 'Large', 'Medium', 'Small', 'Extra', 'Large']  # Extra is new
}
test_df = pd.DataFrame(test_data)

print("=" * 50)
print("1. TRAINING DATA")
print("=" * 50)
print(train_df)

print("\\n" + "=" * 50)
print("2. TEST DATA (WITH UNKNOWN CATEGORIES)")
print("=" * 50)
print(test_df)

print("\\n" + "=" * 50)
print("3. ONE-HOT ENCODING WITH handle_unknown='ignore'")
print("=" * 50)

# Create encoder with ignore option
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit on training data
encoder.fit(train_df[['Color', 'Size']])

# Transform training data
train_encoded = encoder.transform(train_df[['Color', 'Size']])
feature_names = encoder.get_feature_names_out(['Color', 'Size'])
train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names)

print("Training data encoded:")
print(train_encoded_df)

# Transform test data (unknown categories will be ignored)
test_encoded = encoder.transform(test_df[['Color', 'Size']])
test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names)

print("\\nTest data encoded:")
print(test_encoded_df)

print("\\n" + "=" * 50)
print("4. VERIFICATION")
print("=" * 50)
print(f"Training data shape: {train_encoded_df.shape}")
print(f"Test data shape: {test_encoded_df.shape}")
print("Unknown categories are automatically handled!")
print("Yellow → All 0s in Color columns")
print("Purple → All 0s in Color columns")
print("Extra → All 0s in Size columns")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is One-Hot Encoding?",
      options: [
        "Assigning integers to categories",
        "Creating binary columns for each category",
        "Using target means for encoding",
        "Assigning frequencies to categories"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding creates a binary (0/1) column for each category, with a 1 indicating the presence of that category.",
    },
    {
      id: 2,
      question: "What is the main advantage of One-Hot Encoding?",
      options: [
        "It's memory efficient",
        "It doesn't create false ordinal relationships",
        "It works with all algorithms",
        "It handles high cardinality well"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding avoids creating false ordinal relationships because each category gets its own binary column.",
    },
    {
      id: 3,
      question: "What is the main disadvantage of One-Hot Encoding?",
      options: [
        "It's too slow",
        "It creates too many columns for high-cardinality features",
        "It doesn't work with nominal data",
        "It causes data loss"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding creates one column per category, which can lead to high dimensionality with many categories.",
    },
    {
      id: 4,
      question: "What does drop_first=True do in pandas get_dummies?",
      options: [
        "Drops the first row of data",
        "Drops the first category to avoid multicollinearity",
        "Drops the first column",
        "Drops all categorical columns"
      ],
      correctAnswer: 1,
      explanation: "drop_first=True removes the first category column, which helps avoid multicollinearity (dummy variable trap).",
    },
    {
      id: 5,
      question: "What is the dummy variable trap?",
      options: [
        "When categories are lost",
        "When one-hot encoded columns are perfectly correlated",
        "When data becomes sparse",
        "When memory runs out"
      ],
      correctAnswer: 1,
      explanation: "The dummy variable trap occurs when one-hot encoded columns are perfectly correlated (one column can be predicted from others), causing multicollinearity issues.",
    },
    {
      id: 6,
      question: "What does handle_unknown='ignore' do in OneHotEncoder?",
      options: [
        "Removes unknown categories",
        "Sets all columns to 0 for unknown categories",
        "Raises an error",
        "Creates new columns"
      ],
      correctAnswer: 1,
      explanation: "handle_unknown='ignore' sets all columns to 0 for unknown categories, effectively ignoring them rather than throwing an error.",
    },
    {
      id: 7,
      question: "When should you use One-Hot Encoding?",
      options: [
        "For ordinal data",
        "For nominal data with few categories",
        "For high cardinality data",
        "For continuous data"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding is best for nominal data with few categories (typically < 20) where there is no natural order.",
    },
    {
      id: 8,
      question: "What is the difference between One-Hot Encoding and Label Encoding?",
      options: [
        "They are the same",
        "One-Hot creates binary columns, Label assigns integers",
        "Label creates binary columns, One-Hot assigns integers",
        "One-Hot is for ordinal data"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding creates binary columns for each category, while Label Encoding assigns a single integer to each category.",
    },
    {
      id: 9,
      question: "What is the advantage of using sparse matrix with OneHotEncoder?",
      options: [
        "It's faster to compute",
        "It saves memory for high-cardinality data",
        "It works with all algorithms",
        "It preserves the original data"
      ],
      correctAnswer: 1,
      explanation: "Sparse matrices save significant memory for high-cardinality data by only storing non-zero values.",
    },
    {
      id: 10,
      question: "Which encoding method should you use for high cardinality features?",
      options: [
        "One-Hot Encoding",
        "Label Encoding",
        "Binary Encoding",
        "Frequency Encoding"
      ],
      correctAnswer: 1,
      explanation: "For high cardinality features, Label Encoding or Frequency Encoding are preferred over One-Hot Encoding to avoid creating too many columns.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
    

        {/* 1. What is One-Hot Encoding? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <SquareSplitHorizontal className="h-5 w-5 text-primary" />
            1. What is One-Hot Encoding?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                One-Hot Encoding is a technique that <span className="font-semibold text-foreground">converts categorical variables into binary columns</span>, where each category becomes a separate column with values 0 or 1.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center gap-4 text-sm">
                  <div className="text-center">
                    <p className="font-mono">Red</p>
                    <p className="text-xs">[1, 0, 0]</p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono">Blue</p>
                    <p className="text-xs">[0, 1, 0]</p>
                  </div>
                  <div className="text-center">
                    <p className="font-mono">Green</p>
                    <p className="text-xs">[0, 0, 1]</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Each category gets its own binary column</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Characteristics</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Creates binary (0/1) columns</li>
                    <li>No false ordinal relationships</li>
                    <li>Works with all algorithms</li>
                    <li>Simple interpretation</li>
                    <li>Handles nominal data perfectly</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. When to Use One-Hot Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            2. When to Use One-Hot Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Best Use Cases
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Nominal Data</span> - No natural order</li>
                <li>• <span className="font-medium text-foreground">Low Cardinality</span> - Few categories (&lt; 20)</li>
                <li>• <span className="font-medium text-foreground">Linear Models</span> - SVM, Logistic Regression</li>
                <li>• <span className="font-medium text-foreground">Distance-based Models</span> - KNN, K-Means</li>
                <li>• <span className="font-medium text-foreground">When interpretability is important</span></li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                When to Avoid
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">High Cardinality</span> - Many categories</li>
                <li>• <span className="font-medium text-foreground">Memory Constraints</span> - Creates many columns</li>
                <li>• <span className="font-medium text-foreground">Ordinal Data</span> - Use Ordinal Encoding instead</li>
                <li>• <span className="font-medium text-foreground">Deep Learning</span> - May use embeddings instead</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3. How One-Hot Encoding Works */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            3. How One-Hot Encoding Works
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
                  <p className="font-mono text-sm">Create binary columns</p>
                  <p className="text-xs text-muted-foreground mt-1">Red, Blue, Green columns</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 3</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Assign 1/0 values</p>
                  <p className="text-xs text-muted-foreground mt-1">Red: [1,0,0]</p>
                  <p className="text-xs text-muted-foreground">Blue: [0,1,0]</p>
                  <p className="text-xs text-muted-foreground">Green: [0,0,1]</p>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-muted/50 p-3 rounded-lg">
              <p className="text-xs text-muted-foreground text-center">
                <span className="font-medium text-foreground">Note:</span> Each row has exactly one column with value 1, the rest are 0
              </p>
            </div>
          </div>
        </section>

        {/* 4. One-Hot Encoding in Pandas */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Table className="h-5 w-5 text-primary" />
            4. One-Hot Encoding in Pandas
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Basic Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Using get_dummies</p>
                <p className="font-mono text-xs text-primary mt-2">df_encoded = pd.get_dummies(df, columns=['color'])</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Key Parameters</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium">columns</span>: Which columns to encode</li>
                    <li><span className="font-medium">drop_first</span>: Avoid multicollinearity</li>
                    <li><span className="font-medium">prefix</span>: Column name prefix</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Drop First Category</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary"># Drop first to avoid multicollinearity</p>
                <p className="font-mono text-xs text-primary mt-2">df_encoded = pd.get_dummies(</p>
                <p className="font-mono text-xs text-primary ml-4">df, columns=['color'], drop_first=True</p>
                <p className="font-mono text-xs text-primary">)</p>
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                <p> Reduces dimensionality</p>
                <p> Avoids dummy variable trap</p>
              </div>
            </div>
          </div>
        </section>

        {/* 5. One-Hot Encoding in Scikit-learn */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Cog className="h-5 w-5 text-primary" />
            5. One-Hot Encoding in Scikit-learn
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Basic Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary">from sklearn.preprocessing import OneHotEncoder</p>
                <p className="font-mono text-xs text-primary mt-2">encoder = OneHotEncoder()</p>
                <p className="font-mono text-xs text-primary mt-1">encoded = encoder.fit_transform(df[['color']])</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Key Parameters</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium">sparse_output</span>: Return sparse matrix</li>
                    <li><span className="font-medium">handle_unknown</span>: Handle unseen categories</li>
                    <li><span className="font-medium">drop</span>: Drop one category</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Important Methods</h4>
              <div className="space-y-3">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">encoder.categories_</p>
                  <p className="text-xs text-muted-foreground">Shows all unique categories</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">encoder.get_feature_names_out()</p>
                  <p className="text-xs text-muted-foreground">Get column names</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">encoder.inverse_transform()</p>
                  <p className="text-xs text-muted-foreground">Convert back to original</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Advantages and Disadvantages */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            6. Advantages and Disadvantages
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Advantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">No False Order</span> - Doesn't create artificial ordinal relationships</li>
                <li>• <span className="font-medium text-foreground">Works with All Algorithms</span> - Universal compatibility</li>
                <li>• <span className="font-medium text-foreground">Interpretable</span> - Easy to understand and explain</li>
                <li>• <span className="font-medium text-foreground">Handles Nominal Data</span> - Perfect for categories without order</li>
                <li>• <span className="font-medium text-foreground">Simple Implementation</span> - Straightforward in pandas and sklearn</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                Disadvantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">High Dimensionality</span> - Creates many columns</li>
                <li>• <span className="font-medium text-foreground">Memory Intensive</span> - Especially with high cardinality</li>
                <li>• <span className="font-medium text-foreground">Multicollinearity</span> - Can cause dummy variable trap</li>
                <li>• <span className="font-medium text-foreground">Sparse Data</span> - Many zeros in the matrix</li>
                <li>• <span className="font-medium text-foreground">Not for Ordinal Data</span> - Loses order information</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 7. Handling High Cardinality */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Grid className="h-5 w-5 text-primary" />
            7. Handling High Cardinality
          </h2>

          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">The Problem</h4>
              <p className="text-sm text-muted-foreground">
                One-Hot Encoding creates one column per category, which can cause:
              </p>
              <div className="grid md:grid-cols-3 gap-3 mt-3">
                <div className="bg-destructive/10 p-3 rounded">
                  <p className="font-medium text-foreground text-sm">High Memory Usage</p>
                  <p className="text-xs text-muted-foreground">100 categories = 100 columns</p>
                </div>
                <div className="bg-destructive/10 p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Sparse Data</p>
                  <p className="text-xs text-muted-foreground">Mostly zeros, inefficient</p>
                </div>
                <div className="bg-destructive/10 p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Slow Training</p>
                  <p className="text-xs text-muted-foreground">More features = slower</p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Solutions</h4>
              <div className="grid md:grid-cols-3 gap-3">
                <div className="bg-card p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Use Sparse Matrix</p>
                  <p className="text-xs text-muted-foreground">sparse_output=True in sklearn</p>
                </div>
                <div className="bg-card p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Group Rare Categories</p>
                  <p className="text-xs text-muted-foreground">Combine infrequent categories</p>
                </div>
                <div className="bg-card p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Alternative Encoding</p>
                  <p className="text-xs text-muted-foreground">Target or Frequency Encoding</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. The Dummy Variable Trap */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            8. The Dummy Variable Trap
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">What is it?</h4>
                <p className="text-sm text-muted-foreground">
                  The dummy variable trap occurs when one-hot encoded columns become perfectly correlated, causing multicollinearity issues in linear models.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Example</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-xs">Red: [1, 0, 0]</p>
                  <p className="font-mono text-xs">Blue: [0, 1, 0]</p>
                  <p className="font-mono text-xs">Green: [0, 0, 1]</p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Red + Blue + Green = [1, 1, 1] (perfect correlation)
                  </p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Solution</h4>
                <div className="bg-green-50 p-3 rounded dark:bg-green-950">
                  <p className="text-sm text-muted-foreground">
                    Use <span className="font-medium text-foreground">drop_first=True</span> to remove one category column:
                  </p>
                  <div className="bg-muted p-2 rounded mt-2">
                    <p className="font-mono text-xs">df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Now only 2 columns are needed for 3 categories, avoiding perfect correlation.
                  </p>
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
                  <h4 className="font-semibold text-foreground mb-1">High Dimensionality</h4>
                  <p className="text-sm text-muted-foreground">
                    Using One-Hot Encoding on high-cardinality features creates too many columns.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Multicollinearity</h4>
                  <p className="text-sm text-muted-foreground">
                    Not dropping one category can cause the dummy variable trap.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Unseen Categories</h4>
                  <p className="text-sm text-muted-foreground">
                    Not handling unknown categories in test data can cause errors.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Memory Issues</h4>
                  <p className="text-sm text-muted-foreground">
                    Using dense matrix instead of sparse for large datasets.
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
                <h4 className="font-semibold text-foreground mb-1">Use drop_first</h4>
                <p className="text-sm text-muted-foreground">Always drop one category to avoid multicollinearity</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Sparse Matrix</h4>
                <p className="text-sm text-muted-foreground">Enable sparse_output=True for large datasets</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Handle Unknown Categories</h4>
                <p className="text-sm text-muted-foreground">Use handle_unknown='ignore' in sklearn</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Limit Cardinality</h4>
                <p className="text-sm text-muted-foreground">Group rare categories or use alternative encoding</p>
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
            See one-hot encoding in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="One-Hot Encoding Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}