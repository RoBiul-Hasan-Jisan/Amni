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
 
} from "lucide-react";

export default function TargetEncodingPage() {
  const result = getSubtopicBySlug("machine-learning", "target-encoding");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-basic",
      label: "Basic Target Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create sample data with relationship to target
np.random.seed(42)
n_samples = 1000

data = {
    'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
    'Product': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
    'Feature1': np.random.randn(n_samples),
    'Feature2': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

# Create target with relationship to categories
category_effects = {'A': 0.8, 'B': 0.5, 'C': 0.2, 'D': -0.3, 'E': -0.6}
product_effects = {'X': 0.4, 'Y': 0.1, 'Z': -0.2, 'W': -0.5}

df['Target_Continuous'] = (
    df['Category'].map(category_effects) + 
    df['Product'].map(product_effects) + 
    np.random.randn(n_samples) * 0.3
)
df['Target_Binary'] = (df['Target_Continuous'] > 0).astype(int)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print("First 5 rows:")
print(df.head())
print("\\nTarget statistics:")
print(df['Target_Continuous'].describe())

print("\\n" + "=" * 50)
print("2. TARGET ENCODING (MEAN ENCODING)")
print("=" * 50)

# Calculate mean target for each category
category_means = df.groupby('Category')['Target_Continuous'].mean()
product_means = df.groupby('Product')['Target_Continuous'].mean()

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
print(df[['Category', 'Category_TargetEncoded', 'Product', 'Product_TargetEncoded', 'Target_Continuous']].head(10))

print("\\n" + "=" * 50)
print("4. CORRELATION WITH TARGET")
print("=" * 50)
print("Category mean vs target:", np.corrcoef(df['Category_TargetEncoded'], df['Target_Continuous'])[0, 1])
print("Product mean vs target:", np.corrcoef(df['Product_TargetEncoded'], df['Target_Continuous'])[0, 1])`,
    },
    {
      language: "python-smoothing",
      label: "Target Encoding with Smoothing",
      code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Create data with small categories
np.random.seed(42)
n_samples = 500

categories = ['A']*400 + ['B']*50 + ['C']*30 + ['D']*15 + ['E']*5
data = {
    'Category': categories,
    'Feature': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

# Create target
category_effects = {'A': 0.3, 'B': 0.1, 'C': -0.2, 'D': 0.5, 'E': -0.8}
df['Target'] = df['Category'].map(category_effects) + np.random.randn(n_samples) * 0.5

print("=" * 50)
print("1. DATA WITH SMALL CATEGORIES")
print("=" * 50)
print("Category distribution:")
print(df['Category'].value_counts())

print("\\n" + "=" * 50)
print("2. TARGET ENCODING WITHOUT SMOOTHING")
print("=" * 50)

# Calculate means
category_means = df.groupby('Category')['Target'].mean()
df['Category_Mean'] = df['Category'].map(category_means)

print("Category means:")
print(category_means)

print("\\n" + "=" * 50)
print("3. TARGET ENCODING WITH SMOOTHING")
print("=" * 50)

# Calculate global mean and smoothing
global_mean = df['Target'].mean()
smoothing_factor = 10

def smooth_target_encoding(df, col, target, global_mean, smoothing):
    # Calculate mean and count for each category
    stats = df.groupby(col)[target].agg(['mean', 'count'])
    
    # Apply smoothing
    stats['smoothed'] = (
        stats['mean'] * stats['count'] + global_mean * smoothing
    ) / (stats['count'] + smoothing)
    
    return stats['smoothed']

smoothed_means = smooth_target_encoding(df, 'Category', 'Target', global_mean, smoothing_factor)
df['Category_Smoothed'] = df['Category'].map(smoothed_means)

print("Comparison of methods:")
print("\\nCategory | Raw Mean | Smoothed Mean | Global Mean")
print("-" * 50)
for category in df['Category'].unique():
    raw = category_means[category]
    smoothed = smoothed_means[category]
    print(f"{category:8} | {raw:8.3f} | {smoothed:12.3f} | {global_mean:8.3f}")

print("\\n" + "=" * 50)
print("4. EFFECT OF SMOOTHING ON SMALL CATEGORIES")
print("=" * 50)

# Show how smoothing affects small categories
print("Category E has only 5 samples, so it's heavily smoothed toward global mean")
print(f"Category E raw mean: {category_means['E']:.3f}")
print(f"Category E smoothed: {smoothed_means['E']:.3f}")
print(f"Global mean: {global_mean:.3f}")`,
    },
    {
      language: "python-crossval",
      label: "K-Fold Target Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create data
np.random.seed(42)
n_samples = 1000

data = {
    'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
    'Feature': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

category_effects = {'A': 0.6, 'B': 0.3, 'C': 0.0, 'D': -0.3, 'E': -0.6}
df['Target'] = df['Category'].map(category_effects) + np.random.randn(n_samples) * 0.5

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(f"Total samples: {len(df)}")
print("\\nCategory distribution:")
print(df['Category'].value_counts())

print("\\n" + "=" * 50)
print("2. K-FOLD TARGET ENCODING (PREVENTING LEAKAGE)")
print("=" * 50)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create encoded column in training data
train_df['Category_Encoded_CV'] = np.nan

for train_idx, val_idx in kf.split(train_df):
    # Training fold data
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]
    
    # Calculate encoding on training fold
    fold_means = train_fold.groupby('Category')['Target'].mean()
    
    # Apply to validation fold
    train_df.loc[val_idx, 'Category_Encoded_CV'] = val_fold['Category'].map(fold_means)

# For test data, use the mean from all training data
global_train_mean = train_df['Target'].mean()
test_df['Category_Encoded_CV'] = test_df['Category'].map(
    train_df.groupby('Category')['Target'].mean()
).fillna(global_train_mean)

print("Training data with cross-validated encoding:")
print(train_df[['Category', 'Target', 'Category_Encoded_CV']].head(10))

print("\\n" + "=" * 50)
print("3. COMPARISON WITH NAIVE ENCODING")
print("=" * 50)

# Naive encoding (using whole training data)
naive_means = train_df.groupby('Category')['Target'].mean()
train_df['Category_Encoded_Naive'] = train_df['Category'].map(naive_means)
test_df['Category_Encoded_Naive'] = test_df['Category'].map(naive_means).fillna(global_train_mean)

print("Correlation with target in training (naive):", 
      np.corrcoef(train_df['Category_Encoded_Naive'], train_df['Target'])[0, 1])
print("Correlation with target in test (naive):", 
      np.corrcoef(test_df['Category_Encoded_Naive'], test_df['Target'])[0, 1])

print("\\nCorrelation with target in training (CV):", 
      np.corrcoef(train_df['Category_Encoded_CV'], train_df['Target'])[0, 1])
print("Correlation with target in test (CV):", 
      np.corrcoef(test_df['Category_Encoded_CV'], test_df['Target'])[0, 1])

print("\\n" + "=" * 50)
print("4. WHY CROSS-VALIDATION IS IMPORTANT")
print("=" * 50)
print("Naive encoding uses target information from the entire dataset")
print("This creates a strong correlation in training (can overfit)")
print("CV encoding uses only out-of-fold data, preventing leakage")`,
    },
    {
      language: "python-binary",
      label: "Target Encoding for Binary Classification",
      code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create data
np.random.seed(42)
n_samples = 1000

data = {
    'Category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
    'Product': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
    'Feature': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

# Create binary target
category_probs = {'A': 0.9, 'B': 0.7, 'C': 0.5, 'D': 0.3, 'E': 0.1}
product_probs = {'X': 0.8, 'Y': 0.6, 'Z': 0.4, 'W': 0.2}

df['Target'] = 0
for idx, row in df.iterrows():
    prob = category_probs[row['Category']] * 0.7 + product_probs[row['Product']] * 0.3
    df.loc[idx, 'Target'] = 1 if np.random.random() < prob else 0

print("=" * 50)
print("1. BINARY TARGET DATA")
print("=" * 50)
print("Target distribution:")
print(df['Target'].value_counts())
print(f"Positive rate: {df['Target'].mean():.3f}")

print("\\n" + "=" * 50)
print("2. TARGET ENCODING (PROBABILITY ENCODING)")
print("=" * 50)

# Calculate probability of positive class for each category
category_probs_encoded = df.groupby('Category')['Target'].mean()
product_probs_encoded = df.groupby('Product')['Target'].mean()

print("\\nCategory probabilities:")
print(category_probs_encoded)

print("\\nProduct probabilities:")
print(product_probs_encoded)

# Apply encoding
df['Category_TargetEncoded'] = df['Category'].map(category_probs_encoded)
df['Product_TargetEncoded'] = df['Product'].map(product_probs_encoded)

print("\\n" + "=" * 50)
print("3. ENCODED DATA")
print("=" * 50)
print(df[['Category', 'Category_TargetEncoded', 'Product', 'Product_TargetEncoded', 'Target']].head(10))

print("\\n" + "=" * 50)
print("4. COMPARISON")
print("=" * 50)

# Split data for validation
X = df[['Category_TargetEncoded', 'Product_TargetEncoded']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple threshold classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.3f}")

print("\\n" + "=" * 50)
print("5. HANDLING UNSEEN CATEGORIES")
print("=" * 50)
print("If a new category appears in test data, use global mean")
global_mean = df['Target'].mean()
print(f"Global mean (fallback): {global_mean:.3f}")`,
    },
    {
      language: "python-comparison",
      label: "Comparison with Other Encodings",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Create data
np.random.seed(42)
n_samples = 2000
n_categories = 20

data = {
    'Category': np.random.choice([f'Cat_{i}' for i in range(n_categories)], n_samples),
    'Feature': np.random.randn(n_samples)
}
df = pd.DataFrame(data)

# Create target with relationship to categories
category_effects = {f'Cat_{i}': np.random.randn() * 0.5 for i in range(n_categories)}
df['Target'] = df['Category'].map(category_effects) + np.random.randn(n_samples) * 0.3

print("=" * 50)
print("ENCODING METHODS COMPARISON")
print("=" * 50)

# Split data
X = df[['Category']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# 1. Label Encoding
print("\\n1. Label Encoding:")
start = time.time()
le = LabelEncoder()
X_train_le = le.fit_transform(X_train['Category'])
X_test_le = le.transform(X_test['Category'])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_le.reshape(-1, 1), y_train)
y_pred = model.predict(X_test_le.reshape(-1, 1))
r2 = r2_score(y_test, y_pred)
results['Label Encoding'] = {'R2': r2, 'Time': time.time() - start}
print(f"  R2 Score: {r2:.3f}")
print(f"  Time: {time.time() - start:.3f}s")

# 2. One-Hot Encoding
print("\\n2. One-Hot Encoding:")
start = time.time()
ohe = OneHotEncoder(sparse_output=False)
X_train_ohe = ohe.fit_transform(X_train[['Category']])
X_test_ohe = ohe.transform(X_test[['Category']])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_ohe, y_train)
y_pred = model.predict(X_test_ohe)
r2 = r2_score(y_test, y_pred)
results['One-Hot Encoding'] = {'R2': r2, 'Time': time.time() - start}
print(f"  R2 Score: {r2:.3f}")
print(f"  Time: {time.time() - start:.3f}s")

# 3. Target Encoding
print("\\n3. Target Encoding:")
start = time.time()
# Get target means
target_means = df.groupby('Category')['Target'].mean()
X_train_target = X_train['Category'].map(target_means)
X_test_target = X_test['Category'].map(target_means)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_target.values.reshape(-1, 1), y_train)
y_pred = model.predict(X_test_target.values.reshape(-1, 1))
r2 = r2_score(y_test, y_pred)
results['Target Encoding'] = {'R2': r2, 'Time': time.time() - start}
print(f"  R2 Score: {r2:.3f}")
print(f"  Time: {time.time() - start:.3f}s")

print("\\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(pd.DataFrame(results).T)`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Target Encoding?",
      options: [
        "Encoding based on category frequency",
        "Replacing categories with target statistics (e.g., mean)",
        "Creating binary columns for categories",
        "Assigning random integers to categories"
      ],
      correctAnswer: 1,
      explanation: "Target Encoding replaces categories with the mean (or other statistic) of the target variable for that category.",
    },
    {
      id: 2,
      question: "What is the main advantage of Target Encoding?",
      options: [
        "It's memory efficient",
        "It captures category-target relationship",
        "It works with all algorithms",
        "It's simple to implement"
      ],
      correctAnswer: 1,
      explanation: "Target Encoding captures the relationship between categories and the target variable, which can improve model performance.",
    },
    {
      id: 3,
      question: "What is the main risk of Target Encoding?",
      options: [
        "It's too slow",
        "It can cause overfitting",
        "It doesn't work with binary targets",
        "It loses category information"
      ],
      correctAnswer: 1,
      explanation: "Target Encoding can cause overfitting because it uses information from the target variable. Use cross-validation or smoothing to mitigate.",
    },
    {
      id: 4,
      question: "What is smoothing in Target Encoding?",
      options: [
        "Removing categories",
        "Combining category mean with global mean to reduce overfitting",
        "Scaling the encoded values",
        "Dropping rare categories"
      ],
      correctAnswer: 1,
      explanation: "Smoothing combines the category mean with the global mean, especially useful for small categories to prevent overfitting.",
    },
    {
      id: 5,
      question: "What is the formula for smoothed target encoding?",
      options: [
        "(category_mean + global_mean) / 2",
        "(category_mean * count + global_mean * smoothing) / (count + smoothing)",
        "category_mean * count / (count + smoothing)",
        "category_mean + global_mean * smoothing"
      ],
      correctAnswer: 1,
      explanation: "The smoothed encoding balances the category mean with the global mean based on the category count and smoothing factor.",
    },
    {
      id: 6,
      question: "How can you prevent data leakage in Target Encoding?",
      options: [
        "Use the entire dataset",
        "Use K-Fold cross-validation",
        "Remove target variable",
        "Use only training data"
      ],
      correctAnswer: 1,
      explanation: "K-Fold cross-validation computes encodings using only out-of-fold data, preventing leakage from training to validation.",
    },
    {
      id: 7,
      question: "When should you use Target Encoding?",
      options: [
        "For nominal data with few categories",
        "For high cardinality categorical features",
        "For ordinal data",
        "For binary data only"
      ],
      correctAnswer: 1,
      explanation: "Target Encoding is particularly useful for high-cardinality features where One-Hot Encoding would create too many columns.",
    },
    {
      id: 8,
      question: "What statistic is commonly used for Target Encoding?",
      options: [
        "Median",
        "Mean (for regression) or Probability (for classification)",
        "Mode",
        "Standard Deviation"
      ],
      correctAnswer: 1,
      explanation: "For regression, the mean target is used. For classification, the probability of the positive class is used.",
    },
    {
      id: 9,
      question: "How do you handle unseen categories in Target Encoding?",
      options: [
        "Raise an error",
        "Use the global mean as fallback",
        "Create a new category",
        "Drop the row"
      ],
      correctAnswer: 1,
      explanation: "When an unseen category appears in test data, use the global mean as a fallback value.",
    },
    {
      id: 10,
      question: "What is the advantage of Target Encoding over One-Hot Encoding for high cardinality features?",
      options: [
        "It creates more columns",
        "It creates fewer columns (just one)",
        "It's more accurate",
        "It's faster to compute"
      ],
      correctAnswer: 1,
      explanation: "Target Encoding creates only one column regardless of cardinality, while One-Hot Encoding creates one column per category.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
     

        {/* 1. What is Target Encoding? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Crosshair className="h-5 w-5 text-primary" />
            1. What is Target Encoding?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Target Encoding (also known as <span className="font-semibold text-foreground">Mean Encoding</span>) is a technique that <span className="font-semibold text-foreground">replaces categorical values with statistics</span> calculated from the target variable.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">Category A (mean target=0.8) → 0.8</p>
                <p className="font-mono text-sm">Category B (mean target=0.5) → 0.5</p>
                <p className="text-xs text-muted-foreground mt-2">Each category becomes its target mean</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Characteristics</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Uses target information</li>
                    <li>Single column output</li>
                    <li>Captures category-target relationship</li>
                    <li>Handles high cardinality</li>
                    <li>Improves model performance</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. When to Use Target Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            2. When to Use Target Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Best Use Cases
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">High Cardinality</span> - Many categories</li>
                <li>• <span className="font-medium text-foreground">Nominal Data</span> - Categories with no order</li>
                <li>• <span className="font-medium text-foreground">Tree-based Models</span> - Random Forest, XGBoost</li>
                <li>• <span className="font-medium text-foreground">When relationships matter</span></li>
                <li>• <span className="font-medium text-foreground">Memory Constraints</span> - Single column</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                When to Avoid
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Small Datasets</span> - Risk of overfitting</li>
                <li>• <span className="font-medium text-foreground">Linear Models</span> - Can cause multicollinearity</li>
                <li>• <span className="font-medium text-foreground">When target is noisy</span></li>
                <li>• <span className="font-medium text-foreground">Interpretability needed</span></li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3. How Target Encoding Works */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            3. How Target Encoding Works
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-4 gap-4">
              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 1</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Group by category</p>
                  <p className="text-xs text-muted-foreground mt-1">Category A → [0.9, 0.7, 0.8]</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 2</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Calculate statistics</p>
                  <p className="text-xs text-muted-foreground mt-1">Mean = 0.8</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 3</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Apply smoothing</p>
                  <p className="text-xs text-muted-foreground mt-1">Combine with global mean</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 4</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Transform</p>
                  <p className="text-xs text-muted-foreground mt-1">Category A → 0.8</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Smoothing */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            4. Smoothing (Regularization)
          </h2>

          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Why Smoothing?</h4>
              <p className="text-sm text-muted-foreground">
                Categories with few samples can have extreme means, leading to overfitting. Smoothing pulls them toward the global mean.
              </p>
              <div className="bg-muted p-3 rounded-lg mt-3">
                <p className="font-mono text-sm">Smoothed = (mean × count + global_mean × smoothing) / (count + smoothing)</p>
                <p className="text-xs text-muted-foreground mt-1">Where smoothing controls the strength of regularization</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-destructive/10 p-3 rounded">
                <p className="font-medium text-foreground text-sm">Without Smoothing</p>
                <p className="text-xs text-muted-foreground">
                  Category E (only 2 samples, mean=0.9) → 0.9
                </p>
                <p className="text-xs text-destructive mt-1">⚠️ Overfitting risk</p>
              </div>

              <div className="bg-green-50 p-3 rounded dark:bg-green-950">
                <p className="font-medium text-foreground text-sm">With Smoothing</p>
                <p className="text-xs text-muted-foreground">
                  Category E (2 samples, mean=0.9) → 0.7 (smoothed)
                </p>
                <p className="text-xs text-green-500 mt-1"> Reduced overfitting</p>
              </div>
            </div>
          </div>
        </section>

        {/* 5. K-Fold Target Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            5. K-Fold Target Encoding
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">The Problem</h4>
                <p className="text-sm text-muted-foreground">
                  Using the entire dataset to compute target means causes data leakage. K-Fold prevents this.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">The Solution</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-xs">For each fold in K-Fold:</p>
                  <p className="font-mono text-xs ml-4">1. Compute means on training folds</p>
                  <p className="font-mono text-xs ml-4">2. Apply to validation fold</p>
                  <p className="font-mono text-xs mt-1">3. For test data, use mean of all training data</p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-3">
                <div className="bg-destructive/10 p-3 rounded">
                  <p className="font-medium text-foreground text-sm"> Naive Encoding</p>
                  <p className="text-xs text-muted-foreground">
                    Uses all data → Data leakage → Overfitting
                  </p>
                </div>
                <div className="bg-green-50 p-3 rounded dark:bg-green-950">
                  <p className="font-medium text-foreground text-sm"> K-Fold Encoding</p>
                  <p className="text-xs text-muted-foreground">
                    Uses out-of-fold data → No leakage → Better generalization
                  </p>
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
                <li>• <span className="font-medium text-foreground">Captures Relationships</span> - Uses target information</li>
                <li>• <span className="font-medium text-foreground">Memory Efficient</span> - Single column</li>
                <li>• <span className="font-medium text-foreground">Handles High Cardinality</span> - Works with many categories</li>
                <li>• <span className="font-medium text-foreground">Improves Performance</span> - Often boosts model accuracy</li>
                <li>• <span className="font-medium text-foreground">Flexible</span> - Can use different statistics</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                Disadvantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Risk of Overfitting</span> - Needs regularization</li>
                <li>• <span className="font-medium text-foreground">Data Leakage</span> - Must use cross-validation</li>
                <li>• <span className="font-medium text-foreground">Not for Linear Models</span> - Can cause issues</li>
                <li>• <span className="font-medium text-foreground">Requires Target</span> - Only for supervised learning</li>
                <li>• <span className="font-medium text-foreground">Less Interpretable</span> - Harder to explain</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 7. Target Encoding vs Other Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            7. Target Encoding vs Other Methods
          </h2>

          <div className="bg-card border border-border rounded-lg p-4 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-2">Method</th>
                  <th className="text-left p-2">Output</th>
                  <th className="text-left p-2">Cardinality</th>
                  <th className="text-left p-2">Leakage Risk</th>
                  <th className="text-left p-2">Memory</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Target Encoding</td>
                  <td className="text-muted-foreground">Single numeric</td>
                  <td className="text-muted-foreground">Any</td>
                  <td className="text-muted-foreground">High (use CV)</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">One-Hot Encoding</td>
                  <td className="text-muted-foreground">Multiple binary</td>
                  <td className="text-muted-foreground">Low (&lt;20)</td>
                  <td className="text-muted-foreground">None</td>
                  <td className="text-muted-foreground">High</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-2 font-medium text-primary">Label Encoding</td>
                  <td className="text-muted-foreground">Single integer</td>
                  <td className="text-muted-foreground">Any</td>
                  <td className="text-muted-foreground">None</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
                <tr>
                  <td className="p-2 font-medium text-primary">Frequency Encoding</td>
                  <td className="text-muted-foreground">Single numeric</td>
                  <td className="text-muted-foreground">Any</td>
                  <td className="text-muted-foreground">None</td>
                  <td className="text-muted-foreground">Low</td>
                </tr>
              </tbody>
            </table>
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
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Using target means from the entire dataset. Always use K-Fold cross-validation.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Overfitting</h4>
                  <p className="text-sm text-muted-foreground">
                    Not using smoothing for small categories. Apply smoothing to regularize.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Unseen Categories</h4>
                  <p className="text-sm text-muted-foreground">
                    New categories in test data. Use global mean as fallback.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Target Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Encoding before train-test split. Always split first, then encode.
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
                <h4 className="font-semibold text-foreground mb-1">Always Use K-Fold</h4>
                <p className="text-sm text-muted-foreground">Prevent data leakage by encoding only on training folds</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Apply Smoothing</h4>
                <p className="text-sm text-muted-foreground">Regularize small categories to prevent overfitting</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Handle Unseen Categories</h4>
                <p className="text-sm text-muted-foreground">Use global mean as fallback for new categories</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Split Before Encoding</h4>
                <p className="text-sm text-muted-foreground">Always split data into train/test before any encoding</p>
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
            See target encoding in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Target Encoding Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}