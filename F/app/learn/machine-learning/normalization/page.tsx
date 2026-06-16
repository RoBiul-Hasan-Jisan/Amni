import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { 
  AlertCircle, 
  CheckCircle2, 
  Lightbulb,
  Brain,
  Target,
  Layers,
  GitBranch,
  Activity,
  Shield,
  Sparkles,
  Database,
  Cpu,
  LineChart,
  GraduationCap,
  Network,
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
  AlignRight
} from "lucide-react";

export default function NormalizationPage() {
  const result = getSubtopicBySlug("machine-learning", "normalization");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-minmax",
      label: "Min-Max Normalization",
      code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Create sample data with different scales
np.random.seed(42)
n_samples = 300

data = {
    'Age': np.random.randint(18, 80, n_samples),
    'Salary': np.random.normal(50000, 15000, n_samples),
    'Experience': np.random.randint(0, 35, n_samples),
    'Score': np.random.uniform(0, 100, n_samples)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. BEFORE MIN-MAX NORMALIZATION")
print("=" * 50)
print("First 5 rows:")
print(df.head())
print("\\nStatistics:")
print(df.describe())

# Split data
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Apply Min-Max Normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrame with scaled data
df_train_scaled = pd.DataFrame(X_train_scaled, columns=df.columns)
df_test_scaled = pd.DataFrame(X_test_scaled, columns=df.columns)

print("\\n" + "=" * 50)
print("2. AFTER MIN-MAX NORMALIZATION (TRAIN SET)")
print("=" * 50)
print("First 5 rows:")
print(df_train_scaled.head())
print("\\nStatistics:")
print(df_train_scaled.describe())

print("\\n" + "=" * 50)
print("3. SCALER PARAMETERS")
print("=" * 50)
print("Min values (training):", scaler.data_min_)
print("Max values (training):", scaler.data_max_)
print("Scale:", scaler.scale_)

# Visualize comparison
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, col in enumerate(df.columns):
    # Original
    axes[0, i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
    axes[0, i].set_title(f'Original: {col}')
    axes[0, i].axvline(df[col].min(), color='red', linestyle='--', label='Min')
    axes[0, i].axvline(df[col].max(), color='green', linestyle='--', label='Max')
    axes[0, i].legend()
    
    # Normalized
    axes[1, i].hist(df_train_scaled[col], bins=30, alpha=0.7, edgecolor='black')
    axes[1, i].set_title(f'Normalized: {col}')
    axes[1, i].axvline(0, color='red', linestyle='--', label='Min=0')
    axes[1, i].axvline(1, color='green', linestyle='--', label='Max=1')
    axes[1, i].legend()

plt.tight_layout()
plt.savefig('minmax_normalization.png', dpi=300, bbox_inches='tight')
print("\\nVisualization saved as 'minmax_normalization.png'")`,
    },
    {
      language: "python-custom",
      label: "Custom Range Normalization",
      code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create sample data
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
df = pd.DataFrame(data, columns=['Value'])

print("Original data:", data)

# Normalize to different ranges
scalers = {
    'Default [0,1]': MinMaxScaler(),
    'Range [-1,1]': MinMaxScaler(feature_range=(-1, 1)),
    'Range [0,255]': MinMaxScaler(feature_range=(0, 255)),
    'Range [-100,100]': MinMaxScaler(feature_range=(-100, 100))
}

print("\\n" + "=" * 50)
print("NORMALIZATION WITH DIFFERENT RANGES")
print("=" * 50)

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(df).flatten()
    print(f"\\n{name}:")
    print(f"  Data: {scaled}")
    print(f"  Min: {scaled.min():.2f}")
    print(f"  Max: {scaled.max():.2f}")
    print(f"  Mean: {scaled.mean():.2f}")

# Example with negative values
data_negative = np.array([-50, -30, -10, 10, 30, 50])
df_neg = pd.DataFrame(data_negative, columns=['Value'])

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_neg = scaler.fit_transform(df_neg).flatten()

print("\\n" + "=" * 50)
print("NORMALIZATION WITH NEGATIVE VALUES")
print("=" * 50)
print(f"Original: {data_negative}")
print(f"Normalized: {scaled_neg}")
print(f"Min: {scaled_neg.min():.2f}")
print(f"Max: {scaled_neg.max():.2f}")`,
    },
    {
      language: "python-pipeline",
      label: "Normalization with Pipeline",
      code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data with different scales
np.random.seed(42)
n_samples = 500

X = np.random.randn(n_samples, 3)
X[:, 0] = X[:, 0] * 10 + 50  # Scale 1
X[:, 1] = X[:, 1] * 100 + 500  # Scale 2
X[:, 2] = X[:, 2] * 1000 + 5000  # Scale 3

y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline with normalization and model
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

print("=" * 50)
print("PIPELINE WITH MIN-MAX NORMALIZATION")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Access the fitted scaler
scaler = pipeline.named_steps['scaler']
print("\\nScaler Parameters:")
print(f"Min values: {scaler.data_min_}")
print(f"Max values: {scaler.data_max_}")
print(f"Scale: {scaler.scale_}")

# Transform new data
new_data = np.array([[60, 600, 6000]])
scaled_new = pipeline.named_steps['scaler'].transform(new_data)
print(f"\\nNew data (scaled): {scaled_new[0]}")`,
    },
    {
      language: "python-vector",
      label: "Vector Normalization",
      code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
data = np.random.randn(100, 3) * 10
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print("First 5 rows:")
print(df.head())
print("\\nStatistics:")
print(df.describe())

# L2 Normalization (Euclidean norm)
df_l2 = pd.DataFrame(
    normalize(data, norm='l2'),
    columns=df.columns
)

# L1 Normalization (Manhattan norm)
df_l1 = pd.DataFrame(
    normalize(data, norm='l1'),
    columns=df.columns
)

# Max normalization
df_max = pd.DataFrame(
    normalize(data, norm='max'),
    columns=df.columns
)

print("\\n" + "=" * 50)
print("2. L2 NORMALIZATION (Euclidean)")
print("=" * 50)
print("First 5 rows:")
print(df_l2.head())
print("\\nRow norms (should be 1):")
print(np.linalg.norm(df_l2.values, axis=1)[:5])

print("\\n" + "=" * 50)
print("3. L1 NORMALIZATION (Manhattan)")
print("=" * 50)
print("First 5 rows:")
print(df_l1.head())
print("\\nRow L1 norms (should be 1):")
print(np.sum(np.abs(df_l1.values), axis=1)[:5])

print("\\n" + "=" * 50)
print("4. MAX NORMALIZATION")
print("=" * 50)
print("First 5 rows:")
print(df_max.head())

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original
axes[0, 0].scatter(df['Feature1'], df['Feature2'], alpha=0.5)
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('Feature1')
axes[0, 0].set_ylabel('Feature2')

# L2 Normalized
axes[0, 1].scatter(df_l2['Feature1'], df_l2['Feature2'], alpha=0.5)
axes[0, 1].set_title('L2 Normalized')
axes[0, 1].set_xlabel('Feature1')
axes[0, 1].set_ylabel('Feature2')

# L1 Normalized
axes[1, 0].scatter(df_l1['Feature1'], df_l1['Feature2'], alpha=0.5)
axes[1, 0].set_title('L1 Normalized')
axes[1, 0].set_xlabel('Feature1')
axes[1, 0].set_ylabel('Feature2')

# Max Normalized
axes[1, 1].scatter(df_max['Feature1'], df_max['Feature2'], alpha=0.5)
axes[1, 1].set_title('Max Normalized')
axes[1, 1].set_xlabel('Feature1')
axes[1, 1].set_ylabel('Feature2')

plt.tight_layout()
plt.savefig('vector_normalization.png', dpi=300, bbox_inches='tight')
print("\\nVisualization saved as 'vector_normalization.png'")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Normalization in machine learning?",
      options: [
        "Transforming data to have mean 0 and standard deviation 1",
        "Scaling data to a specific range, typically [0,1]",
        "Removing outliers from data",
        "Converting categorical data to numerical"
      ],
      correctAnswer: 1,
      explanation: "Normalization scales data to a specific range, usually [0,1], using the formula (x - min) / (max - min).",
    },
    {
      id: 2,
      question: "What is the formula for Min-Max normalization?",
      options: [
        "x' = (x - mean) / std",
        "x' = (x - min) / (max - min)",
        "x' = x / max_absolute_value",
        "x' = log(x)"
      ],
      correctAnswer: 1,
      explanation: "Min-Max normalization uses the formula x' = (x - min) / (max - min) to scale values to the range [0,1].",
    },
    {
      id: 3,
      question: "What is the default range of values after Min-Max normalization?",
      options: [
        "[-1, 1]",
        "[0, 1]",
        "[-∞, ∞]",
        "[0, ∞]"
      ],
      correctAnswer: 1,
      explanation: "Min-Max normalization scales data to the range [0,1] by default, though you can specify other ranges.",
    },
    {
      id: 4,
      question: "What is the difference between normalization and standardization?",
      options: [
        "They are the same thing",
        "Normalization scales to [0,1], Standardization gives mean=0, std=1",
        "Standardization scales to [0,1], Normalization gives mean=0, std=1",
        "Normalization is for categorical data"
      ],
      correctAnswer: 1,
      explanation: "Normalization scales data to a specific range (usually [0,1]), while standardization centers data around 0 with unit variance.",
    },
    {
      id: 5,
      question: "When should you use Min-Max normalization?",
      options: [
        "When data has outliers",
        "When you want to preserve the shape of the distribution",
        "When data needs to be in a bounded range",
        "When data is categorical"
      ],
      correctAnswer: 2,
      explanation: "Min-Max normalization is used when you need data in a bounded range, such as [0,1] for neural networks or image processing.",
    },
    {
      id: 6,
      question: "What is the disadvantage of Min-Max normalization?",
      options: [
        "It's computationally expensive",
        "It's sensitive to outliers",
        "It doesn't work with numerical data",
        "It changes the data distribution"
      ],
      correctAnswer: 1,
      explanation: "Min-Max normalization is sensitive to outliers because extreme values can compress the normal range of data.",
    },
    {
      id: 7,
      question: "What is L2 normalization?",
      options: [
        "Scaling to [0,1]",
        "Scaling each row to unit Euclidean norm",
        "Scaling to [-1,1]",
        "Scaling to have mean 0"
      ],
      correctAnswer: 1,
      explanation: "L2 normalization scales each sample/row to have a Euclidean norm (L2 norm) of 1.",
    },
    {
      id: 8,
      question: "What is the difference between L1 and L2 normalization?",
      options: [
        "L1 uses Manhattan norm, L2 uses Euclidean norm",
        "L1 is for classification, L2 is for regression",
        "They are the same",
        "L1 is for features, L2 is for samples"
      ],
      correctAnswer: 0,
      explanation: "L1 normalization scales to unit Manhattan (L1) norm, while L2 normalization scales to unit Euclidean (L2) norm.",
    },
    {
      id: 9,
      question: "What is MaxAbs scaling?",
      options: [
        "Scaling to [0,1]",
        "Scaling by dividing by the maximum absolute value",
        "Scaling to have mean 0",
        "Scaling to unit norm"
      ],
      correctAnswer: 1,
      explanation: "MaxAbs scaling divides each value by the maximum absolute value in the feature, resulting in a range of [-1,1].",
    },
    {
      id: 10,
      question: "When should you use MaxAbs scaling?",
      options: [
        "When data has many outliers",
        "When working with sparse data",
        "When data is categorical",
        "When data is normally distributed"
      ],
      correctAnswer: 1,
      explanation: "MaxAbs scaling is particularly useful for sparse data because it preserves the sparsity (zeros remain zeros).",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
     

        {/* 1. What is Normalization? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlignCenter className="h-5 w-5 text-primary" />
            1. What is Normalization?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Normalization is a scaling technique that <span className="font-semibold text-foreground">transforms data to a specific range</span>, typically [0, 1]. It's a crucial preprocessing step for many machine learning algorithms.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Raw Data</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">Normalization</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">[0, 1] Range</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Bringing features to the same scale</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">When to Use Normalization</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                    <li>Data needs to be in a bounded range</li>
                    <li>Neural networks with sigmoid/tanh activation</li>
                    <li>Image pixel values (0-255 → 0-1)</li>
                    <li>Distance-based algorithms (KNN, K-Means)</li>
                    <li>Gradient descent optimization</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Min-Max Normalization Formula */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Minimize className="h-5 w-5 text-primary" />
            2. Min-Max Normalization Formula
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3 text-center">Formula</h4>
              <div className="bg-muted p-4 rounded-lg text-center">
                <p className="font-mono text-xl">x' = (x - min) / (max - min)</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Scales to range [0, 1]
                </p>
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between border-b border-border pb-2">
                  <span className="text-sm text-muted-foreground">Min value after normalization</span>
                  <span className="font-mono font-bold text-primary">0</span>
                </div>
                <div className="flex items-center justify-between border-b border-border pb-2">
                  <span className="text-sm text-muted-foreground">Max value after normalization</span>
                  <span className="font-mono font-bold text-primary">1</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Range</span>
                  <span className="font-mono font-bold text-primary">[0, 1]</span>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3">Key Properties</h4>
              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <div className="bg-primary/10 rounded-full p-1 mt-0.5">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">Bounded Range</p>
                    <p className="text-xs text-muted-foreground">All values are within [0, 1]</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="bg-primary/10 rounded-full p-1 mt-0.5">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">Preserves Distribution Shape</p>
                    <p className="text-xs text-muted-foreground">The relative relationships between values are maintained</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="bg-primary/10 rounded-full p-1 mt-0.5">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">Outlier Sensitivity</p>
                    <p className="text-xs text-muted-foreground">Extreme values can compress the normal range</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Implementation Steps */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            3. Implementation Steps
          </h2>

          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">1</div>
                  <div className="w-px h-8 bg-border"></div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Split Data</h4>
                  <p className="text-sm text-muted-foreground">Split your data into training and test sets before normalization</p>
                  <div className="mt-2 bg-muted p-2 rounded text-xs font-mono text-primary">X_train, X_test = train_test_split(X, test_size=0.2)</div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">2</div>
                  <div className="w-px h-8 bg-border"></div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Create Scaler</h4>
                  <p className="text-sm text-muted-foreground">Initialize the MinMaxScaler from sklearn</p>
                  <div className="mt-2 bg-muted p-2 rounded text-xs font-mono text-primary">scaler = MinMaxScaler()</div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">3</div>
                  <div className="w-px h-8 bg-border"></div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Fit on Training Data</h4>
                  <p className="text-sm text-muted-foreground">Learn min and max from training data</p>
                  <div className="mt-2 bg-muted p-2 rounded text-xs font-mono text-primary">scaler.fit(X_train)</div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">4</div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Transform Data</h4>
                  <p className="text-sm text-muted-foreground">Transform both training and test data using the fitted scaler</p>
                  <div className="mt-2 bg-muted p-2 rounded text-xs font-mono text-primary">X_train_scaled = scaler.transform(X_train)<br />X_test_scaled = scaler.transform(X_test)</div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground text-sm">Important: Prevent Data Leakage</h4>
                <p className="text-sm text-muted-foreground">
                  Never fit the scaler on the entire dataset. Always fit on training data only, then transform both training and test data.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Custom Range Normalization */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Move className="h-5 w-5 text-primary" />
            4. Custom Range Normalization
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3">Specifying Custom Ranges</h4>
              <div className="space-y-3">
                <div className="bg-muted p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Default Range</p>
                  <div className="bg-background p-2 rounded text-xs font-mono text-primary mt-1">MinMaxScaler()</div>
                  <p className="text-xs text-muted-foreground mt-1">Scales to [0, 1]</p>
                </div>
                <div className="bg-muted p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Custom Range [-1, 1]</p>
                  <div className="bg-background p-2 rounded text-xs font-mono text-primary mt-1">MinMaxScaler(feature_range=(-1, 1))</div>
                  <p className="text-xs text-muted-foreground mt-1">Scales to [-1, 1]</p>
                </div>
                <div className="bg-muted p-3 rounded">
                  <p className="font-medium text-foreground text-sm">Custom Range [0, 255]</p>
                  <div className="bg-background p-2 rounded text-xs font-mono text-primary mt-1">MinMaxScaler(feature_range=(0, 255))</div>
                  <p className="text-xs text-muted-foreground mt-1">Scales to [0, 255] (image pixel values)</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3">When to Use Custom Ranges</h4>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-3 bg-primary/5 border border-primary/20 rounded">
                  <CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-foreground">[-1, 1] Range</p>
                    <p className="text-xs text-muted-foreground">Useful when features should be centered around zero</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-primary/5 border border-primary/20 rounded">
                  <CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-foreground">[0, 255] Range</p>
                    <p className="text-xs text-muted-foreground">Useful for image processing and pixel values</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-3 bg-primary/5 border border-primary/20 rounded">
                  <CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-foreground">Domain-Specific Ranges</p>
                    <p className="text-xs text-muted-foreground">Use domain knowledge to set appropriate ranges</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Pros and Cons */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            5. Pros and Cons
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                Advantages
              </h4>
              <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li>Bounded range [0, 1] by default</li>
                <li>Preserves the shape of the distribution</li>
                <li>Works well with neural networks</li>
                <li>Simple and interpretable</li>
                <li>Fast to compute</li>
                <li>Useful for image processing</li>
              </ul>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                Limitations
              </h4>
              <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li>Highly sensitive to outliers</li>
                <li>Extreme values compress normal data</li>
                <li>Assumes data has bounded values</li>
                <li>Not suitable for data with unknown min/max</li>
                <li>Can be affected by future data points</li>
                <li>Requires known min/max for new data</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 6. Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            6. Common Mistakes
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Outlier Sensitivity</h4>
                  <p className="text-sm text-muted-foreground">
                    Using Min-Max normalization when data has significant outliers. The outliers will compress the normal data range.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Fitting the scaler on the entire dataset before splitting. Always fit on training data only.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Ignoring Categorical Variables</h4>
                  <p className="text-sm text-muted-foreground">
                    Applying normalization to categorical variables. Only scale numerical features.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Wrong Scaler Choice</h4>
                  <p className="text-sm text-muted-foreground">
                    Using MinMaxScaler when features don't have bounded values. Consider StandardScaler instead.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            7. Best Practices
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Pipelines</h4>
                <p className="text-sm text-muted-foreground">Create pipelines with normalization and model for clean, reproducible code</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Check for Outliers First</h4>
                <p className="text-sm text-muted-foreground">Visualize your data before applying MinMaxScaler</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Save the Scaler</h4>
                <p className="text-sm text-muted-foreground">Save the fitted scaler for use in production/deployment</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Consider Domain Knowledge</h4>
                <p className="text-sm text-muted-foreground">Use domain knowledge to choose appropriate normalization ranges</p>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Impact on Algorithms */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            8. Impact on Different Algorithms
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Algorithms that Benefit from Normalization</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Neural Networks</span> - Faster convergence with [0,1] range</li>
                <li>• <span className="font-medium text-foreground">KNN</span> - Distance-based, sensitive to scale</li>
                <li>• <span className="font-medium text-foreground">K-Means</span> - Distance-based clustering</li>
                <li>• <span className="font-medium text-foreground">PCA</span> - Variance-based, needs comparable scales</li>
                <li>• <span className="font-medium text-foreground">Gradient Descent</span> - Better convergence</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Algorithms that Don't Need Normalization</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Decision Trees</span> - Scale invariant</li>
                <li>• <span className="font-medium text-foreground">Random Forest</span> - Scale invariant</li>
                <li>• <span className="font-medium text-foreground">Gradient Boosting</span> - Scale invariant</li>
                <li>• <span className="font-medium text-foreground">Naive Bayes</span> - Doesn't need scaling</li>
                <li>• <span className="font-medium text-foreground">XGBoost</span> - Handles scaling internally</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Code Examples
          </h2>
          <p className="text-muted-foreground mb-4">
            See normalization techniques in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Normalization Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}