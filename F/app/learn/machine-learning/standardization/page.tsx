import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { 
  AlertCircle, 
  CheckCircle2, 
  Brain,
  Target,
  Scale,
  Sigma,
  Calculator,
  ArrowRight,
  Lightbulb,
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
  Minus,
  Plus,
  Equal,
  X,
  Maximize,
  Minimize,
  Move
} from "lucide-react";

export default function StandardizationPage() {
  const result = getSubtopicBySlug("machine-learning", "standardization");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-standardization",
      label: "Standardization (Z-Score)",
      code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create sample data with different scales
np.random.seed(42)
n_samples = 200

data = {
    'Feature1': np.random.normal(100, 20, n_samples),  # Mean=100, Std=20
    'Feature2': np.random.normal(50, 10, n_samples),   # Mean=50, Std=10
    'Feature3': np.random.normal(0, 5, n_samples),      # Mean=0, Std=5
    'Feature4': np.random.exponential(5, n_samples)     # Exponential distribution
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. BEFORE STANDARDIZATION")
print("=" * 50)
print("First 5 rows:")
print(df.head())
print("\\nStatistics:")
print(df.describe())

# Split data (important for scaling)
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Apply Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrame with scaled data
df_train_scaled = pd.DataFrame(X_train_scaled, columns=df.columns)
df_test_scaled = pd.DataFrame(X_test_scaled, columns=df.columns)

print("\\n" + "=" * 50)
print("2. AFTER STANDARDIZATION (TRAIN SET)")
print("=" * 50)
print("First 5 rows:")
print(df_train_scaled.head())
print("\\nStatistics:")
print(df_train_scaled.describe())

print("\\n" + "=" * 50)
print("3. SCALER PARAMETERS")
print("=" * 50)
print("Means:", scaler.mean_)
print("Scales (Std):", scaler.scale_)

# Visualize comparison
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, col in enumerate(df.columns):
    # Original
    axes[0, i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
    axes[0, i].set_title(f'Original: {col}')
    axes[0, i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    axes[0, i].legend()
    
    # Scaled
    axes[1, i].hist(df_train_scaled[col], bins=30, alpha=0.7, edgecolor='black')
    axes[1, i].set_title(f'Scaled: {col}')
    axes[1, i].axvline(0, color='red', linestyle='--', label='Mean=0')
    axes[1, i].legend()

plt.tight_layout()
plt.savefig('standardization_demo.png', dpi=300, bbox_inches='tight')
print("\\nVisualization saved as 'standardization_demo.png'")`,
    },
    {
      language: "python-pipeline",
      label: "Standardization with Pipeline",
      code: `import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
np.random.seed(42)
n_samples = 500

X = np.random.randn(n_samples, 3) * [10, 100, 1000]  # Different scales
y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline with standardization and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

print("=" * 50)
print("PIPELINE WITH STANDARDIZATION")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Access the fitted scaler
scaler = pipeline.named_steps['scaler']
print("\\nScaler Parameters:")
print(f"Means: {scaler.mean_}")
print(f"Scales: {scaler.scale_}")`,
    },
    {
      language: "python-manual",
      label: "Manual Standardization",
      code: `import numpy as np
import pandas as pd

# Create sample data
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("Original data:", data)

# Manual standardization
mean = np.mean(data)
std = np.std(data)
standardized = (data - mean) / std

print("\\n" + "=" * 50)
print("MANUAL STANDARDIZATION")
print("=" * 50)
print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
print(f"Standardized data: {standardized}")
print(f"New mean: {np.mean(standardized):.2f}")
print(f"New std: {np.std(standardized):.2f}")

# Verify with sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sklearn_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

print("\\nVerification with sklearn:")
print(f"Sklearn standardized: {sklearn_scaled}")
print(f"Difference: {np.max(np.abs(standardized - sklearn_scaled)):.10f}")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is standardization in machine learning?",
      options: [
        "Scaling data to range [0,1]",
        "Transforming data to have mean 0 and standard deviation 1",
        "Removing outliers from data",
        "Converting categorical data to numerical"
      ],
      correctAnswer: 1,
      explanation: "Standardization (Z-score scaling) transforms data to have a mean of 0 and standard deviation of 1.",
    },
    {
      id: 2,
      question: "What is the formula for standardization?",
      options: [
        "x' = (x - min) / (max - min)",
        "x' = (x - mean) / standard_deviation",
        "x' = x / max_absolute_value",
        "x' = log(x)"
      ],
      correctAnswer: 1,
      explanation: "Standardization uses z = (x - μ) / σ, where μ is the mean and σ is the standard deviation.",
    },
    {
      id: 3,
      question: "What is the range of data after standardization?",
      options: [
        "[0, 1]",
        "[-1, 1]",
        "No fixed range, values are centered around 0",
        "[-∞, ∞]"
      ],
      correctAnswer: 2,
      explanation: "Standardization centers data around 0 with std=1, but there is no fixed range. Values can be any real number.",
    },
    {
      id: 4,
      question: "Why is feature scaling important?",
      options: [
        "To make data look nicer",
        "To ensure features contribute equally to the model",
        "To reduce the number of features",
        "To handle missing values"
      ],
      correctAnswer: 1,
      explanation: "Feature scaling ensures that features with larger scales don't dominate the model, especially important for distance-based algorithms.",
    },
    {
      id: 5,
      question: "When should you use standardization?",
      options: [
        "When features are on different scales",
        "When data has many outliers",
        "When data needs to be in [0,1] range",
        "When data is categorical"
      ],
      correctAnswer: 0,
      explanation: "Standardization is useful when features have different scales and you want them to contribute equally to the model.",
    },
    {
      id: 6,
      question: "Should you fit the scaler on training data or entire dataset?",
      options: [
        "On the entire dataset",
        "Only on training data, then transform test data",
        "Only on test data",
        "It doesn't matter"
      ],
      correctAnswer: 1,
      explanation: "Always fit the scaler on training data only, then transform both train and test data to prevent data leakage.",
    },
    {
      id: 7,
      question: "Which algorithms require feature scaling?",
      options: [
        "Decision Trees",
        "SVM, KNN, and Neural Networks",
        "Random Forest",
        "All algorithms"
      ],
      correctAnswer: 1,
      explanation: "Distance-based algorithms (SVM, KNN) and gradient-based algorithms (Neural Networks) require feature scaling.",
    },
    {
      id: 8,
      question: "What happens if you don't scale features for distance-based algorithms?",
      options: [
        "Model will be more accurate",
        "Features with larger scales may dominate the distance calculations",
        "Model will train faster",
        "No effect on the model"
      ],
      correctAnswer: 1,
      explanation: "Without scaling, features with larger magnitudes can dominate distance-based algorithms like KNN and SVM.",
    },
    {
      id: 9,
      question: "What is the difference between standardization and normalization?",
      options: [
        "They are the same thing",
        "Standardization gives mean=0, std=1; Normalization scales to [0,1]",
        "Normalization gives mean=0, std=1; Standardization scales to [0,1]",
        "Standardization is for categorical data"
      ],
      correctAnswer: 1,
      explanation: "Standardization centers data around 0 with unit variance, while normalization scales data to a specific range (usually [0,1]).",
    },
    {
      id: 10,
      question: "What is the key property of standardized data?",
      options: [
        "All values are between 0 and 1",
        "The data has mean 0 and standard deviation 1",
        "All outliers are removed",
        "The data follows a normal distribution"
      ],
      correctAnswer: 1,
      explanation: "Standardization ensures that the transformed data has a mean of 0 and standard deviation of 1, regardless of the original distribution.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
       

        {/* 1. What is Standardization? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            1. What is Standardization?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Standardization is a technique used to <span className="font-semibold text-foreground">transform features to have a mean of 0 and standard deviation of 1</span>. It's one of the most important preprocessing steps for many machine learning algorithms.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Raw Features</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">Standardization</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">μ=0, σ=1</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Each feature → Mean 0, Standard Deviation 1</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Why Standardize?</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                    <li>Prevents features with larger scales from dominating</li>
                    <li>Improves convergence of gradient descent</li>
                    <li>Necessary for distance-based algorithms</li>
                    <li>Makes coefficients comparable</li>
                    <li>Improves numerical stability</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Formula and Properties */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Sigma className="h-5 w-5 text-primary" />
            2. Formula and Properties
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3 text-center">Standardization Formula</h4>
              <div className="bg-muted p-4 rounded-lg text-center">
                <p className="font-mono text-xl">z = (x - μ) / σ</p>
                <p className="text-xs text-muted-foreground mt-2">
                  μ = mean, σ = standard deviation
                </p>
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between border-b border-border pb-2">
                  <span className="text-sm text-muted-foreground">Mean after standardization</span>
                  <span className="font-mono font-bold text-primary">0</span>
                </div>
                <div className="flex items-center justify-between border-b border-border pb-2">
                  <span className="text-sm text-muted-foreground">Standard Deviation after standardization</span>
                  <span className="font-mono font-bold text-primary">1</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Range</span>
                  <span className="font-mono font-bold text-primary">No fixed range</span>
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
                    <p className="text-sm font-medium text-foreground">Preserves Outliers</p>
                    <p className="text-xs text-muted-foreground">Outliers remain visible in the data</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="bg-primary/10 rounded-full p-1 mt-0.5">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">Centers Data</p>
                    <p className="text-xs text-muted-foreground">Data is centered around zero</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="bg-primary/10 rounded-full p-1 mt-0.5">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">Unit Variance</p>
                    <p className="text-xs text-muted-foreground">All features have variance of 1</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <div className="bg-primary/10 rounded-full p-1 mt-0.5">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">Scale Invariant</p>
                    <p className="text-xs text-muted-foreground">Results don't depend on original scale</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3. When to Use Standardization */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            3. When to Use Standardization
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 text-green-600">Best For</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">SVM</span> - Feature scaling is critical</li>
                <li>• <span className="font-medium text-foreground">KNN</span> - Distance-based, very sensitive</li>
                <li>• <span className="font-medium text-foreground">Logistic Regression</span> - Improves convergence</li>
                <li>• <span className="font-medium text-foreground">Neural Networks</span> - Faster training</li>
                <li>• <span className="font-medium text-foreground">PCA</span> - Variance-based</li>
                <li>• <span className="font-medium text-foreground">K-Means</span> - Distance-based clustering</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 text-muted-foreground">When Data Has</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Gaussian-like distribution</span> - Works well with normal data</li>
                <li>• <span className="font-medium text-foreground">Different scales</span> - Features in different ranges</li>
                <li>• <span className="font-medium text-foreground">No extreme outliers</span> - Outliers are preserved</li>
                <li>• <span className="font-medium text-foreground">Unknown bounds</span> - No fixed min/max values</li>
              </ul>
              <div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded">
                <p className="text-xs text-yellow-800 dark:text-yellow-200">
                  ⚠️ Not ideal for data with many extreme outliers
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Implementation Steps */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            4. Implementation Steps
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
                  <p className="text-sm text-muted-foreground">Split your data into training and test sets before scaling</p>
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
                  <p className="text-sm text-muted-foreground">Initialize the StandardScaler from sklearn</p>
                  <div className="mt-2 bg-muted p-2 rounded text-xs font-mono text-primary">scaler = StandardScaler()</div>
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
                  <p className="text-sm text-muted-foreground">Learn mean and standard deviation from training data</p>
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

        {/* 5. Python Implementation */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Calculator className="h-5 w-5 text-primary" />
            5. Python Implementation
          </h2>
          <p className="text-muted-foreground mb-4">
            See standardization in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* 6. Mathematical Explanation */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            6. Mathematical Explanation
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3">Step-by-Step Calculation</h4>
              <div className="space-y-3">
                <div className="bg-muted p-3 rounded">
                  <p className="text-sm font-medium text-foreground">1. Calculate Mean</p>
                  <p className="font-mono text-xs text-primary">μ = (x₁ + x₂ + ... + xₙ) / n</p>
                </div>
                <div className="bg-muted p-3 rounded">
                  <p className="text-sm font-medium text-foreground">2. Calculate Standard Deviation</p>
                  <p className="font-mono text-xs text-primary">σ = √(Σ(xᵢ - μ)² / n)</p>
                </div>
                <div className="bg-muted p-3 rounded">
                  <p className="text-sm font-medium text-foreground">3. Standardize</p>
                  <p className="font-mono text-xs text-primary">z = (x - μ) / σ</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-3">Example Calculation</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between border-b border-border pb-2">
                  <span className="text-muted-foreground">Original data</span>
                  <span className="font-mono">[10, 20, 30, 40, 50]</span>
                </div>
                <div className="flex justify-between border-b border-border pb-2">
                  <span className="text-muted-foreground">Mean (μ)</span>
                  <span className="font-mono font-medium">30</span>
                </div>
                <div className="flex justify-between border-b border-border pb-2">
                  <span className="text-muted-foreground">Std Dev (σ)</span>
                  <span className="font-mono font-medium">14.14</span>
                </div>
                <div className="flex justify-between border-b border-border pb-2">
                  <span className="text-muted-foreground">For x=10</span>
                  <span className="font-mono">(10-30)/14.14 = <span className="font-bold text-primary">-1.41</span></span>
                </div>
                <div className="flex justify-between border-b border-border pb-2">
                  <span className="text-muted-foreground">For x=30</span>
                  <span className="font-mono">(30-30)/14.14 = <span className="font-bold text-primary">0</span></span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">For x=50</span>
                  <span className="font-mono">(50-30)/14.14 = <span className="font-bold text-primary">1.41</span></span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Impact on Algorithms */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            7. Impact on Different Algorithms
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 text-green-600">Algorithms that Benefit</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">SVM</span> - Prevents features from dominating</li>
                <li>• <span className="font-medium text-foreground">KNN</span> - Equal contribution from all features</li>
                <li>• <span className="font-medium text-foreground">Logistic Regression</span> - Faster gradient descent</li>
                <li>• <span className="font-medium text-foreground">Neural Networks</span> - Better weight initialization</li>
                <li>• <span className="font-medium text-foreground">PCA</span> - Proper variance-based components</li>
                <li>• <span className="font-medium text-foreground">K-Means</span> - Balanced clustering</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 text-muted-foreground">Algorithms Unaffected</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Decision Trees</span> - Scale invariant</li>
                <li>• <span className="font-medium text-foreground">Random Forest</span> - Splits based on values</li>
                <li>• <span className="font-medium text-foreground">Gradient Boosting</span> - Handles different scales</li>
                <li>• <span className="font-medium text-foreground">XGBoost</span> - Built-in robustness</li>
              </ul>
              <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
                <p className="text-xs text-blue-800 dark:text-blue-200">
                  💡 For tree-based models, standardization is optional but doesn't hurt
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Common Mistakes and Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            8. Common Mistakes and Best Practices
          </h2>

          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-destructive" />
                  Common Mistakes
                </h4>
                <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                  <li>Scaling before splitting data</li>
                  <li>Fitting scaler on test data</li>
                  <li>Forgetting to save scaler for deployment</li>
                  <li>Scaling categorical variables</li>
                  <li>Not checking for outliers first</li>
                </ul>
              </div>

              <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-primary" />
                  Best Practices
                </h4>
                <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                  <li>Use pipelines for clean code</li>
                  <li>Fit on training data only</li>
                  <li>Save scaler for production</li>
                  <li>Check feature distributions first</li>
                  <li>Consider RobustScaler for outliers</li>
                </ul>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Pipeline Example</h4>
              <div className="bg-muted p-3 rounded text-xs font-mono">
                <p className="text-primary">from sklearn.pipeline import Pipeline</p>
                <p className="text-primary">from sklearn.preprocessing import StandardScaler</p>
                <p className="text-primary">from sklearn.linear_model import LogisticRegression</p>
                <p className="text-foreground mt-2">pipeline = Pipeline([</p>
                <p className="text-foreground pl-4">('scaler', StandardScaler()),</p>
                <p className="text-foreground pl-4">('classifier', LogisticRegression())</p>
                <p className="text-foreground">])</p>
                <p className="text-primary mt-2">pipeline.fit(X_train, y_train)</p>
                <p className="text-primary">predictions = pipeline.predict(X_test)</p>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Pros and Cons */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            9. Pros and Cons
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                Advantages
              </h4>
              <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li>Works well with Gaussian-distributed data</li>
                <li>Preserves outliers (useful for detection)</li>
                <li>No fixed range restrictions</li>
                <li>Makes coefficients interpretable</li>
                <li>Standard choice for many algorithms</li>
                <li>Robust to scale differences</li>
              </ul>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                Limitations
              </h4>
              <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1">
                <li>Assumes Gaussian distribution</li>
                <li>Sensitive to outliers (preserves them)</li>
                <li>No fixed range for bounded data</li>
                <li>Not ideal for sparse data</li>
                <li>Can be affected by extreme values</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Standardization Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}