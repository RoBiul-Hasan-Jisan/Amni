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
  Wand2
} from "lucide-react";

export default function DataPreprocessingPage() {
  const result = getSubtopicBySlug("machine-learning", "data-preprocessing");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-cleaning",
      label: "Data Cleaning",
      code: `import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Create sample dataset with missing values
data = {
    'Age': [25, 30, np.nan, 35, 40, 28, np.nan, 45],
    'Salary': [50000, 60000, 55000, np.nan, 75000, 48000, 62000, 80000],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', np.nan, 'IT', 'Finance'],
    'Experience': [2, 5, 3, 7, 10, 4, 6, 12]
}
df = pd.DataFrame(data)

print("Original Data:")
print(df)
print("\\nMissing Values:")
print(df.isnull().sum())

# 1. Handle missing values - Numeric columns
numeric_cols = ['Age', 'Salary']
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# 2. Handle missing values - Categorical columns
categorical_cols = ['Department']
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

print("\\nAfter Handling Missing Values:")
print(df)

# 3. Encode categorical variables
le = LabelEncoder()
df['Department_Encoded'] = le.fit_transform(df['Department'])
df = df.drop('Department', axis=1)

print("\\nAfter Encoding:")
print(df)`,
    },
    {
      language: "python-scaling",
      label: "Feature Scaling",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Create sample data
data = {
    'Feature1': [100, 200, 150, 300, 250, 180, 220, 350],
    'Feature2': [10, 20, 15, 30, 25, 18, 22, 35],
    'Feature3': [1, 2, 1.5, 3, 2.5, 1.8, 2.2, 3.5]
}
df = pd.DataFrame(data)

print("Original Data:")
print(df)
print("\\nStatistics:")
print(df.describe())

# 1. Standardization (Z-score scaling)
scaler_standard = StandardScaler()
df_standard = pd.DataFrame(
    scaler_standard.fit_transform(df),
    columns=df.columns
)

print("\\nStandardized Data:")
print(df_standard)
print("\\nStandardized Statistics:")
print(df_standard.describe())

# 2. Min-Max Scaling (Normalization)
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=df.columns
)

print("\\nMin-Max Scaled Data:")
print(df_minmax)
print("\\nMin-Max Statistics:")
print(df_minmax.describe())

# 3. Robust Scaling (for outliers)
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(
    scaler_robust.fit_transform(df),
    columns=df.columns
)

print("\\nRobust Scaled Data:")
print(df_robust.head())`,
    },
    {
      language: "python-encoding",
      label: "Encoding Categorical Data",
      code: `import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Create sample data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'Size': ['S', 'M', 'L', 'M', 'L', 'S'],
    'Price': [100, 150, 200, 120, 180, 220]
}
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# 1. Label Encoding (for ordinal data)
le = LabelEncoder()
df['Size_LabelEncoded'] = le.fit_transform(df['Size'])
print("\\nLabel Encoded Size:")
print(df[['Size', 'Size_LabelEncoded']])

# 2. One-Hot Encoding (for nominal data)
df_onehot = pd.get_dummies(df, columns=['Color'], prefix=['Color'])
print("\\nOne-Hot Encoded Color:")
print(df_onehot.head())

# 3. ColumnTransformer for mixed data
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Color']),
        ('label', LabelEncoder(), ['Size'])  # Note: This won't work directly in ColumnTransformer
    ]
)

# Alternative: Use separate encodings
print("\\nAlternative Approach:")
print("Use OneHotEncoder for nominal categories")
print("Use LabelEncoder for ordinal categories")`,
    },
    {
      language: "python-split",
      label: "Train-Test Split",
      code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Create sample data
np.random.seed(42)
data = {
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Feature3': np.random.randn(100),
    'Target': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df['Target'].value_counts())

# 1. Simple Train-Test Split
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"\\nSimple Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train target distribution: {y_train.value_counts().to_dict()}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# 2. Stratified Split (maintains class distribution)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train_strat, X_test_strat = X.iloc[train_idx], X.iloc[test_idx]
    y_train_strat, y_test_strat = y.iloc[train_idx], y.iloc[test_idx]

print(f"\\nStratified Split:")
print(f"Training set: {X_train_strat.shape[0]} samples")
print(f"Test set: {X_test_strat.shape[0]} samples")
print(f"Train target distribution: {y_train_strat.value_counts().to_dict()}")
print(f"Test target distribution: {y_test_strat.value_counts().to_dict()}")`,
    },
    {
      language: "python-outliers",
      label: "Outlier Detection",
      code: `import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

# Create sample data with outliers
np.random.seed(42)
data = np.random.randn(100, 2) * 10
data = np.vstack([data, [[100, 100], [-100, -100]]])  # Add outliers
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

print("Data shape:", df.shape)
print("\\nFirst 5 rows:")
print(df.head())

# 1. Z-Score Method
z_scores = np.abs(stats.zscore(df))
threshold = 3
outliers_z = np.where(z_scores > threshold)
print(f"\\nZ-Score Method:")
print(f"Number of outliers detected: {len(outliers_z[0])}")

# 2. IQR Method (Interquartile Range)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
print(f"\\nIQR Method:")
print(f"Number of outliers detected: {outliers_iqr.sum().sum()}")

# 3. Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_if = iso_forest.fit_predict(df)
print(f"\\nIsolation Forest:")
print(f"Number of outliers detected: {sum(outliers_if == -1)}")

# Show outlier detection results
print("\\nOutlier Detection Summary:")
print("- Z-Score: Detects extreme values based on standard deviation")
print("- IQR: Detects values outside 1.5 * IQR range")
print("- Isolation Forest: Detects anomalies using isolation mechanism")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Data Preprocessing?",
      options: [
        "Cleaning and transforming raw data into a usable format",
        "Storing data in a database",
        "Visualizing data",
        "Collecting data from sources"
      ],
      correctAnswer: 0,
      explanation: "Data preprocessing is the process of cleaning, transforming, and organizing raw data into a format suitable for analysis and modeling.",
    },
    {
      id: 2,
      question: "Why is handling missing values important?",
      options: [
        "It makes the data look cleaner",
        "Missing values can cause errors and bias in models",
        "It reduces dataset size",
        "It increases processing speed"
      ],
      correctAnswer: 1,
      explanation: "Missing values can lead to biased models, errors during training, and poor predictions. Proper handling is essential for model quality.",
    },
    {
      id: 3,
      question: "What is the difference between Standardization and Normalization?",
      options: [
        "They are the same thing",
        "Standardization centers data around 0 with std=1, Normalization scales to [0,1]",
        "Normalization centers data around 0, Standardization scales to [0,1]",
        "Standardization is for numeric data, Normalization is for categorical data"
      ],
      correctAnswer: 1,
      explanation: "Standardization (Z-score) centers data to mean=0 and std=1, while Normalization (Min-Max) scales data to a specific range, usually [0,1].",
    },
    {
      id: 4,
      question: "What is One-Hot Encoding used for?",
      options: [
        "Encoding numeric values",
        "Encoding categorical variables into binary columns",
        "Scaling numeric features",
        "Handling missing values"
      ],
      correctAnswer: 1,
      explanation: "One-Hot Encoding creates binary columns for each category, representing categorical variables in a format suitable for machine learning algorithms.",
    },
    {
      id: 5,
      question: "What is the purpose of Train-Test Split?",
      options: [
        "To make training faster",
        "To evaluate model performance on unseen data",
        "To reduce the dataset size",
        "To clean the data"
      ],
      correctAnswer: 1,
      explanation: "Train-Test Split separates data into training and testing sets to evaluate model performance on unseen data and prevent overfitting.",
    },
    {
      id: 6,
      question: "What is the IQR method used for?",
      options: [
        "Data scaling",
        "Outlier detection using interquartile range",
        "Feature encoding",
        "Data visualization"
      ],
      correctAnswer: 1,
      explanation: "IQR (Interquartile Range) method detects outliers by identifying values that fall outside 1.5 × IQR below Q1 or above Q3.",
    },
    {
      id: 7,
      question: "When should you use Label Encoding instead of One-Hot Encoding?",
      options: [
        "For numeric data",
        "For ordinal categorical data where order matters",
        "For all categorical data",
        "For continuous data"
      ],
      correctAnswer: 1,
      explanation: "Label Encoding is appropriate for ordinal categorical data where there is a meaningful order (e.g., Small, Medium, Large).",
    },
    {
      id: 8,
      question: "What is the effect of not scaling features?",
      options: [
        "No effect on model performance",
        "Features with larger scales can dominate the model",
        "It improves model speed",
        "It prevents overfitting"
      ],
      correctAnswer: 1,
      explanation: "Without scaling, features with larger ranges can dominate distance-based algorithms, leading to biased models.",
    },
    {
      id: 9,
      question: "What is a stratified train-test split?",
      options: [
        "Splitting data alphabetically",
        "Splitting while maintaining the original class distribution",
        "Splitting by feature values",
        "Splitting randomly"
      ],
      correctAnswer: 1,
      explanation: "Stratified split maintains the same proportion of classes in both training and test sets, especially important for imbalanced datasets.",
    },
    {
      id: 10,
      question: "What is the purpose of feature engineering?",
      options: [
        "To remove features from the dataset",
        "To create new features from existing ones to improve model performance",
        "To scale the dataset",
        "To split the data"
      ],
      correctAnswer: 1,
      explanation: "Feature engineering involves creating new features or transforming existing ones to better represent the problem and improve model performance.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
       

        {/* 1. What is Data Preprocessing? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            1. What is Data Preprocessing?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Data preprocessing is the process of cleaning, transforming, and organizing raw data into a format that is suitable for machine learning models.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Raw Data</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">Preprocessing</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">Clean Data</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Garbage In → Garbage Out</p>
              </div>
              <div className="mt-3 text-sm text-muted-foreground">
                <p><span className="font-semibold text-foreground">Key Rule:</span> Better data beats better algorithms!</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Why is it important?</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Raw data is often messy and incomplete</li>
                    <li>Improves model accuracy and performance</li>
                    <li>Reduces training time</li>
                    <li>Prevents bias and errors</li>
                    <li>Ensures data is in the right format</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. The Data Preprocessing Pipeline */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            2. The Data Preprocessing Pipeline
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">1</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Collection</h4>
                    <p className="text-xs text-muted-foreground">Gather raw data from various sources</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">2</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Cleaning</h4>
                    <p className="text-xs text-muted-foreground">Handle missing values, duplicates, outliers</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">3</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Transformation</h4>
                    <p className="text-xs text-muted-foreground">Scale, encode, and transform features</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">4</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Engineering</h4>
                    <p className="text-xs text-muted-foreground">Create new features from existing data</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">5</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Splitting</h4>
                    <p className="text-xs text-muted-foreground">Divide into train, validation, test sets</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">6</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Augmentation</h4>
                    <p className="text-xs text-muted-foreground">Increase dataset size (for images/text)</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">7</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Quality Check</h4>
                    <p className="text-xs text-muted-foreground">Validate data quality and distributions</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Handling Missing Values */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Filter className="h-5 w-5 text-primary" />
            3. Handling Missing Values
          </h2>

          <div className="space-y-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Common Techniques</h4>
                <div className="space-y-3">
                  <div>
                    <h5 className="font-medium text-foreground text-sm">Mean/Median Imputation</h5>
                    <p className="text-xs text-muted-foreground">Replace missing values with mean or median</p>
                    <div className="bg-muted p-2 rounded mt-1">
                      <code className="text-xs">df.fillna(df.mean())</code>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-medium text-foreground text-sm">Mode Imputation</h5>
                    <p className="text-xs text-muted-foreground">Replace with most frequent value (categorical)</p>
                    <div className="bg-muted p-2 rounded mt-1">
                      <code className="text-xs">df.fillna(df.mode()[0])</code>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-medium text-foreground text-sm">Forward/Backward Fill</h5>
                    <p className="text-xs text-muted-foreground">Use previous or next value (time series)</p>
                    <div className="bg-muted p-2 rounded mt-1">
                      <code className="text-xs">df.fillna(method='ffill')</code>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Advanced Techniques</h4>
                <div className="space-y-3">
                  <div>
                    <h5 className="font-medium text-foreground text-sm">KNN Imputation</h5>
                    <p className="text-xs text-muted-foreground">Use k-nearest neighbors to estimate values</p>
                  </div>
                  <div>
                    <h5 className="font-medium text-foreground text-sm">Regression Imputation</h5>
                    <p className="text-xs text-muted-foreground">Predict missing values using regression</p>
                  </div>
                  <div>
                    <h5 className="font-medium text-foreground text-sm">Multiple Imputation</h5>
                    <p className="text-xs text-muted-foreground">Create multiple imputed datasets</p>
                  </div>
                  <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                    <p className="text-xs text-foreground font-medium">⚠️ When to Drop</p>
                    <p className="text-xs text-muted-foreground">
  Drop rows/columns with more than 50–60% missing values
</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Feature Scaling */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Sliders className="h-5 w-5 text-primary" />
            4. Feature Scaling
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Standardization (Z-score)</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">z = (x - μ) / σ</p>
                <p className="text-xs text-muted-foreground mt-1">Mean = 0, Std = 1</p>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use when:</span> Data has outliers, algorithms assume Gaussian distribution (SVM, Logistic Regression)
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Normalization (Min-Max)</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">x' = (x - min) / (max - min)</p>
                <p className="text-xs text-muted-foreground mt-1">Range = [0, 1]</p>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use when:</span> Data has bounded ranges, algorithms use distance (KNN, Neural Networks)
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Robust Scaling</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">x' = (x - median) / IQR</p>
                <p className="text-xs text-muted-foreground mt-1">Uses median and IQR</p>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use when:</span> Data has many outliers
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Max Abs Scaling</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">x' = x / max|X|</p>
                <p className="text-xs text-muted-foreground mt-1">Range = [-1, 1]</p>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use when:</span> Sparse data with zeros
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Encoding Categorical Variables */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Wand2 className="h-5 w-5 text-primary" />
            5. Encoding Categorical Variables
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Label Encoding</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="text-xs text-muted-foreground">Converts categories to numbers</p>
                <div className="mt-2 text-xs">
                  <p className="font-medium text-foreground">Example:</p>
                  <p>Red → 0, Blue → 1, Green → 2</p>
                </div>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use for:</span> Ordinal data (order matters)
                </p>
                <p className="text-xs text-destructive mt-1">
                  ⚠️ Not suitable for nominal data (creates false ordinal relationships)
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">One-Hot Encoding</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="text-xs text-muted-foreground">Creates binary columns for each category</p>
                <div className="mt-2 text-xs">
                  <p className="font-medium text-foreground">Example:</p>
                  <p>Red: [1,0,0], Blue: [0,1,0], Green: [0,0,1]</p>
                </div>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use for:</span> Nominal data (no order)
                </p>
                <p className="text-xs text-destructive mt-1">
                  ⚠️ Creates many columns (curse of dimensionality)
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Target Encoding</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="text-xs text-muted-foreground">Replaces category with mean target value</p>
                <div className="mt-2 text-xs">
                  <p className="font-medium text-foreground">Example:</p>
                  <p>Category 'A' has 80% positive → encode as 0.8</p>
                </div>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use for:</span> High-cardinality categorical variables
                </p>
                <p className="text-xs text-destructive mt-1">
                  ⚠️ Risk of overfitting (use cross-validation)
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Frequency Encoding</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="text-xs text-muted-foreground">Replaces category with its frequency</p>
                <div className="mt-2 text-xs">
                  <p className="font-medium text-foreground">Example:</p>
                  <p>Category appears 100 times → encode as 100</p>
                </div>
              </div>
              <div className="mt-2">
                <p className="text-xs text-muted-foreground">
                  <span className="font-semibold text-foreground">Use for:</span> High-cardinality categorical variables
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                   Simple, no risk of overfitting
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Handling Outliers */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <ScatterChart className="h-5 w-5 text-primary" />
            6. Handling Outliers
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Z-Score Method</h4>
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">
                  Identifies values beyond ±3 standard deviations from mean
                </p>
                <div className="bg-muted p-2 rounded">
                  <code className="text-xs">z = (x - μ) / σ</code>
                </div>
                <p className="text-xs text-muted-foreground">
  <span className="font-medium text-foreground">Threshold:</span> |z| &gt; 3
</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">IQR Method</h4>
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">
                  Uses interquartile range to detect outliers
                </p>
                <div className="bg-muted p-2 rounded">
                  <code className="text-xs">IQR = Q3 - Q1</code>
                </div>
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">Threshold:</span> &lt; Q1 - 1.5×IQR or &gt; Q3 + 1.5×IQR
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Isolation Forest</h4>
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">
                  ML-based outlier detection using isolation mechanism
                </p>
                <div className="bg-muted p-2 rounded">
                  <code className="text-xs">from sklearn.ensemble import IsolationForest</code>
                </div>
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">Pros:</span> Works with high-dimensional data
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4 bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">Outlier Treatment Options</h4>
            <div className="grid md:grid-cols-4 gap-3 text-xs">
              <div className="bg-muted p-2 rounded text-center">
                <p className="font-medium text-foreground">Remove</p>
                <p className="text-muted-foreground">Drop outlier rows</p>
                <p className="text-destructive">⚠️ If few outliers</p>
              </div>
              <div className="bg-muted p-2 rounded text-center">
                <p className="font-medium text-foreground">Clip</p>
                <p className="text-muted-foreground">Cap at threshold</p>
                <p className="text-muted-foreground"> Preserves data</p>
              </div>
              <div className="bg-muted p-2 rounded text-center">
                <p className="font-medium text-foreground">Transform</p>
                <p className="text-muted-foreground">Log/Box-Cox</p>
                <p className="text-muted-foreground"> Reduces impact</p>
              </div>
              <div className="bg-muted p-2 rounded text-center">
                <p className="font-medium text-foreground">Impute</p>
                <p className="text-muted-foreground">Replace with mean/median</p>
                <p className="text-muted-foreground">⚠️ May lose information</p>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Feature Engineering */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Wand2 className="h-5 w-5 text-primary" />
            7. Feature Engineering
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Creating New Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Polynomial Features</p>
                  <p className="text-muted-foreground">x², x³, x×y, etc.</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Interaction Features</p>
                  <p className="text-muted-foreground">Feature1 × Feature2</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Aggregation</p>
                  <p className="text-muted-foreground">Group by: mean, sum, count, etc.</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Domain Features</p>
                  <p className="text-muted-foreground">Business/domain-specific calculations</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Feature Transformation</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Log Transformation</p>
                  <p className="text-muted-foreground">Reduces skewness, handles scale</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Box-Cox Transformation</p>
                  <p className="text-muted-foreground">Makes data more normal-like</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Binning</p>
                  <p className="text-muted-foreground">Convert continuous to categorical</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Encoding Temporal</p>
                  <p className="text-muted-foreground">Year, month, day, hour, etc.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Data Splitting */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            8. Data Splitting
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Training Set</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="text-2xl font-bold text-primary">60-80%</p>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Used to train the model. The model learns patterns from this data.
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Validation Set</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="text-2xl font-bold text-blue-500">10-20%</p>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Used to tune hyperparameters and prevent overfitting.
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Test Set</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="text-2xl font-bold text-green-500">10-20%</p>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                Used only for final evaluation. Never seen during training.
              </p>
            </div>
          </div>

          <div className="mt-4 bg-muted/50 rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">Splitting Methods</h4>
            <div className="grid md:grid-cols-3 gap-3 text-xs">
              <div className="bg-card p-2 rounded">
                <p className="font-medium text-foreground">Simple Split</p>
                <p className="text-muted-foreground">Random 70-30 or 80-20 split</p>
              </div>
              <div className="bg-card p-2 rounded">
                <p className="font-medium text-foreground">Stratified Split</p>
                <p className="text-muted-foreground">Maintains class distribution</p>
              </div>
              <div className="bg-card p-2 rounded">
                <p className="font-medium text-foreground">Time-Based Split</p>
                <p className="text-muted-foreground">For time series data</p>
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
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Using information from test set during preprocessing. Always split before scaling/encoding!
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Wrong Imputation</h4>
                  <p className="text-sm text-muted-foreground">
                    Using mean for categorical data or mode for continuous data.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Inappropriate Scaling</h4>
                  <p className="text-sm text-muted-foreground">
                    Scaling categorical variables or using wrong scaling method.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Ignoring Outliers</h4>
                  <p className="text-sm text-muted-foreground">
                    Not detecting and handling outliers, leading to biased models.
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
                <h4 className="font-semibold text-foreground mb-1">Fit on train only</h4>
                <p className="text-sm text-muted-foreground">Always fit preprocessors on training data, then transform test data</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Validate preprocessing</h4>
                <p className="text-sm text-muted-foreground">Check distributions, missing values, and data types after each step</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Document steps</h4>
                <p className="text-sm text-muted-foreground">Keep track of all preprocessing steps for reproducibility</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use pipelines</h4>
                <p className="text-sm text-muted-foreground">Create preprocessing pipelines for cleaner, reproducible code</p>
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
            See data preprocessing techniques in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Data Preprocessing Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}