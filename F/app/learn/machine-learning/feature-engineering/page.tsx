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
  Construction
} from "lucide-react";

export default function FeatureEngineeringPage() {
  const result = getSubtopicBySlug("machine-learning", "feature-engineering");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-features",
      label: "Feature Creation",
      code: `import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample dataset
np.random.seed(42)
n = 100

data = {
    'Date': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
    'Price': np.random.normal(100, 20, n),
    'Quantity': np.random.randint(1, 100, n),
    'Category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'Customer_Age': np.random.randint(18, 65, n),
    'Purchase_Year': np.random.randint(2020, 2024, n)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. FEATURE CREATION FROM DATES")
print("=" * 50)

# Extract date components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

print("Date features created:")
print(df[['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend']].head())

print("\\n" + "=" * 50)
print("2. FEATURE CREATION FROM MATH OPERATIONS")
print("=" * 50)

# Mathematical transformations
df['Price_Squared'] = df['Price'] ** 2
df['Price_Log'] = np.log1p(df['Price'] - df['Price'].min() + 1)
df['Price_Sqrt'] = np.sqrt(df['Price'] - df['Price'].min() + 1)
df['Price_Reciprocal'] = 1 / (df['Price'] + 1)

print("Mathematical features created:")
print(df[['Price', 'Price_Squared', 'Price_Log', 'Price_Sqrt', 'Price_Reciprocal']].head())

print("\\n" + "=" * 50)
print("3. INTERACTION FEATURES")
print("=" * 50)

# Interaction features
df['Price_Quantity'] = df['Price'] * df['Quantity']
df['Price_Age'] = df['Price'] * df['Customer_Age']
df['Quantity_Age'] = df['Quantity'] * df['Customer_Age']

print("Interaction features created:")
print(df[['Price', 'Quantity', 'Customer_Age', 'Price_Quantity', 'Price_Age', 'Quantity_Age']].head())

print("\\n" + "=" * 50)
print("4. AGGREGATION FEATURES")
print("=" * 50)

# Aggregation by category
category_stats = df.groupby('Category').agg({
    'Price': ['mean', 'std', 'min', 'max'],
    'Quantity': ['mean', 'std']
}).reset_index()
category_stats.columns = ['Category', 'Price_Mean', 'Price_Std', 'Price_Min', 'Price_Max', 'Quantity_Mean', 'Quantity_Std']

df = df.merge(category_stats, on='Category', how='left')

print("Aggregation features created:")
print(df[['Category', 'Price_Mean', 'Price_Std', 'Quantity_Mean']].drop_duplicates().head())`,
    },
    {
      language: "python-transform",
      label: "Feature Transformation",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy import stats

# Create skewed data
np.random.seed(42)
skewed_data = np.random.exponential(scale=2, size=1000)
df = pd.DataFrame({'Original': skewed_data})

print("=" * 50)
print("1. SKEWNESS ANALYSIS")
print("=" * 50)
print(f"Original skewness: {stats.skew(df['Original']):.3f}")

# Visualize original distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Log Transformation
df['Log'] = np.log1p(df['Original'])
print(f"Log transform skewness: {stats.skew(df['Log']):.3f}")

# 2. Square Root Transformation
df['Sqrt'] = np.sqrt(df['Original'])
print(f"Square root skewness: {stats.skew(df['Sqrt']):.3f}")

# 3. Box-Cox Transformation
boxcox = PowerTransformer(method='box-cox')
df['BoxCox'] = boxcox.fit_transform(df[['Original']])
print(f"Box-Cox skewness: {stats.skew(df['BoxCox']):.3f}")

# 4. Yeo-Johnson Transformation
yeojohnson = PowerTransformer(method='yeo-johnson')
df['YeoJohnson'] = yeojohnson.fit_transform(df[['Original']])
print(f"Yeo-Johnson skewness: {stats.skew(df['YeoJohnson']):.3f}")

# 5. Quantile Transformation
quantile = QuantileTransformer(output_distribution='normal')
df['Quantile'] = quantile.fit_transform(df[['Original']])
print(f"Quantile transform skewness: {stats.skew(df['Quantile']):.3f}")

print("\\n" + "=" * 50)
print("2. TRANSFORMATION SUMMARY")
print("=" * 50)
print(df.describe())

# Create histograms
for idx, col in enumerate(df.columns):
    row = idx // 3
    col_idx = idx % 3
    axes[row, col_idx].hist(df[col], bins=30, alpha=0.7)
    axes[row, col_idx].set_title(f'{col} Distribution\\nSkew: {stats.skew(df[col]):.3f}')
    axes[row, col_idx].axvline(df[col].mean(), color='red', linestyle='--')

plt.tight_layout()
plt.savefig('transformations.png', dpi=300, bbox_inches='tight')
print("\\nTransformation plots saved as 'transformations.png'")`,
    },
    {
      language: "python-binning",
      label: "Binning and Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

# Create sample data
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 80, 100),
    'Income': np.random.normal(50000, 20000, 100),
    'Score': np.random.normal(70, 15, 100)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. MANUAL BINNING")
print("=" * 50)

# Manual binning
df['Age_Group'] = pd.cut(
    df['Age'], 
    bins=[0, 25, 45, 65, 100], 
    labels=['Young', 'Middle', 'Senior', 'Elderly']
)

print("Age groups created:")
print(df[['Age', 'Age_Group']].head(10))

print("\\n" + "=" * 50)
print("2. QUANTILE BINNING")
print("=" * 50)

# Quantile-based binning
df['Income_Level'] = pd.qcut(df['Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print("Income quantile bins:")
print(df[['Income', 'Income_Level']].head(10))
print("\\nIncome level distribution:")
print(df['Income_Level'].value_counts())

print("\\n" + "=" * 50)
print("3. KBINSDISCRETIZER (Scikit-learn)")
print("=" * 50)

# KBinsDiscretizer for uniform binning
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df['Score_Binned'] = kbd.fit_transform(df[['Score']]).astype(int)

print("Score bins:")
print(df[['Score', 'Score_Binned']].head(10))
print("\\nScore bin distribution:")
print(df['Score_Binned'].value_counts().sort_index())

print("\\n" + "=" * 50)
print("4. TARGET-BASED BINNING")
print("=" * 50)

# Create target column for demonstration
df['Target'] = (df['Score'] > df['Score'].median()).astype(int)

# Calculate mean target for each age group
target_binning = df.groupby('Age_Group')['Target'].mean().round(2)
print("Target mean by age group:")
print(target_binning)

# Assign based on target
df['Age_Target_Encoding'] = df['Age_Group'].map(target_binning)
print("\\nTarget encoded features:")
print(df[['Age_Group', 'Age_Target_Encoding']].drop_duplicates())`,
    },
    {
      language: "python-feature-selection",
      label: "Feature Selection",
      code: `import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Create sample dataset
np.random.seed(42)
n = 500

data = {
    'Feature1': np.random.randn(n),
    'Feature2': np.random.randn(n) * 2,
    'Feature3': np.random.randn(n) * 3,
    'Feature4': np.random.randn(n) * 0.5,
    'Feature5': np.random.randn(n) * 4,
    'Feature6': np.random.randn(n) * 0.3,
    'Feature7': np.random.choice(['A', 'B', 'C'], n),
    'Feature8': np.random.choice(['X', 'Y', 'Z'], n)
}

# Create target with some dependencies
target = (data['Feature1'] + data['Feature2'] + np.random.randn(n) * 0.5 > 0).astype(int)
df = pd.DataFrame(data)
df['Target'] = target

print("=" * 50)
print("1. UNIVARIATE FEATURE SELECTION")
print("=" * 50)

# Encode categorical features
le = LabelEncoder()
df['Feature7_Encoded'] = le.fit_transform(df['Feature7'])
df['Feature8_Encoded'] = le.fit_transform(df['Feature8'])

# Select features
X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 
        'Feature7_Encoded', 'Feature8_Encoded']]
y = df['Target']

# SelectKBest with f_classif
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
selected_scores = selector.scores_[selector.get_support()]

print("Selected features with f_classif:")
for feat, score in zip(selected_features, selected_scores):
    print(f"  {feat}: {score:.3f}")

print("\\n" + "=" * 50)
print("2. MUTUAL INFORMATION")
print("=" * 50)

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_features = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("Mutual Information scores:")
print(mi_features)

print("\\n" + "=" * 50)
print("3. FEATURE IMPORTANCE (Random Forest)")
print("=" * 50)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Random Forest feature importance:")
print(importance)

print("\\n" + "=" * 50)
print("4. CORRELATION ANALYSIS")
print("=" * 50)

correlation = df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Target']].corr()
print("Correlation with target:")
print(correlation['Target'].sort_values(ascending=False))`,
    },
    {
      language: "python-time",
      label: "Time Series Features",
      code: `import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
data = {
    'Date': dates,
    'Value': np.random.normal(100, 20, 365) + np.sin(np.arange(365) * 2 * np.pi / 30) * 10,
    'Volume': np.random.randint(50, 500, 365)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. TIME-BASED FEATURES")
print("=" * 50)

# Extract time components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Quarter'] = df['Date'].dt.quarter
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

print("Time features created:")
print(df[['Date', 'Year', 'Month', 'DayOfWeek', 'Quarter', 'IsWeekend']].head())

print("\\n" + "=" * 50)
print("2. LAG FEATURES")
print("=" * 50)

# Create lag features
for lag in [1, 3, 7, 14, 30]:
    df[f'Value_Lag_{lag}'] = df['Value'].shift(lag)
    df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

print("Lag features created:")
print(df[['Date', 'Value', 'Value_Lag_1', 'Value_Lag_7', 'Value_Lag_30']].head(10))

print("\\n" + "=" * 50)
print("3. ROLLING STATISTICS")
print("=" * 50)

# Rolling statistics
for window in [3, 7, 14, 30]:
    df[f'Value_Rolling_Mean_{window}'] = df['Value'].rolling(window=window).mean()
    df[f'Value_Rolling_Std_{window}'] = df['Value'].rolling(window=window).std()
    df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()

print("Rolling statistics created:")
print(df[['Date', 'Value', 'Value_Rolling_Mean_7', 'Value_Rolling_Std_7']].head(10))

print("\\n" + "=" * 50)
print("4. SEASONAL FEATURES")
print("=" * 50)

# Seasonal features
df['Season'] = df['Month'].apply(lambda x: 
    'Winter' if x in [12, 1, 2] else
    'Spring' if x in [3, 4, 5] else
    'Summer' if x in [6, 7, 8] else
    'Fall'
)

# Day of week features
day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
               3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['DayName'] = df['DayOfWeek'].map(day_mapping)

# Cyclical encoding
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

print("Seasonal features created:")
print(df[['Date', 'Season', 'DayName', 'Month_Sin', 'Month_Cos']].head(10))

print("\\n" + "=" * 50)
print("5. FINAL FEATURE SUMMARY")
print("=" * 50)
print(f"Total features created: {len(df.columns)}")
print("\\nFeature list:")
for col in df.columns:
    print(f"  - {col}")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Feature Engineering?",
      options: [
        "Selecting algorithms for a model",
        "Creating and transforming features to improve model performance",
        "Cleaning missing values",
        "Splitting data into train/test sets"
      ],
      correctAnswer: 1,
      explanation: "Feature engineering is the process of creating, transforming, and selecting features to improve model performance and make the patterns in data more apparent.",
    },
    {
      id: 2,
      question: "What is the difference between feature engineering and feature selection?",
      options: [
        "They are the same thing",
        "Feature engineering creates new features, feature selection chooses existing ones",
        "Feature selection creates new features, feature engineering chooses existing ones",
        "They are both about data cleaning"
      ],
      correctAnswer: 1,
      explanation: "Feature engineering creates new features from existing data, while feature selection chooses the most relevant features from the existing set.",
    },
    {
      id: 3,
      question: "Why would you create interaction features?",
      options: [
        "To reduce dimensionality",
        "To capture relationships between variables",
        "To handle missing values",
        "To speed up training"
      ],
      correctAnswer: 1,
      explanation: "Interaction features capture relationships between variables (e.g., product of two features) that can improve model performance.",
    },
    {
      id: 4,
      question: "What is the purpose of polynomial features?",
      options: [
        "To reduce overfitting",
        "To capture non-linear relationships",
        "To encode categorical data",
        "To handle outliers"
      ],
      correctAnswer: 1,
      explanation: "Polynomial features (x², x³, etc.) allow linear models to capture non-linear relationships in the data.",
    },
    {
      id: 5,
      question: "When should you use log transformation?",
      options: [
        "When data has outliers",
        "When data is skewed (right-skewed)",
        "When data is categorical",
        "When data is already normal"
      ],
      correctAnswer: 1,
      explanation: "Log transformation is particularly effective for right-skewed data, helping to make the distribution more normal and reduce the impact of outliers.",
    },
    {
      id: 6,
      question: "What is the Box-Cox transformation?",
      options: [
        "A method for handling missing values",
        "A family of power transformations to make data more normal",
        "A feature selection technique",
        "A scaling method"
      ],
      correctAnswer: 1,
      explanation: "Box-Cox is a family of power transformations that can make data more normally distributed by finding the optimal lambda parameter.",
    },
    {
      id: 7,
      question: "What is target-based encoding?",
      options: [
        "Encoding features based on their values",
        "Encoding categorical features using target variable statistics",
        "Encoding numerical features",
        "Encoding using one-hot"
      ],
      correctAnswer: 1,
      explanation: "Target-based encoding replaces categories with the mean (or other statistic) of the target variable for that category.",
    },
    {
      id: 8,
      question: "What are lag features used for?",
      options: [
        "Classification problems",
        "Time series forecasting",
        "Clustering",
        "Dimensionality reduction"
      ],
      correctAnswer: 1,
      explanation: "Lag features are used in time series analysis to capture past values (e.g., yesterday's price) as features for predicting future values.",
    },
    {
      id: 9,
      question: "What is feature importance in Random Forest?",
      options: [
        "How many features a model has",
        "A measure of how much each feature contributes to predictions",
        "How many trees are in the forest",
        "The accuracy of the model"
      ],
      correctAnswer: 1,
      explanation: "Feature importance in Random Forest measures the contribution of each feature to the model's predictions, helping identify the most valuable features.",
    },
    {
      id: 10,
      question: "What is the purpose of binning features?",
      options: [
        "To increase dimensionality",
        "To create categorical features from continuous ones",
        "To remove outliers",
        "To scale features"
      ],
      correctAnswer: 1,
      explanation: "Binning (discretization) converts continuous variables into categorical ones by grouping values into bins, which can help with handling non-linear relationships.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
   

        {/* 1. What is Feature Engineering? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Hammer className="h-5 w-5 text-primary" />
            1. What is Feature Engineering?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Feature engineering is the process of creating, transforming, and selecting features to improve model performance. It's about making the patterns in data more apparent to algorithms.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Raw Data</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">Feature Engineering</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">Better Model</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Better features → Better predictions</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Why is Feature Engineering important?</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Improves model accuracy</li>
                    <li>Reduces overfitting</li>
                    <li>Handles non-linear relationships</li>
                    <li>Makes patterns more apparent</li>
                    <li>Can reduce training time</li>
                    <li>Domain knowledge integration</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. The Feature Engineering Framework */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Construction className="h-5 w-5 text-primary" />
            2. The Feature Engineering Framework
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">1</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Creation</h4>
                    <p className="text-xs text-muted-foreground">Generate new features from existing data</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">2</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Transformation</h4>
                    <p className="text-xs text-muted-foreground">Apply mathematical transformations</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">3</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Encoding</h4>
                    <p className="text-xs text-muted-foreground">Convert categorical data to numerical</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">4</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Selection</h4>
                    <p className="text-xs text-muted-foreground">Choose the most relevant features</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">5</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Extraction</h4>
                    <p className="text-xs text-muted-foreground">Extract meaningful information</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">6</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Evaluation</h4>
                    <p className="text-xs text-muted-foreground">Assess feature importance and impact</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">7</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Iteration</h4>
                    <p className="text-xs text-muted-foreground">Iteratively improve and refine features</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Feature Creation */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Wand2 className="h-5 w-5 text-primary" />
            3. Feature Creation
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Mathematical Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Polynomial Features</p>
                  <p className="text-muted-foreground">x², x³, x×y, etc. for non-linear relationships</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Interaction Features</p>
                  <p className="text-muted-foreground">Feature1 × Feature2, Feature1 / Feature2</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Trigonometric Features</p>
                  <p className="text-muted-foreground">sin(x), cos(x) for cyclical patterns</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Logarithmic Features</p>
                  <p className="text-muted-foreground">log(x), log1p(x) for skewed data</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Domain-Specific Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Date/Time Features</p>
                  <p className="text-muted-foreground">Year, month, day, hour, weekday, season</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Text Features</p>
                  <p className="text-muted-foreground">Word count, sentiment, TF-IDF, embeddings</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Location Features</p>
                  <p className="text-muted-foreground">Distance, area, clusters, density</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Domain Knowledge</p>
                  <p className="text-muted-foreground">Business-specific calculations and ratios</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Aggregation Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Grouped Statistics</p>
                  <p className="text-muted-foreground">Mean, median, std, min, max by category</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Count Features</p>
                  <p className="text-muted-foreground">Number of occurrences, frequency</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Ratio Features</p>
                  <p className="text-muted-foreground">Proportions, percentages, rates</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Time Series Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Lag Features</p>
                  <p className="text-muted-foreground">Value at t-1, t-2, ..., t-n</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Rolling Statistics</p>
                  <p className="text-muted-foreground">Rolling mean, std, min, max</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Trend Features</p>
                  <p className="text-muted-foreground">Slope, acceleration, moving averages</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Feature Transformation */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Wrench className="h-5 w-5 text-primary" />
            4. Feature Transformation
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Scaling Transformations</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Standardization</p>
                  <p className="text-muted-foreground">z = (x - μ) / σ → Mean=0, Std=1</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Normalization</p>
                  <p className="text-muted-foreground">x' = (x - min) / (max - min) → Range [0,1]</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Robust Scaling</p>
                  <p className="text-muted-foreground">x' = (x - median) / IQR → Robust to outliers</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Distribution Transformations</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Log Transform</p>
                  <p className="text-muted-foreground">Handles right-skewed data</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Box-Cox Transform</p>
                  <p className="text-muted-foreground">Power transformation for normality</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Yeo-Johnson</p>
                  <p className="text-muted-foreground">Handles negative values</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Quantile Transform</p>
                  <p className="text-muted-foreground">Maps to uniform or normal</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Feature Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Filter className="h-5 w-5 text-primary" />
            5. Feature Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Categorical Encoding</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">One-Hot Encoding</p>
                  <p className="text-muted-foreground">Binary columns per category</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Label Encoding</p>
                  <p className="text-muted-foreground">Integer labels for categories</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Target Encoding</p>
                  <p className="text-muted-foreground">Mean target per category</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Frequency Encoding</p>
                  <p className="text-muted-foreground">Category frequency</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Ordinal Encoding</p>
                  <p className="text-muted-foreground">For ordered categories</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Binning/Discretization</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Equal-Width Binning</p>
                  <p className="text-muted-foreground">Equal width intervals</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Equal-Frequency Binning</p>
                  <p className="text-muted-foreground">Equal number of samples per bin</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Quantile Binning</p>
                  <p className="text-muted-foreground">Based on quantiles</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Custom Binning</p>
                  <p className="text-muted-foreground">Domain-specific boundaries</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Feature Selection */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            6. Feature Selection Methods
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Filter Methods</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Correlation</p>
                  <p className="text-muted-foreground">Select based on correlation with target</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Chi-Square</p>
                  <p className="text-muted-foreground">For categorical features</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Mutual Information</p>
                  <p className="text-muted-foreground">Non-linear relationships</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Variance Threshold</p>
                  <p className="text-muted-foreground">Remove low-variance features</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Wrapper Methods</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Forward Selection</p>
                  <p className="text-muted-foreground">Add features one by one</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Backward Elimination</p>
                  <p className="text-muted-foreground">Remove features one by one</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Recursive Elimination</p>
                  <p className="text-muted-foreground">Iteratively remove least important</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Exhaustive Search</p>
                  <p className="text-muted-foreground">Try all combinations</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Embedded Methods</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Lasso Regularization</p>
                  <p className="text-muted-foreground">L1 penalty for feature selection</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Random Forest Importance</p>
                  <p className="text-muted-foreground">Feature importance scores</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">XGBoost Importance</p>
                  <p className="text-muted-foreground">Gain-based importance</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Elastic Net</p>
                  <p className="text-muted-foreground">L1 + L2 regularization</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Feature Extraction */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            7. Feature Extraction
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Dimensionality Reduction</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">PCA</p>
                  <p className="text-muted-foreground">Principal Component Analysis</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">t-SNE</p>
                  <p className="text-muted-foreground">Non-linear reduction</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">UMAP</p>
                  <p className="text-muted-foreground">Uniform manifold projection</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Text Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">TF-IDF</p>
                  <p className="text-muted-foreground">Term frequency-inverse document frequency</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Word Embeddings</p>
                  <p className="text-muted-foreground">Word2Vec, GloVe, FastText</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">BERT Embeddings</p>
                  <p className="text-muted-foreground">Contextual embeddings</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Image Features</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">CNN Features</p>
                  <p className="text-muted-foreground">Convolutional neural network features</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">HOG</p>
                  <p className="text-muted-foreground">Histogram of Oriented Gradients</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">SIFT</p>
                  <p className="text-muted-foreground">Scale-Invariant Feature Transform</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Feature Importance */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            8. Feature Importance Analysis
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Methods to Calculate Importance</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Permutation Importance</p>
                  <p className="text-muted-foreground">Measure drop in performance when feature is shuffled</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Coefficient Importance</p>
                  <p className="text-muted-foreground">Magnitude of coefficients (linear models)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">SHAP Values</p>
                  <p className="text-muted-foreground">Shapley additive explanations</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">LIME</p>
                  <p className="text-muted-foreground">Local Interpretable Model-agnostic Explanations</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Interpreting Importance</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">High Importance</p>
                  <p className="text-muted-foreground">Features with strong predictive power</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Low Importance</p>
                  <p className="text-muted-foreground">Features with little predictive value</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Negative Importance</p>
                  <p className="text-muted-foreground">Features that may hurt performance</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Redundant Features</p>
                  <p className="text-muted-foreground">Highly correlated with other features</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Common Pitfalls */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            9. Common Pitfalls in Feature Engineering
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Using information from test set when creating features. Always create features using only training data.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Over-engineering</h4>
                  <p className="text-sm text-muted-foreground">
                    Creating too many features can lead to overfitting. Focus on quality over quantity.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Ignoring Domain Knowledge</h4>
                  <p className="text-sm text-muted-foreground">
                    Failing to incorporate domain-specific insights can limit feature engineering effectiveness.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Premature Optimization</h4>
                  <p className="text-sm text-muted-foreground">
                    Investing too much time in feature engineering before trying simpler approaches.
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
                <h4 className="font-semibold text-foreground mb-1">Start with EDA</h4>
                <p className="text-sm text-muted-foreground">Understand the data before engineering features</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Iterate and Validate</h4>
                <p className="text-sm text-muted-foreground">Test each feature's impact on model performance</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Domain Knowledge</h4>
                <p className="text-sm text-muted-foreground">Leverage domain expertise for meaningful features</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Automate Where Possible</h4>
                <p className="text-sm text-muted-foreground">Use feature engineering pipelines for reproducibility</p>
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
            See feature engineering techniques in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Feature Engineering Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}