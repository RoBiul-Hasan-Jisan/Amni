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
  Circle
} from "lucide-react";

export default function ExploratoryDataAnalysisPage() {
  const result = getSubtopicBySlug("machine-learning", "exploratory-data-analysis");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-basic",
      label: "Basic EDA with Pandas",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 65, 200),
    'Salary': np.random.normal(50000, 15000, 200),
    'Experience': np.random.randint(0, 30, 200),
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 200),
    'Performance': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], 200)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. BASIC INFORMATION")
print("=" * 50)
print("Dataset shape:", df.shape)
print("\\nColumn info:")
print(df.info())

print("\\n" + "=" * 50)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df.describe())

print("\\n" + "=" * 50)
print("3. DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\\n" + "=" * 50)
print("4. MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\\n" + "=" * 50)
print("5. UNIQUE VALUES")
print("=" * 50)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")`,
    },
    {
      language: "python-visualization",
      label: "Data Visualization",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 65, 200),
    'Salary': np.random.normal(50000, 15000, 200),
    'Experience': np.random.randint(0, 30, 200),
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 200)
}
df = pd.DataFrame(data)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Distribution Plot (Histogram)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df['Age'], kde=True, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')
sns.histplot(df['Salary'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Salary Distribution')
sns.histplot(df['Experience'], kde=True, ax=axes[1,0])
axes[1,0].set_title('Experience Distribution')

# 2. Box Plot
sns.boxplot(x='Department', y='Salary', data=df, ax=axes[1,1])
axes[1,1].set_title('Salary by Department')
plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = df[['Age', 'Salary', 'Experience']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation.png', dpi=300, bbox_inches='tight')

# 4. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Experience', y='Salary', hue='Department', data=df)
plt.title('Experience vs Salary by Department')
plt.savefig('scatter.png', dpi=300, bbox_inches='tight')

print("Visualizations saved successfully!")`,
    },
    {
      language: "python-outliers",
      label: "Outlier Detection",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset with outliers
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 65, 150),
    'Salary': np.random.normal(50000, 15000, 150),
    'Experience': np.random.randint(0, 25, 150)
}
df = pd.DataFrame(data)

# Add some outliers
df.loc[0, 'Salary'] = 250000  # Extreme high salary
df.loc[1, 'Salary'] = -5000   # Extreme low salary
df.loc[2, 'Age'] = 120        # Extreme age

print("=" * 50)
print("1. OUTLIER DETECTION USING IQR")
print("=" * 50)

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

for col in ['Age', 'Salary', 'Experience']:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    print(f"\\nColumn: {col}")
    print(f"  Lower bound: {lower:.2f}")
    print(f"  Upper bound: {upper:.2f}")
    print(f"  Outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier values: {outliers[col].values}")

print("\\n" + "=" * 50)
print("2. BOX PLOT VISUALIZATION")
print("=" * 50)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['Age', 'Salary', 'Experience']):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Box Plot - {col}')
plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
print("Box plots saved as 'boxplots.png'")

print("\\n" + "=" * 50)
print("3. Z-SCORE METHOD")
print("=" * 50)
from scipy import stats
z_scores = np.abs(stats.zscore(df[['Age', 'Salary', 'Experience']]))
threshold = 3
outliers_z = np.where(z_scores > threshold)
print(f"Number of outliers detected: {len(outliers_z[0])}")`,
    },
    {
      language: "python-correlation",
      label: "Correlation Analysis",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Create sample dataset
np.random.seed(42)
n = 200
data = {
    'Age': np.random.randint(18, 65, n),
    'Experience': np.random.randint(0, 30, n),
    'Salary': np.random.normal(50000, 15000, n),
    'Education_Years': np.random.randint(10, 20, n),
    'Performance_Score': np.random.uniform(1, 5, n)
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. PEARSON CORRELATION")
print("=" * 50)
pearson_corr = df.corr(method='pearson')
print(pearson_corr)

print("\\n" + "=" * 50)
print("2. SPEARMAN CORRELATION")
print("=" * 50)
spearman_corr = df.corr(method='spearman')
print(spearman_corr)

print("\\n" + "=" * 50)
print("3. DETAILED CORRELATION ANALYSIS")
print("=" * 50)
for col1 in df.columns:
    for col2 in df.columns:
        if col1 < col2:
            pearson, p_pearson = pearsonr(df[col1], df[col2])
            spearman, p_spearman = spearmanr(df[col1], df[col2])
            print(f"{col1} vs {col2}:")
            print(f"  Pearson: {pearson:.3f} (p={p_pearson:.4f})")
            print(f"  Spearman: {spearman:.3f} (p={p_spearman:.4f})")

print("\\n" + "=" * 50)
print("4. HEATMAP VISUALIZATION")
print("=" * 50)
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f', 
            linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as 'heatmap.png'")`,
    },
    {
      language: "python-categorical",
      label: "Categorical Data Analysis",
      code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
np.random.seed(42)
n = 300
data = {
    'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n, 
                                   p=[0.3, 0.2, 0.25, 0.15, 0.1]),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
    'Performance': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], n,
                                   p=[0.2, 0.3, 0.35, 0.15]),
    'Salary': np.random.normal(50000, 15000, n),
    'Satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.15, 0.25, 0.3, 0.2])
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. FREQUENCY TABLES")
print("=" * 50)
for col in ['Department', 'Education', 'Performance']:
    print(f"\\n{col}:")
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True))

print("\\n" + "=" * 50)
print("2. CONTINGENCY TABLES")
print("=" * 50)
print("Department vs Performance:")
print(pd.crosstab(df['Department'], df['Performance'], normalize='index'))

print("\\n" + "=" * 50)
print("3. GROUPED STATISTICS")
print("=" * 50)
print(df.groupby('Department')['Salary'].agg(['mean', 'median', 'std']))

print("\\n" + "=" * 50)
print("4. CROSS TABULATION")
print("=" * 50)
print("Education vs Department:")
print(pd.crosstab(df['Education'], df['Department']))

print("\\n" + "=" * 50)
print("5. CATEGORICAL PLOTS")
print("=" * 50)

# Count plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for i, col in enumerate(['Department', 'Education', 'Performance', 'Satisfaction']):
    row = i // 2
    col_idx = i % 2
    sns.countplot(x=col, data=df, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} Distribution')
    axes[row, col_idx].tick_params(rotation=45)
plt.tight_layout()
plt.savefig('categorical_plots.png', dpi=300, bbox_inches='tight')

print("Categorical plots saved as 'categorical_plots.png'")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Exploratory Data Analysis (EDA)?",
      options: [
        "Final model evaluation",
        "Initial investigation of data to understand its main characteristics",
        "Data cleaning only",
        "Model deployment"
      ],
      correctAnswer: 1,
      explanation: "EDA is the initial process of analyzing data to discover patterns, spot anomalies, test hypotheses, and check assumptions using summary statistics and visualizations.",
    },
    {
      id: 2,
      question: "What is the purpose of descriptive statistics in EDA?",
      options: [
        "To visualize data",
        "To summarize and describe the main features of data",
        "To clean the data",
        "To build models"
      ],
      correctAnswer: 1,
      explanation: "Descriptive statistics summarize and describe the main features of data, including measures of central tendency, dispersion, and shape of distribution.",
    },
    {
      id: 3,
      question: "Which visualization is best for showing distribution of a single continuous variable?",
      options: [
        "Scatter plot",
        "Histogram with KDE",
        "Box plot",
        "Bar chart"
      ],
      correctAnswer: 1,
      explanation: "Histograms with KDE (Kernel Density Estimation) are excellent for visualizing the distribution of a single continuous variable, showing both frequency and probability density.",
    },
    {
      id: 4,
      question: "What does a box plot show?",
      options: [
        "Only the mean",
        "Median, quartiles, and outliers",
        "Correlation between variables",
        "Data distribution over time"
      ],
      correctAnswer: 1,
      explanation: "A box plot displays the median, quartiles (Q1, Q3), interquartile range (IQR), and potential outliers, providing a comprehensive summary of data distribution.",
    },
    {
      id: 5,
      question: "What is the IQR method used for in EDA?",
      options: [
        "Data scaling",
        "Outlier detection",
        "Feature encoding",
        "Correlation analysis"
      ],
      correctAnswer: 1,
      explanation: "The IQR (Interquartile Range) method identifies outliers by detecting values that fall outside 1.5 × IQR below Q1 or above Q3.",
    },
    {
      id: 6,
      question: "What is a correlation matrix used for?",
      options: [
        "To show data distribution",
        "To display relationships between multiple variables",
        "To encode categorical data",
        "To handle missing values"
      ],
      correctAnswer: 1,
      explanation: "A correlation matrix shows the pairwise correlation coefficients between multiple variables, helping identify linear relationships in the data.",
    },
    {
      id: 7,
      question: "What is the difference between Pearson and Spearman correlation?",
      options: [
        "They are the same",
        "Pearson measures linear relationships, Spearman measures monotonic relationships",
        "Spearman measures linear, Pearson measures monotonic",
        "Spearman is for categorical data"
      ],
      correctAnswer: 1,
      explanation: "Pearson correlation measures linear relationships, while Spearman correlation measures monotonic relationships and is less sensitive to outliers.",
    },
    {
      id: 8,
      question: "What is a pair plot used for?",
      options: [
        "Single variable analysis",
        "Visualizing relationships between all pairs of variables",
        "Time series analysis",
        "Categorical data analysis"
      ],
      correctAnswer: 1,
      explanation: "Pair plots (or scatter plot matrices) show scatter plots for every pair of variables, providing a comprehensive view of relationships in the dataset.",
    },
    {
      id: 9,
      question: "What is the purpose of handling outliers in EDA?",
      options: [
        "To make data look better",
        "To prevent bias in analysis and modeling",
        "To increase dataset size",
        "To speed up computation"
      ],
      correctAnswer: 1,
      explanation: "Handling outliers prevents them from skewing statistical analyses and biasing machine learning models, ensuring more reliable results.",
    },
    {
      id: 10,
      question: "What is a violin plot?",
      options: [
        "A type of line chart",
        "A combination of box plot and kernel density estimation",
        "A scatter plot variant",
        "A bar chart variant"
      ],
      correctAnswer: 1,
      explanation: "A violin plot combines a box plot with a kernel density plot, showing the full distribution shape along with summary statistics.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
        

        {/* 1. What is EDA? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Eye className="h-5 w-5 text-primary" />
            1. What is Exploratory Data Analysis (EDA)?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics, often using visual methods. It's a critical first step before any formal modeling.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Raw Data</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">EDA</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">Insights</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Understanding data → Better models</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Why is EDA important?</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Understand data structure and patterns</li>
                    <li>Detect anomalies and outliers</li>
                    <li>Identify relationships between variables</li>
                    <li>Formulate hypotheses</li>
                    <li>Guide feature engineering decisions</li>
                    <li>Choose appropriate models</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. EDA Framework */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <FileSearch className="h-5 w-5 text-primary" />
            2. The EDA Framework
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">1</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Understanding</h4>
                    <p className="text-xs text-muted-foreground">Explore structure, size, and types</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">2</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Cleaning</h4>
                    <p className="text-xs text-muted-foreground">Handle missing values, duplicates, errors</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">3</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Univariate Analysis</h4>
                    <p className="text-xs text-muted-foreground">Analyze each variable individually</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">4</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Bivariate Analysis</h4>
                    <p className="text-xs text-muted-foreground">Analyze relationships between pairs</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">5</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Multivariate Analysis</h4>
                    <p className="text-xs text-muted-foreground">Explore complex interactions</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">6</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Visualization</h4>
                    <p className="text-xs text-muted-foreground">Create informative plots and charts</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">7</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Hypothesis Testing</h4>
                    <p className="text-xs text-muted-foreground">Formulate and test assumptions</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">8</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Documentation</h4>
                    <p className="text-xs text-muted-foreground">Record findings and insights</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Data Understanding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            3. Data Understanding
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Structure</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Shape (rows × columns)</li>
                <li>Column names and types</li>
                <li>Memory usage</li>
                <li>Data quality flags</li>
              </ul>
              <div className="bg-muted p-2 rounded mt-2">
                <code className="text-xs">df.info()</code>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Quality</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Missing values count</li>
                <li>Duplicate rows</li>
                <li>Data types consistency</li>
                <li>Unique values validation</li>
              </ul>
              <div className="bg-muted p-2 rounded mt-2">
                <code className="text-xs">df.isnull().sum()</code>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Summary</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Descriptive statistics</li>
                <li>Central tendency</li>
                <li>Dispersion measures</li>
                <li>Distribution shape</li>
              </ul>
              <div className="bg-muted p-2 rounded mt-2">
                <code className="text-xs">df.describe()</code>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Univariate Analysis */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <ChartBar className="h-5 w-5 text-primary" />
            4. Univariate Analysis
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Numerical Variables</h4>
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-foreground text-sm">Visualizations</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium text-foreground">Histogram</span>: Distribution shape</li>
                    <li><span className="font-medium text-foreground">Box Plot</span>: Quartiles, outliers</li>
                    <li><span className="font-medium text-foreground">Violin Plot</span>: Full distribution</li>
                    <li><span className="font-medium text-foreground">QQ Plot</span>: Normality check</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Statistics</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Mean, Median, Mode</li>
                    <li>Std Dev, Variance, IQR</li>
                    <li>Skewness, Kurtosis</li>
                    <li>Min, Max, Range</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Categorical Variables</h4>
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-foreground text-sm">Visualizations</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium text-foreground">Bar Chart</span>: Frequency counts</li>
                    <li><span className="font-medium text-foreground">Pie Chart</span>: Proportions</li>
                    <li><span className="font-medium text-foreground">Count Plot</span>: Distribution</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Statistics</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Frequency distribution</li>
                    <li>Mode</li>
                    <li>Unique values count</li>
                    <li>Proportion/Percentage</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Bivariate Analysis */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <ChartScatter className="h-5 w-5 text-primary" />
            5. Bivariate Analysis
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Numerical vs Numerical</h4>
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-foreground text-sm">Visualizations</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium text-foreground">Scatter Plot</span>: Relationships</li>
                    <li><span className="font-medium text-foreground">Heatmap</span>: Correlation matrix</li>
                    <li><span className="font-medium text-foreground">Pair Plot</span>: Multiple variables</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Correlation Tests</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Pearson (linear)</li>
                    <li>Spearman (monotonic)</li>
                    <li>Kendall (ordinal)</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Categorical vs Numerical</h4>
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-foreground text-sm">Visualizations</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium text-foreground">Box Plot</span>: Distribution by category</li>
                    <li><span className="font-medium text-foreground">Violin Plot</span>: Full distribution</li>
                    <li><span className="font-medium text-foreground">Bar Plot</span>: Mean/median values</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Statistical Tests</h5>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>ANOVA</li>
                    <li>T-test</li>
                    <li>Chi-square test</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">Categorical vs Categorical</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-medium text-foreground text-sm">Visualizations</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li><span className="font-medium text-foreground">Stacked Bar Chart</span>: Proportions</li>
                  <li><span className="font-medium text-foreground">Mosaic Plot</span>: Relationships</li>
                  <li><span className="font-medium text-foreground">Heatmap</span>: Frequency matrix</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium text-foreground text-sm">Statistical Tests</h5>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Chi-square test</li>
                  <li>Cramer's V</li>
                  <li>Mutual information</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Multivariate Analysis */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            6. Multivariate Analysis
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Visualization Techniques</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Parallel Coordinates</span></li>
                <li><span className="font-medium text-foreground">Radar Charts</span></li>
                <li><span className="font-medium text-foreground">3D Scatter Plots</span></li>
                <li><span className="font-medium text-foreground">Facet Grids</span></li>
                <li><span className="font-medium text-foreground">Hierarchical Clustering</span></li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Dimensionality Reduction</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">PCA</span>: Principal components</li>
                <li><span className="font-medium text-foreground">t-SNE</span>: Non-linear projection</li>
                <li><span className="font-medium text-foreground">UMAP</span>: Uniform manifold projection</li>
                <li><span className="font-medium text-foreground">MDS</span>: Multidimensional scaling</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Applications</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Identifying clusters</li>
                <li>Understanding complex interactions</li>
                <li>Feature selection insights</li>
                <li>Anomaly detection</li>
                <li>Pattern discovery</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 7. Outlier Detection */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <ScatterChart className="h-5 w-5 text-primary" />
            7. Outlier Detection in EDA
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Statistical Methods</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Z-Score</p>
                  <p className="text-muted-foreground">Outlier if |z| &gt; 3</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">IQR Method</p>
                  <p className="text-muted-foreground">Q1 - 1.5×IQR or Q3 + 1.5×IQR</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">MAD</p>
                  <p className="text-muted-foreground">Median Absolute Deviation</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Visualization Methods</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Box Plots</p>
                  <p className="text-muted-foreground">Visual outlier detection</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Scatter Plots</p>
                  <p className="text-muted-foreground">Identify extreme points</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Histograms</p>
                  <p className="text-muted-foreground">Observe tails and gaps</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Advanced Methods</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Isolation Forest</p>
                  <p className="text-muted-foreground">Tree-based isolation</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">DBSCAN</p>
                  <p className="text-muted-foreground">Density-based clustering</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">LOF</p>
                  <p className="text-muted-foreground">Local Outlier Factor</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Visualization Types */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Eye className="h-5 w-5 text-primary" />
            8. Visualization Types for EDA
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Distribution Plots</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Histogram</p>
                  <p className="text-muted-foreground">Frequency distribution</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">KDE Plot</p>
                  <p className="text-muted-foreground">Smooth density</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Box Plot</p>
                  <p className="text-muted-foreground">Summary statistics</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Violin Plot</p>
                  <p className="text-muted-foreground">Full distribution</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Relationship Plots</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Scatter Plot</p>
                  <p className="text-muted-foreground">Two variables</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Pair Plot</p>
                  <p className="text-muted-foreground">Multiple variables</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Heatmap</p>
                  <p className="text-muted-foreground">Correlation matrix</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Joint Plot</p>
                  <p className="text-muted-foreground">Distribution + scatter</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Categorical Plots</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Bar Chart</p>
                  <p className="text-muted-foreground">Counts/proportions</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Pie Chart</p>
                  <p className="text-muted-foreground">Proportions</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Count Plot</p>
                  <p className="text-muted-foreground">Frequency</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Mosaic Plot</p>
                  <p className="text-muted-foreground">Categorical relationships</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Advanced Plots</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Parallel Coordinates</p>
                  <p className="text-muted-foreground">Multivariate patterns</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Radar Chart</p>
                  <p className="text-muted-foreground">Multi-dimensional</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">3D Scatter</p>
                  <p className="text-muted-foreground">Three variables</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-medium text-foreground">Word Cloud</p>
                  <p className="text-muted-foreground">Text data</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Common Pitfalls */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            9. Common Pitfalls in EDA
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Confirmation Bias</h4>
                  <p className="text-sm text-muted-foreground">
                    Looking for patterns that confirm preconceived notions. Always consider alternative explanations.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Over-plotting</h4>
                  <p className="text-sm text-muted-foreground">
                    Too many points in scatter plots. Use transparency, sampling, or hexbin plots.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Ignoring Data Quality</h4>
                  <p className="text-sm text-muted-foreground">
                    Not checking for missing values, duplicates, or errors before analysis.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Overlooking Outliers</h4>
                  <p className="text-sm text-muted-foreground">
                    Not properly handling outliers that can skew analysis and models.
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
            10. Best Practices for EDA
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Start with the big picture</h4>
                <p className="text-sm text-muted-foreground">Begin with summary statistics and high-level visualizations</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use multiple visualizations</h4>
                <p className="text-sm text-muted-foreground">Different plots reveal different aspects of the data</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Document your findings</h4>
                <p className="text-sm text-muted-foreground">Record insights, hypotheses, and decisions for future reference</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Iterate and refine</h4>
                <p className="text-sm text-muted-foreground">EDA is iterative — each insight leads to new questions and analyses</p>
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
            See EDA techniques in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Exploratory Data Analysis Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}