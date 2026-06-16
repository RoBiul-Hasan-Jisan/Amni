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
  ArrowUpRight
} from "lucide-react";

export default function RegressionPage() {
  const result = getSubtopicBySlug("machine-learning", "regression");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-linear",
      label: "Linear Regression",
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(200, 1) * 10
y = 3 * X.squeeze() + 5 + np.random.randn(200) * 5

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"Coefficient (Slope): {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

print("\\n" + "=" * 50)
print("EVALUATION METRICS")
print("=" * 50)
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.3f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.3f}")
print(f"Train R² Score: {r2_score(y_train, y_pred_train):.3f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
print("\\nPlot saved as 'linear_regression.png'")`,
    },
    {
      language: "python-polynomial",
      label: "Polynomial Regression",
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.random.randn(200, 1) * 2
y = 0.5 * X.squeeze()**2 - 0.3 * X.squeeze()**3 + 2 * X.squeeze() + 3 + np.random.randn(200) * 1.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare different degrees
degrees = [1, 2, 3, 4, 5]
models = []
train_scores = []
test_scores = []

plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees):
    # Create polynomial regression pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    models.append(model)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Store scores
    train_scores.append(r2_score(y_train, y_train_pred))
    test_scores.append(r2_score(y_test, y_test_pred))
    
    # Plot
    plt.subplot(2, 3, i+1)
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    plt.scatter(X_train, y_train, alpha=0.5, s=10, label='Train')
    plt.scatter(X_test, y_test, alpha=0.5, s=10, label='Test')
    plt.plot(X_plot, y_plot, color='red', label=f'Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Degree {degree}\\nTrain R²: {r2_score(y_train, y_train_pred):.3f}')
    plt.legend()

plt.tight_layout()
plt.savefig('polynomial_regression.png', dpi=300, bbox_inches='tight')

# Summary
print("=" * 50)
print("POLYNOMIAL REGRESSION COMPARISON")
print("=" * 50)
for degree, train, test in zip(degrees, train_scores, test_scores):
    print(f"Degree {degree}: Train R²={train:.3f}, Test R²={test:.3f}")

print("\\nPlot saved as 'polynomial_regression.png'")`,
    },
    {
      language: "python-ridge",
      label: "Ridge Regression",
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate data with many features
np.random.seed(42)
n_samples = 200
n_features = 20
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples) * 2 + X[:, 0] * 3 + X[:, 1] * 2 + np.random.randn(n_samples) * 0.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores = []
test_scores = []
coeffs = []

print("=" * 50)
print("RIDGE REGRESSION COMPARISON")
print("=" * 50)

for alpha in alphas:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_scores.append(train_r2)
    test_scores.append(test_r2)
    coeffs.append(model.named_steps['ridge'].coef_)
    
    print(f"Alpha={alpha:.3f}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")

# Find best alpha using cross-validation
ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=5)
ridge_cv.fit(X_train, y_train)
print(f"\\nBest alpha from CV: {ridge_cv.alpha_}")

# Visualize coefficient paths
plt.figure(figsize=(10, 6))
for i in range(n_features):
    coeff_path = [coeff[i] for coeff in coeffs]
    plt.plot(alphas, coeff_path, label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Paths')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ridge_paths.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'ridge_paths.png'")`,
    },
    {
      language: "python-lasso",
      label: "Lasso Regression",
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate data with redundant features
np.random.seed(42)
n_samples = 200
n_features = 20
X = np.random.randn(n_samples, n_features)
# Only first 5 features are relevant
true_coeffs = np.zeros(n_features)
true_coeffs[:5] = [3, 2, 1.5, 1, 0.5]
y = X @ true_coeffs + np.random.randn(n_samples) * 0.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Test different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores = []
test_scores = []
coeffs = []
selected_features = []

print("=" * 50)
print("LASSO REGRESSION COMPARISON")
print("=" * 50)

for alpha in alphas:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=alpha, max_iter=10000))
    ])
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_scores.append(train_r2)
    test_scores.append(test_r2)
    coeffs.append(model.named_steps['lasso'].coef_)
    
    n_selected = np.sum(np.abs(model.named_steps['lasso'].coef_) > 0.001)
    selected_features.append(n_selected)
    
    print(f"Alpha={alpha:.3f}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Selected={n_selected}")

# Find best alpha using cross-validation
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"\\nBest alpha from CV: {lasso_cv.alpha_:.3f}")
print(f"Number of features selected: {np.sum(np.abs(lasso_cv.coef_) > 0.001)}")

# Visualize coefficient paths
plt.figure(figsize=(10, 6))
for i in range(n_features):
    coeff_path = [coeff[i] for coeff in coeffs]
    if np.any(np.abs(coeff_path) > 0.001):
        plt.plot(alphas, coeff_path, label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Paths')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lasso_paths.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'lasso_paths.png'")`,
    },
    {
      language: "python-tree",
      label: "Decision Tree & Random Forest",
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.random.randn(300, 5)
y = np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 3) + X[:, 2]**2 + np.random.randn(300) * 0.3

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 50)
print("DECISION TREE VS RANDOM FOREST")
print("=" * 50)

# Decision Tree with different depths
depths = [2, 5, 10, 15, 20, None]
dt_train_scores = []
dt_test_scores = []

print("\\nDecision Tree:")
for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    dt_train_scores.append(train_r2)
    dt_test_scores.append(test_r2)
    
    print(f"  Depth={depth}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")

# Random Forest with different n_estimators
n_estimators = [10, 50, 100, 200, 500]
rf_train_scores = []
rf_test_scores = []

print("\\nRandom Forest:")
for n in n_estimators:
    rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    rf_train_scores.append(train_r2)
    rf_test_scores.append(test_r2)
    
    print(f"  n_estimators={n}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")

# Feature importance from Random Forest
rf_best = RandomForestRegressor(n_estimators=100, random_state=42)
rf_best.fit(X_train, y_train)

print("\\nFeature Importance:")
for i, imp in enumerate(rf_best.feature_importances_):
    print(f"  Feature {i+1}: {imp:.3f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree results
ax1.plot(depths[:-1], dt_train_scores[:-1], marker='o', label='Train')
ax1.plot(depths[:-1], dt_test_scores[:-1], marker='o', label='Test')
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('R² Score')
ax1.set_title('Decision Tree Performance')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Random Forest results
ax2.plot(n_estimators, rf_train_scores, marker='o', label='Train')
ax2.plot(n_estimators, rf_test_scores, marker='o', label='Test')
ax2.set_xlabel('Number of Estimators')
ax2.set_ylabel('R² Score')
ax2.set_title('Random Forest Performance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tree_regression.png', dpi=300, bbox_inches='tight')
print("\\nPlot saved as 'tree_regression.png'")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Regression in Machine Learning?",
      options: [
        "Predicting categorical values",
        "Predicting continuous values",
        "Clustering data points",
        "Reducing dimensionality"
      ],
      correctAnswer: 1,
      explanation: "Regression is a supervised learning technique used to predict continuous numerical values (e.g., house prices, temperature).",
    },
    {
      id: 2,
      question: "What is the difference between Linear and Polynomial Regression?",
      options: [
        "They are the same",
        "Linear assumes a straight-line relationship, Polynomial can capture curves",
        "Polynomial is for classification",
        "Linear is for multiple variables only"
      ],
      correctAnswer: 1,
      explanation: "Linear regression assumes a linear relationship between features and target, while polynomial regression can capture non-linear relationships using higher-degree terms.",
    },
    {
      id: 3,
      question: "What is the purpose of R² Score?",
      options: [
        "To measure model speed",
        "To measure how well the model explains the variance in the target variable",
        "To count the number of features",
        "To measure data size"
      ],
      correctAnswer: 1,
      explanation: "R² Score (Coefficient of Determination) measures the proportion of variance in the target variable that is explained by the model.",
    },
    {
      id: 4,
      question: "What is the difference between L1 and L2 Regularization?",
      options: [
        "L1 is for classification, L2 is for regression",
        "L1 (Lasso) can shrink coefficients to zero, L2 (Ridge) shrinks them towards zero",
        "L2 is for linear models only",
        "They are the same"
      ],
      correctAnswer: 1,
      explanation: "L1 regularization (Lasso) adds absolute value penalty and can perform feature selection by shrinking some coefficients to zero. L2 regularization (Ridge) adds squared penalty and shrinks coefficients towards zero.",
    },
    {
      id: 5,
      question: "What is the Bias-Variance Tradeoff in Regression?",
      options: [
        "A tradeoff between training and test data",
        "A tradeoff between model complexity and generalization",
        "A tradeoff between features and samples",
        "A tradeoff between accuracy and speed"
      ],
      correctAnswer: 1,
      explanation: "The bias-variance tradeoff balances model complexity (variance) against model simplicity (bias) to achieve optimal generalization.",
    },
    {
      id: 6,
      question: "What is RMSE (Root Mean Squared Error)?",
      options: [
        "A measure of model accuracy",
        "A measure of how spread out the errors are",
        "A measure of model complexity",
        "A measure of feature importance"
      ],
      correctAnswer: 0,
      explanation: "RMSE is a measure of model accuracy that calculates the square root of the average squared differences between predictions and actual values.",
    },
    {
      id: 7,
      question: "When should you use Random Forest over Linear Regression?",
      options: [
        "When data is linear",
        "When there are complex non-linear patterns and many features",
        "When you need interpretability",
        "When you have small dataset"
      ],
      correctAnswer: 1,
      explanation: "Random Forest is better for complex, non-linear patterns with many features, while linear regression is simpler and more interpretable for linear relationships.",
    },
    {
      id: 8,
      question: "What is the purpose of Cross-Validation in Regression?",
      options: [
        "To increase training speed",
        "To better evaluate model performance and prevent overfitting",
        "To reduce the number of features",
        "To clean the data"
      ],
      correctAnswer: 1,
      explanation: "Cross-validation provides a more robust evaluation by testing the model on multiple validation sets, helping to detect overfitting.",
    },
    {
      id: 9,
      question: "What is the advantage of using a Pipeline in scikit-learn?",
      options: [
        "It makes the code run faster",
        "It ensures consistent preprocessing and prevents data leakage",
        "It automatically selects the best model",
        "It reduces memory usage"
      ],
      correctAnswer: 1,
      explanation: "Pipelines ensure consistent preprocessing between training and testing, preventing data leakage and making the code cleaner and more reproducible.",
    },
    {
      id: 10,
      question: "Which metric is NOT appropriate for regression evaluation?",
      options: [
        "Mean Squared Error (MSE)",
        "R² Score",
        "Accuracy",
        "Mean Absolute Error (MAE)"
      ],
      correctAnswer: 2,
      explanation: "Accuracy is a classification metric, not appropriate for regression. The others (MSE, R², MAE) are regression metrics.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
      

        {/* 1. What is Regression? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Ruler className="h-5 w-5 text-primary" />
            1. What is Regression?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Regression is a supervised learning technique used to predict <span className="font-semibold text-foreground">continuous numerical values</span> based on input features. It models the relationship between variables.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Features (X)</span>
                  <span className="text-primary">→</span>
                  <span className="text-foreground font-medium">Regression Model</span>
                  <span className="text-primary">→</span>
                  <span className="text-green-500 font-medium">Continuous Output</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Predicting continuous values</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Common Use Cases</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>House price prediction</li>
                    <li>Stock market forecasting</li>
                    <li>Sales and revenue prediction</li>
                    <li>Temperature forecasting</li>
                    <li>Customer lifetime value</li>
                    <li>Risk assessment</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Types of Regression */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            2. Types of Regression
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Linear Regression</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Simple Linear</span>: One feature</li>
                <li><span className="font-medium text-foreground">Multiple Linear</span>: Multiple features</li>
                <li><span className="font-medium text-foreground">Assumption</span>: Linear relationship</li>
                <li><span className="font-medium text-foreground">Pros</span>: Simple, interpretable</li>
                <li><span className="font-medium text-foreground">Cons</span>: Limited to linear patterns</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Polynomial Regression</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Degree</span>: Controls curve flexibility</li>
                <li><span className="font-medium text-foreground">Use</span>: Non-linear relationships</li>
                <li><span className="font-medium text-foreground">Pros</span>: Captures curves</li>
                <li><span className="font-medium text-foreground">Cons</span>: Risk of overfitting</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Regularized Regression</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Ridge (L2)</span>: Shrinks coefficients</li>
                <li><span className="font-medium text-foreground">Lasso (L1)</span>: Feature selection</li>
                <li><span className="font-medium text-foreground">Elastic Net</span>: Combines L1 + L2</li>
                <li><span className="font-medium text-foreground">Use</span>: High-dimensional data</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Tree-Based Regression</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Decision Tree</span>: Tree-based predictions</li>
                <li><span className="font-medium text-foreground">Random Forest</span>: Ensemble of trees</li>
                <li><span className="font-medium text-foreground">Gradient Boosting</span>: Sequential trees</li>
                <li><span className="font-medium text-foreground">XGBoost</span>: Optimized boosting</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3. Key Concepts */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            3. Key Concepts in Regression
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Regression Assumptions</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Linearity of relationship</li>
                <li>Independence of errors</li>
                <li>Homoscedasticity</li>
                <li>Normality of residuals</li>
                <li>No multicollinearity</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Loss Functions</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">MSE</span>: Mean Squared Error</li>
                <li><span className="font-medium text-foreground">MAE</span>: Mean Absolute Error</li>
                <li><span className="font-medium text-foreground">Huber</span>: Combines MSE + MAE</li>
                <li><span className="font-medium text-foreground">Quantile</span>: For quantile regression</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Evaluation Metrics</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">R² Score</span>: Variance explained</li>
                <li><span className="font-medium text-foreground">RMSE</span>: Root Mean Squared Error</li>
                <li><span className="font-medium text-foreground">MAE</span>: Mean Absolute Error</li>
                <li><span className="font-medium text-foreground">MAPE</span>: Mean Absolute Percentage Error</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 4. Linear Regression in Detail */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <LineChart className="h-5 w-5 text-primary" />
            4. Linear Regression in Detail
          </h2>

          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Formula</h3>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-lg">y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε</p>
                <p className="text-xs text-muted-foreground mt-2">
                  y = target, β₀ = intercept, βᵢ = coefficients, xᵢ = features, ε = error
                </p>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Training Methods</h4>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li><span className="font-medium text-foreground">Normal Equation</span>: Closed-form solution</li>
                  <li><span className="font-medium text-foreground">Gradient Descent</span>: Iterative optimization</li>
                  <li><span className="font-medium text-foreground">SVD</span>: Singular Value Decomposition</li>
                </ul>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Pros</h4>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Simple and fast</li>
                  <li>Highly interpretable</li>
                  <li>No hyperparameters</li>
                  <li>Scalable</li>
                </ul>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Cons</h4>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Assumes linearity</li>
                  <li>Sensitive to outliers</li>
                  <li>Multicollinearity issues</li>
                  <li>Cannot capture interactions</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Regularized Regression */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            5. Regularized Regression
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Ridge Regression (L2)</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-sm">Loss = MSE + αΣβ²</p>
              </div>
              <div className="mt-2 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Features</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Shrinks coefficients towards zero</li>
                    <li>Reduces overfitting</li>
                    <li>Handles multicollinearity</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">When to use</p>
                  <p className="text-muted-foreground">When many features are correlated or you want to keep all features</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Lasso Regression (L1)</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-sm">Loss = MSE + αΣ|β|</p>
              </div>
              <div className="mt-2 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Features</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Can shrink coefficients to zero</li>
                    <li>Performs feature selection</li>
                    <li>Sparse solution</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">When to use</p>
                  <p className="text-muted-foreground">When you suspect many features are irrelevant</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 md:col-span-2">
              <h4 className="font-semibold text-foreground mb-2">Comparison</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left p-2">Feature</th>
                      <th className="text-left p-2">Ridge (L2)</th>
                      <th className="text-left p-2">Lasso (L1)</th>
                      <th className="text-left p-2">Elastic Net</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border">
                      <td className="p-2">Penalty</td>
                      <td className="p-2">Σβ²</td>
                      <td className="p-2">Σ|β|</td>
                      <td className="p-2">αΣ|β| + (1-α)Σβ²</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="p-2">Feature Selection</td>
                      <td className="p-2">No</td>
                      <td className="p-2">Yes</td>
                      <td className="p-2">Yes</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="p-2">Sparse Solution</td>
                      <td className="p-2">No</td>
                      <td className="p-2">Yes</td>
                      <td className="p-2">Yes</td>
                    </tr>
                    <tr>
                      <td className="p-2">Handles Correlations</td>
                      <td className="p-2">Good</td>
                      <td className="p-2">Poor</td>
                      <td className="p-2">Good</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Tree-Based Regression */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <TreePine className="h-5 w-5 text-primary" />
            6. Tree-Based Regression
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Decision Tree</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">How it works</p>
                  <p className="text-muted-foreground">Splits data recursively based on features</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <p className="text-muted-foreground">Interpretable, handles non-linearity</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <p className="text-muted-foreground">Prone to overfitting</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Random Forest</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">How it works</p>
                  <p className="text-muted-foreground">Ensemble of decision trees</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <p className="text-muted-foreground">Reduces overfitting, feature importance</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <p className="text-muted-foreground">Less interpretable, slower</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Gradient Boosting</h4>
              <div className="space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">How it works</p>
                  <p className="text-muted-foreground">Sequentially adds weak learners</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <p className="text-muted-foreground">High accuracy, handles various data</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <p className="text-muted-foreground">Computationally expensive</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Evaluation Metrics */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Gauge className="h-5 w-5 text-primary" />
            7. Evaluation Metrics
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Error-Based Metrics</h4>
              <div className="space-y-3">
                <div>
                  <p className="font-medium text-foreground text-sm">MSE (Mean Squared Error)</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">MSE = (1/n)Σ(yᵢ - ŷᵢ)²</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Penalizes large errors, sensitive to outliers</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">MAE (Mean Absolute Error)</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">MAE = (1/n)Σ|yᵢ - ŷᵢ|</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Less sensitive to outliers, easy to interpret</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">RMSE (Root Mean Squared Error)</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">RMSE = √MSE</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Interpretable in original units</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Variance-Based Metrics</h4>
              <div className="space-y-3">
                <div>
                  <p className="font-medium text-foreground text-sm">R² Score</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">R² = 1 - (SS_res / SS_tot)</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Proportion of variance explained (0 to 1)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Adjusted R²</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">Adj R² = 1 - (1-R²)(n-1)/(n-k-1)</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Penalizes adding unnecessary features</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">MAPE</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">MAPE = (1/n)Σ|(yᵢ - ŷᵢ)/yᵢ| × 100%</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Percentage error, useful for business</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Model Selection Guide */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            8. Model Selection Guide
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">When to use each model</h4>
                <div className="space-y-2 text-xs">
                  <div className="bg-muted p-2 rounded">
                    <p className="font-medium text-foreground">Linear Regression</p>
                    <p className="text-muted-foreground">Simple linear relationships, small datasets</p>
                  </div>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-medium text-foreground">Ridge/Lasso</p>
                    <p className="text-muted-foreground">Many features, risk of overfitting</p>
                  </div>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-medium text-foreground">Decision Tree</p>
                    <p className="text-muted-foreground">Non-linear patterns, need interpretability</p>
                  </div>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-medium text-foreground">Random Forest</p>
                    <p className="text-muted-foreground">Complex patterns, high accuracy needed</p>
                  </div>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-medium text-foreground">XGBoost</p>
                    <p className="text-muted-foreground">Competitions, state-of-the-art results</p>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Decision Flow</h4>
                <div className="space-y-2 text-xs">
                  <div className="flex items-start gap-2">
                    <span className="text-primary">1.</span>
                    <p>Is relationship linear? <span className="text-primary">→ Linear/Polynomial</span></p>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-primary">2.</span>
                    <p>Many features? <span className="text-primary">→ Regularization</span></p>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-primary">3.</span>
                    <p>Complex non-linear patterns? <span className="text-primary">→ Tree-based</span></p>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-primary">4.</span>
                    <p>Need feature importance? <span className="text-primary">→ Random Forest</span></p>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-primary">5.</span>
                    <p>Maximum accuracy? <span className="text-primary">→ XGBoost</span></p>
                  </div>
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
                  <h4 className="font-semibold text-foreground mb-1">Overfitting</h4>
                  <p className="text-sm text-muted-foreground">
                    Using too complex models. Use regularization, cross-validation, and simpler models.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Multicollinearity</h4>
                  <p className="text-sm text-muted-foreground">
                    Highly correlated features. Use VIF, remove correlated features, or use regularization.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Outliers</h4>
                  <p className="text-sm text-muted-foreground">
                    Outliers can skew results. Detect and handle them using robust methods.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Using test data in preprocessing. Always split before any preprocessing.
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
                <h4 className="font-semibold text-foreground mb-1">Start Simple</h4>
                <p className="text-sm text-muted-foreground">Begin with linear regression, then add complexity</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Check Assumptions</h4>
                <p className="text-sm text-muted-foreground">Validate linearity, normality, and homoscedasticity</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Feature Engineering</h4>
                <p className="text-sm text-muted-foreground">Create meaningful features, handle interactions</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Cross-Validation</h4>
                <p className="text-sm text-muted-foreground">Use CV for reliable performance estimation</p>
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
            See regression algorithms in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Regression Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}