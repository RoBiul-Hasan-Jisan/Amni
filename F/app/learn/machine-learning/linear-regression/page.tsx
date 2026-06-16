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
 FunctionSquare,
  Minus,
  Plus,
  Equal,
  X,
  Axis3D,
  ArrowRight
} from "lucide-react";

export default function LinearRegressionPage() {
  const result = getSubtopicBySlug("machine-learning", "linear-regression");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-simple",
      label: "Simple Linear Regression",
      code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("=" * 50)
print("SIMPLE LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"Coefficient (Slope): {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

print("\\nEquation: y = {:.3f}x + {:.3f}".format(
    model.coef_[0], model.intercept_
))

print("\\n" + "=" * 50)
print("EVALUATION METRICS")
print("=" * 50)
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.3f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.3f}")
print(f"Train R² Score: {r2_score(y_train, y_train_pred):.3f}")
print(f"Test R² Score: {r2_score(y_test, y_test_pred):.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_simple.png', dpi=300, bbox_inches='tight')
print("\\nPlot saved as 'linear_regression_simple.png'")`,
    },
    {
      language: "python-multiple",
      label: "Multiple Linear Regression",
      code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data with multiple features
np.random.seed(42)
n_samples = 300
n_features = 5

X = np.random.randn(n_samples, n_features)
true_coeffs = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
intercept = 10
y = intercept + X @ true_coeffs + np.random.randn(n_samples) * 2

# Create DataFrame
feature_names = [f'Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

print("=" * 50)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 50)
print("Dataset shape:", df.shape)
print("\\nFeature names:", feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\\n" + "=" * 50)
print("COEFFICIENTS")
print("=" * 50)
print(f"Intercept: {model.intercept_:.3f}")
print("\\nFeature coefficients:")
for feature, coeff in zip(feature_names, model.coef_):
    print(f"  {feature}: {coeff:.3f}")

print("\\n" + "=" * 50)
print("EVALUATION METRICS")
print("=" * 50)
print(f"Train R² Score: {r2_score(y_train, y_train_pred):.3f}")
print(f"Test R² Score: {r2_score(y_test, y_test_pred):.3f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}")

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_names, model.coef_)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients in Multiple Linear Regression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression_coefficients.png', dpi=300, bbox_inches='tight')
print("\\nPlot saved as 'linear_regression_coefficients.png'")`,
    },
    {
      language: "python-assumptions",
      label: "Assumption Checking",
      code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns

# Generate data
np.random.seed(42)
X = np.random.randn(200, 1) * 10
y = 2.5 * X.squeeze() + 3 + np.random.randn(200) * 4

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
residuals = y_train - y_pred

print("=" * 50)
print("LINEAR REGRESSION ASSUMPTION CHECKS")
print("=" * 50)

# 1. Linearity
print("\\n1. Linearity Check:")
print("   Visual inspection of residual plot below")

# 2. Normality of Residuals
print("\\n2. Normality of Residuals:")
shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])
print(f"   Shapiro-Wilk Test: statistic={shapiro_stat:.3f}, p-value={shapiro_p:.3f}")
if shapiro_p > 0.05:
    print("   ✓ Residuals appear normally distributed (p > 0.05)")
else:
    print("   ✗ Residuals may not be normally distributed (p < 0.05)")

# 3. Homoscedasticity (constant variance)
print("\\n3. Homoscedasticity Check:")
# Breusch-Pagan test (simplified)
_, bp_p = stats.chi2_contingency(pd.crosstab(
    pd.qcut(y_pred, q=4),
    pd.qcut(residuals, q=4)
)[:2])
print(f"   Visual inspection of residual plot below")

# 4. Independence of Errors
print("\\n4. Independence of Errors:")
durbin_watson = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"   Durbin-Watson: {durbin_watson:.3f}")
if 1.5 < durbin_watson < 2.5:
    print("   ✓ No autocorrelation detected")
else:
    print("   ✗ Possible autocorrelation in residuals")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs Fitted
axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted (Linearity Check)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normality Check)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Histogram of Residuals
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram of Residuals')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scale-Location Plot
axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Fitted Values')
axes[1, 1].set_ylabel('√|Residuals|')
axes[1, 1].set_title('Scale-Location Plot (Homoscedasticity)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_assumptions.png', dpi=300, bbox_inches='tight')
print("\\nAssumption plots saved as 'regression_assumptions.png'")`,
    },
    {
      language: "python-gradient",
      label: "Gradient Descent Implementation",
      code: `import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Calculate gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.6f}")
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1) * 5
y = 2.5 * X.squeeze() + 4 + np.random.randn(100) * 2

print("=" * 50)
print("GRADIENT DESCENT IMPLEMENTATION")
print("=" * 50)

# Train using custom GD
model_gd = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model_gd.fit(X, y)

print("\\n" + "=" * 50)
print("FINAL PARAMETERS")
print("=" * 50)
print(f"Weight: {model_gd.weights[0]:.3f}")
print(f"Bias: {model_gd.bias:.3f}")

# Compare with sklearn
from sklearn.linear_model import LinearRegression
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

print("\\n" + "=" * 50)
print("COMPARISON WITH SKLEARN")
print("=" * 50)
print(f"Custom GD: weight={model_gd.weights[0]:.3f}, bias={model_gd.bias:.3f}")
print(f"Sklearn: weight={model_sklearn.coef_[0]:.3f}, bias={model_sklearn.intercept_:.3f}")

# Visualize loss convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model_gd.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Convergence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(X_line, model_gd.predict(X_line), color='red', label='GD Fit')
plt.plot(X_line, model_sklearn.predict(X_line), color='green', 
         linestyle='--', label='Sklearn Fit', alpha=0.7)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=300, bbox_inches='tight')
print("\\nPlots saved as 'gradient_descent.png'")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Linear Regression?",
      options: [
        "A classification algorithm",
        "A technique for predicting continuous values using a linear relationship",
        "A clustering algorithm",
        "A dimensionality reduction technique"
      ],
      correctAnswer: 1,
      explanation: "Linear regression models the relationship between a dependent variable and one or more independent variables using a linear equation.",
    },
    {
      id: 2,
      question: "What does the coefficient (slope) represent in linear regression?",
      options: [
        "The starting value when x=0",
        "The change in y for a one-unit change in x",
        "The correlation between x and y",
        "The variance of y"
      ],
      correctAnswer: 1,
      explanation: "The coefficient (slope) represents the change in the target variable (y) for a one-unit change in the feature (x).",
    },
    {
      id: 3,
      question: "What is the equation for simple linear regression?",
      options: [
        "y = ax² + bx + c",
        "y = mx + b",
        "y = eˣ",
        "y = log(x)"
      ],
      correctAnswer: 1,
      explanation: "Simple linear regression is represented by the equation y = mx + b, where m is the slope and b is the intercept.",
    },
    {
      id: 4,
      question: "What is the difference between Simple and Multiple Linear Regression?",
      options: [
        "Simple uses one independent variable, Multiple uses multiple independent variables",
        "Simple uses categorical data, Multiple uses numerical data",
        "Simple is for classification, Multiple is for regression",
        "They are the same"
      ],
      correctAnswer: 0,
      explanation: "Simple linear regression has one independent variable, while multiple linear regression has two or more independent variables.",
    },
    {
      id: 5,
      question: "What is the purpose of the Ordinary Least Squares (OLS) method?",
      options: [
        "To find the maximum likelihood estimate",
        "To minimize the sum of squared residuals",
        "To maximize the R² score",
        "To reduce overfitting"
      ],
      correctAnswer: 1,
      explanation: "OLS minimizes the sum of squared differences between observed and predicted values, finding the best-fitting line.",
    },
    {
      id: 6,
      question: "What is an assumption of linear regression?",
      options: [
        "Non-linear relationship between variables",
        "Normality of residuals",
        "Categorical independent variables only",
        "High multicollinearity"
      ],
      correctAnswer: 1,
      explanation: "Linear regression assumes that residuals are normally distributed, among other assumptions like linearity and homoscedasticity.",
    },
    {
      id: 7,
      question: "What is R² Score in linear regression?",
      options: [
        "The correlation coefficient",
        "The proportion of variance explained by the model",
        "The mean squared error",
        "The intercept term"
      ],
      correctAnswer: 1,
      explanation: "R² Score (Coefficient of Determination) measures the proportion of variance in the target variable explained by the model.",
    },
    {
      id: 8,
      question: "What is the problem with multicollinearity in linear regression?",
      options: [
        "It makes the model faster",
        "It makes coefficient estimates unstable and hard to interpret",
        "It increases the R² score",
        "It reduces the number of features"
      ],
      correctAnswer: 1,
      explanation: "Multicollinearity (high correlation between features) makes coefficient estimates unstable and difficult to interpret.",
    },
    {
      id: 9,
      question: "What is gradient descent in the context of linear regression?",
      options: [
        "A method to find the global minimum of the loss function",
        "A feature selection technique",
        "A regularization method",
        "A way to handle outliers"
      ],
      correctAnswer: 0,
      explanation: "Gradient descent is an optimization algorithm that iteratively updates parameters to minimize the loss function.",
    },
    {
      id: 10,
      question: "What does the intercept represent in linear regression?",
      options: [
        "The slope of the regression line",
        "The predicted value when all independent variables are zero",
        "The correlation between variables",
        "The error term"
      ],
      correctAnswer: 1,
      explanation: "The intercept (bias) represents the predicted value of the target variable when all independent variables are zero.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
       

        {/* 1. What is Linear Regression? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <FunctionSquare className="h-5 w-5 text-primary" />
            1. What is Linear Regression?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Linear regression is a statistical method that models the relationship between a <span className="font-semibold text-foreground">dependent variable (y)</span> and one or more <span className="font-semibold text-foreground">independent variables (x)</span> by fitting a linear equation to observed data.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-lg">y = mx + b</p>
                <p className="text-xs text-muted-foreground mt-2">
                  y = target variable, x = feature, m = slope, b = intercept
                </p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">When to use Linear Regression?</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Relationship between variables is linear</li>
                    <li>Predicting continuous values</li>
                    <li>Need interpretable results</li>
                    <li>Small to medium datasets</li>
                    <li>Understanding feature importance</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Types of Linear Regression */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            2. Types of Linear Regression
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <X className="h-4 w-4 text-primary" />
                <h4 className="font-semibold text-foreground">Simple Linear Regression</h4>
              </div>
              <p className="text-xs text-muted-foreground">One independent variable</p>
              <div className="bg-muted p-2 rounded mt-2 text-center">
                <p className="font-mono text-sm">y = β₀ + β₁x</p>
              </div>
              <ul className="text-xs text-muted-foreground list-disc list-inside mt-2">
                <li>One feature</li>
                <li>Easy to interpret</li>
                <li>Good for understanding relationships</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Grid className="h-4 w-4 text-primary" />
                <h4 className="font-semibold text-foreground">Multiple Linear Regression</h4>
              </div>
              <p className="text-xs text-muted-foreground">Multiple independent variables</p>
              <div className="bg-muted p-2 rounded mt-2 text-center">
                <p className="font-mono text-sm">y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ</p>
              </div>
              <ul className="text-xs text-muted-foreground list-disc list-inside mt-2">
                <li>Multiple features</li>
                <li>Captures complex relationships</li>
                <li>More powerful but less interpretable</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3. The Math Behind Linear Regression */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Sigma className="h-5 w-5 text-primary" />
            3. The Math Behind Linear Regression
          </h2>

          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Ordinary Least Squares (OLS)</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">Minimize: Σ(yᵢ - ŷᵢ)²</p>
                <p className="text-xs text-muted-foreground mt-1">Sum of squared residuals</p>
              </div>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <h5 className="font-medium text-foreground text-sm">Formula for Slope (β₁)</h5>
                  <div className="bg-muted p-2 rounded mt-1">
                    <p className="font-mono text-xs">β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]</p>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Formula for Intercept (β₀)</h5>
                  <div className="bg-muted p-2 rounded mt-1">
                    <p className="font-mono text-xs">β₀ = ȳ - β₁x̄</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Matrix Formulation</h4>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-sm">β = (XᵀX)⁻¹Xᵀy</p>
                <p className="text-xs text-muted-foreground mt-1">Closed-form solution for multiple linear regression</p>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Key Assumptions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            4. Key Assumptions of Linear Regression
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">1. Linearity</h4>
              <p className="text-xs text-muted-foreground">The relationship between independent and dependent variables is linear</p>
              <div className="bg-muted p-2 rounded mt-2">
                <p className="text-xs">Check: Scatter plots, Residual plots</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">2. Independence of Errors</h4>
              <p className="text-xs text-muted-foreground">Residuals are independent of each other</p>
              <div className="bg-muted p-2 rounded mt-2">
                <p className="text-xs">Check: Durbin-Watson statistic</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">3. Homoscedasticity</h4>
              <p className="text-xs text-muted-foreground">Constant variance of residuals</p>
              <div className="bg-muted p-2 rounded mt-2">
                <p className="text-xs">Check: Scale-Location plot, Breusch-Pagan test</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">4. Normality of Residuals</h4>
              <p className="text-xs text-muted-foreground">Residuals are normally distributed</p>
              <div className="bg-muted p-2 rounded mt-2">
                <p className="text-xs">Check: Q-Q plot, Shapiro-Wilk test</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">5. No Multicollinearity</h4>
              <p className="text-xs text-muted-foreground">Independent variables are not highly correlated</p>
              <div className="bg-muted p-2 rounded mt-2">
                <p className="text-xs">Check: VIF (Variance Inflation Factor)</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">6. No Outliers</h4>
              <p className="text-xs text-muted-foreground">No extreme values affecting the model</p>
              <div className="bg-muted p-2 rounded mt-2">
                <p className="text-xs">Check: Box plots, Z-score, Cook's distance</p>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Training Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Cog className="h-5 w-5 text-primary" />
            5. Training Methods
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Normal Equation</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs">β = (XᵀX)⁻¹Xᵀy</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Exact solution</li>
                    <li>No iteration needed</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>O(n³) complexity</li>
                    <li>Not scalable</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Gradient Descent</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs">θⱼ = θⱼ - α∂J/∂θⱼ</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Works for large datasets</li>
                    <li>Flexible</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Requires tuning</li>
                    <li>May converge slowly</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">SVD</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs">X = UΣVᵀ</p>
                </div>
                <div>
                  <p className="font-medium text-foreground">Pros</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Numerically stable</li>
                    <li>Handles singular matrices</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-foreground">Cons</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Computationally expensive</li>
                    <li>Memory intensive</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Evaluation Metrics */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Gauge className="h-5 w-5 text-primary" />
            6. Evaluation Metrics
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
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">MAE (Mean Absolute Error)</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">MAE = (1/n)Σ|yᵢ - ŷᵢ|</p>
                  </div>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">RMSE (Root Mean Squared Error)</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">RMSE = √MSE</p>
                  </div>
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
                  <p className="text-xs text-muted-foreground mt-1">Proportion of variance explained</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Adjusted R²</p>
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">Adj R² = 1 - (1-R²)(n-1)/(n-k-1)</p>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Penalizes unnecessary features</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Pros and Cons */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            7. Pros and Cons
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Pros
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Simple and interpretable</span> - Easy to understand and explain</li>
                <li>• <span className="font-medium text-foreground">Fast to train</span> - Computationally efficient</li>
                <li>• <span className="font-medium text-foreground">No hyperparameters</span> - No tuning required</li>
                <li>• <span className="font-medium text-foreground">Works well with small datasets</span> - Doesn't need much data</li>
                <li>• <span className="font-medium text-foreground">Feature importance</span> - Coefficients show impact</li>
                <li>• <span className="font-medium text-foreground">Statistically sound</span> - Strong theoretical foundations</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                Cons
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Assumes linearity</span> - Can't capture non-linear patterns</li>
                <li>• <span className="font-medium text-foreground">Sensitive to outliers</span> - Outliers can skew results</li>
                <li>• <span className="font-medium text-foreground">Multicollinearity issues</span> - Correlated features cause problems</li>
                <li>• <span className="font-medium text-foreground">Can't handle categorical data</span> - Needs encoding</li>
                <li>• <span className="font-medium text-foreground">Overfitting with many features</span> - Needs regularization</li>
                <li>• <span className="font-medium text-foreground">Assumes normality</span> - Works best with normal data</li>
              </ul>
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
                  <h4 className="font-semibold text-foreground mb-1">Ignoring Assumptions</h4>
                  <p className="text-sm text-muted-foreground">
                    Not checking linearity, normality, or homoscedasticity leads to unreliable results.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Overfitting</h4>
                  <p className="text-sm text-muted-foreground">
                    Using too many features or polynomial terms without regularization.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Preprocessing the entire dataset before splitting into train/test.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Extrapolation</h4>
                  <p className="text-sm text-muted-foreground">
                    Predicting outside the range of training data is unreliable.
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
                <h4 className="font-semibold text-foreground mb-1">Always Check Assumptions</h4>
                <p className="text-sm text-muted-foreground">Validate linearity, normality, and homoscedasticity before interpreting results</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Scale Features</h4>
                <p className="text-sm text-muted-foreground">Standardize or normalize features for better interpretation and performance</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Handle Outliers</h4>
                <p className="text-sm text-muted-foreground">Detect and appropriately handle outliers before training</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Cross-Validation</h4>
                <p className="text-sm text-muted-foreground">Use cross-validation to evaluate model stability and performance</p>
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
            See linear regression in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Linear Regression Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}