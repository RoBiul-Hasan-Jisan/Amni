// app/learn/machine-learning/supervised-learning/page.tsx
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Target, Zap, Network, Layers, Hash, Flame, Brain, Cpu, BarChart, LineChart, PieChart, Database, GitBranch, Code, TestTube, Rocket, Shield, Cloud, TrendingUp, Bot, CpuIcon, Users, ChartBar, ChartLine, Filter } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export default function SupervisedLearningPage() {
  const supervisedCategories = [
    {
      title: "Regression",
      icon: ChartLine,
      description: "Predict continuous numerical values",
      algorithms: ["Linear Regression", "Polynomial Regression", "Ridge/Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"],
      examples: ["House price prediction", "Stock price forecasting", "Temperature prediction"]
    },
    {
      title: "Classification",
      icon: Target,
      description: "Predict discrete categorical labels",
      algorithms: ["Logistic Regression", "Decision Trees", "Random Forest", "SVM", "Naive Bayes", "K-NN"],
      examples: ["Spam detection", "Disease diagnosis", "Image classification"]
    },
    {
      title: "Ensemble Methods",
      icon: Users,
      description: "Combine multiple models for better performance",
      algorithms: ["Random Forest", "Gradient Boosting", "AdaBoost", "XGBoost", "LightGBM"],
      examples: ["Kaggle competitions", "Fraud detection", "Credit scoring"]
    },
    {
      title: "Neural Networks",
      icon: Brain,
      description: "Multi-layer networks for complex patterns",
      algorithms: ["MLP", "CNN", "RNN", "Transformer", "GAN"],
      examples: ["Computer vision", "Natural language processing", "Speech recognition"]
    }
  ];

  const regressionModels = [
    {
      name: "Linear Regression",
      complexity: "O(n²p)",
      assumptions: ["Linearity", "Independence", "Homoscedasticity", "Normality"],
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model coefficients
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient: {model.coef_[0][0]:.4f}")

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\\nMean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`
    },
    {
      name: "Ridge Regression (L2 Regularization)",
      complexity: "O(n²p)",
      assumptions: ["Same as Linear Regression", "Adds penalty for large coefficients"],
      code: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Generate sample data with multiple features
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
# True coefficients
true_coef = np.array([4, 2, 0, 0, -1, 0, 0, 0, 0, 0])
y = X.dot(true_coef) + np.random.randn(n_samples) * 0.5

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression with cross-validation
ridge = Ridge()
parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge_cv = GridSearchCV(ridge, parameters, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Best model
best_ridge = ridge_cv.best_estimator_
y_pred = best_ridge.predict(X_test_scaled)

print(f"Best alpha: {ridge_cv.best_params_['alpha']}")
print(f"Best CV score: {-ridge_cv.best_score_:.4f}")

# Compare coefficients
print("\\nFeature Coefficients:")
print("Feature | True Coef | Ridge Coef")
print("-" * 35)
for i, (true, ridge) in enumerate(zip(true_coef, best_ridge.coef_)):
    print(f"{i:7d} | {true:9.4f} | {ridge:10.4f}")

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"\\nTest MSE: {mse:.4f}")
print(f"Ridge Intercept: {best_ridge.intercept_:.4f}")`
    },
    {
      name: "Decision Tree Regressor",
      complexity: "O(n log n * p)",
      assumptions: ["None (non-parametric)"],
      code: `import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(200) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree with hyperparameter tuning
tree = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score (MSE): {-grid_search.best_score_:.4f}")

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(best_tree, filled=True, feature_names=['X'], rounded=True)
plt.title("Decision Tree Regressor")
plt.show()

# Visualize predictions
X_grid = np.arange(0, 5, 0.01)[:, np.newaxis]
y_grid = best_tree.predict(X_grid)

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(X_grid, y_grid, color='red', linewidth=2, label='Predicted')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.title('Decision Tree Regression: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`
    }
  ];

  const classificationModels = [
    {
      name: "Logistic Regression",
      complexity: "O(n*p)",
      assumptions: ["Binary outcome", "Linearity of log-odds", "No multicollinearity"],
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_clusters_per_class=2,
                          random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\\nConfusion Matrix:")
print(conf_matrix)
print("\\nClassification Report:")
print(class_report)

# Coefficients
print("\\nTop 5 Feature Coefficients:")
coef_df = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'Coefficient': log_reg.coef_[0]
})
coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
print(coef_df.nlargest(5, 'Abs_Coefficient')[['Feature', 'Coefficient']])

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()`
    },
    {
      name: "Random Forest Classifier",
      complexity: "O(n log n * p * trees)",
      assumptions: ["None (non-parametric)"],
      code: `import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with hyperparameter tuning
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\\nTest Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nFeature Importance:")
print(feature_importance)

# Confusion matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=class_names)
disp.plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title("Confusion Matrix")

# Feature Importance Plot
axes[1].barh(range(len(feature_importance)), feature_importance['importance'])
axes[1].set_yticks(range(len(feature_importance)))
axes[1].set_yticklabels(feature_importance['feature'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(best_rf, X, y, cv=10, scoring='accuracy')
print(f"\\n10-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")`
    },
    {
      name: "Support Vector Machine (SVM)",
      complexity: "O(n² to n³)",
      assumptions: ["Classes are separable"],
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Generate non-linear data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    if kernel == 'poly':
        svc = svm.SVC(kernel=kernel, degree=3, random_state=42)
    else:
        svc = svm.SVC(kernel=kernel, random_state=42)
    
    svc.fit(X_train_scaled, y_train)
    y_pred = svc.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[kernel] = {'model': svc, 'accuracy': accuracy}
    
    print(f"{kernel.upper()} Kernel Accuracy: {accuracy:.4f}")

# Hyperparameter tuning for RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

svc = svm.SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_svc = grid_search.best_estimator_
y_pred_best = best_svc.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f"\\nBest RBF Parameters: {grid_search.best_params_}")
print(f"Best RBF Accuracy: {best_accuracy:.4f}")

# Decision boundary visualization
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary ({title})')
    plt.colorbar()
    plt.show()

# Plot decision boundary for best model
plot_decision_boundary(best_svc, X_test_scaled, y_test, 'RBF Kernel')`
    }
  ];

  const keyConcepts = [
    {
      concept: "Bias-Variance Tradeoff",
      description: "Balancing underfitting (high bias) and overfitting (high variance)",
      formula: "Total Error = Bias² + Variance + Irreducible Error",
      visualization: "U-shaped curve showing optimal model complexity"
    },
    {
      concept: "Regularization",
      description: "Techniques to prevent overfitting by adding penalty terms",
      types: ["L1 (Lasso)", "L2 (Ridge)", "Elastic Net (L1 + L2)"],
      effect: "Shrinks coefficients, reduces model complexity"
    },
    {
      concept: "Gradient Descent",
      description: "Optimization algorithm to minimize loss function",
      variants: ["Batch GD", "Stochastic GD", "Mini-batch GD"],
      learning_rate: "Step size for parameter updates"
    },
    {
      concept: "Cross-Validation",
      description: "Resampling technique for robust performance estimation",
      methods: ["k-Fold", "Stratified k-Fold", "Leave-One-Out", "Time Series CV"],
      purpose: "Better utilization of data, reliable performance estimates"
    }
  ];

  const evaluationMetrics = {
    regression: [
      { metric: "Mean Absolute Error (MAE)", formula: "Σ|yᵢ - ŷᵢ|/n", interpretation: "Average absolute error" },
      { metric: "Mean Squared Error (MSE)", formula: "Σ(yᵢ - ŷᵢ)²/n", interpretation: "Penalizes large errors" },
      { metric: "Root Mean Squared Error (RMSE)", formula: "√MSE", interpretation: "In original units" },
      { metric: "R² Score", formula: "1 - SS_res/SS_tot", interpretation: "Variance explained" },
      { metric: "Adjusted R²", formula: "1 - [(1-R²)(n-1)/(n-p-1)]", interpretation: "Accounts for predictors" }
    ],
    classification: [
      { metric: "Accuracy", formula: "(TP+TN)/Total", interpretation: "Overall correctness" },
      { metric: "Precision", formula: "TP/(TP+FP)", interpretation: "Positive predictive value" },
      { metric: "Recall", formula: "TP/(TP+FN)", interpretation: "Sensitivity, true positive rate" },
      { metric: "F1-Score", formula: "2*(Precision*Recall)/(Precision+Recall)", interpretation: "Harmonic mean" },
      { metric: "ROC-AUC", formula: "Area under ROC curve", interpretation: "Overall performance" },
      { metric: "Log Loss", formula: "-Σ[yᵢlog(ŷᵢ)+(1-yᵢ)log(1-ŷᵢ)]", interpretation: "Probability-based error" }
    ]
  };

  const modelSelectionGuide = [
    {
      scenario: "Small dataset, interpretability important",
      recommended: ["Linear/Logistic Regression", "Decision Trees"],
      reason: "Simple models prevent overfitting, easy to explain"
    },
    {
      scenario: "Large dataset, high accuracy needed",
      recommended: ["Random Forest", "Gradient Boosting", "XGBoost"],
      reason: "Ensemble methods handle complex patterns well"
    },
    {
      scenario: "Text/Image data",
      recommended: ["Neural Networks", "CNNs", "Transformers"],
      reason: "Deep learning excels at unstructured data"
    },
    {
      scenario: "Real-time predictions",
      recommended: ["Logistic Regression", "Naive Bayes", "Small Decision Trees"],
      reason: "Fast inference time, low computational cost"
    },
    {
      scenario: "Highly imbalanced classes",
      recommended: ["XGBoost with scale_pos_weight", "SMOTE + Random Forest"],
      reason: "Special handling needed for minority classes"
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the main characteristic of supervised learning?",
      options: [
        "It uses unlabeled data",
        "It learns from labeled input-output pairs",
        "It learns through trial and error",
        "It doesn't require training data"
      ],
      correctAnswer: 1,
      explanation: "Supervised learning uses labeled datasets to train algorithms that predict outcomes accurately.",
    },
    {
      id: 2,
      question: "Which algorithm would you use for predicting house prices?",
      options: ["Logistic Regression", "K-Means", "Linear Regression", "SVM"],
      correctAnswer: 2,
      explanation: "Linear regression is used for predicting continuous numerical values like house prices.",
    },
    {
      id: 3,
      question: "What does regularization prevent in machine learning models?",
      options: ["Underfitting", "Overfitting", "Data leakage", "Feature scaling"],
      correctAnswer: 1,
      explanation: "Regularization techniques like L1/L2 prevent overfitting by adding penalty terms to the loss function.",
    },
    {
      id: 4,
      question: "Which metric is most appropriate for imbalanced classification problems?",
      options: ["Accuracy", "Precision-Recall Curve", "R² Score", "MSE"],
      correctAnswer: 1,
      explanation: "For imbalanced datasets, precision-recall curve is more informative than accuracy.",
    },
    {
      id: 5,
      question: "What is the purpose of cross-validation?",
      options: [
        "To increase training speed",
        "To get more reliable performance estimates",
        "To reduce model complexity",
        "To visualize decision boundaries"
      ],
      correctAnswer: 1,
      explanation: "Cross-validation provides more robust performance estimates by using multiple train-test splits.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Supervised Machine Learning
          </h1>
          <p className="text-lg text-muted-foreground">
            Complete guide to supervised learning: Learn how algorithms predict outcomes from labeled training data through regression and classification.
          </p>
        </header>

        {/* Hero Section */}
        <section className="mb-12">
          <Card className="bg-gradient-to-r from-primary/10 to-blue-500/10 border-primary/20">
            <CardContent className="p-8">
              <div className="flex flex-col md:flex-row items-center gap-8">
                <div className="md:w-2/3">
                  <h2 className="text-2xl font-bold mb-4 text-foreground">
                    What is Supervised Learning?
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Supervised learning is a type of machine learning where models are trained on labeled datasets. 
                    The algorithm learns the mapping function from input variables (features) to output variables (labels). 
                    Once trained, it can predict outputs for new, unseen data.
                  </p>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-background/50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2 text-foreground">Training Phase</h4>
                      <p className="text-sm text-muted-foreground">
                        Model learns patterns from labeled training data (input → output pairs)
                      </p>
                    </div>
                    <div className="bg-background/50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2 text-foreground">Prediction Phase</h4>
                      <p className="text-sm text-muted-foreground">
                        Model applies learned patterns to make predictions on new, unseen data
                      </p>
                    </div>
                  </div>
                </div>
                <div className="md:w-1/3">
                  <div className="bg-primary/5 p-6 rounded-lg">
                    <h3 className="font-semibold mb-3 text-foreground">Key Components</h3>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-center gap-2">
                        <Database className="h-4 w-4 text-primary" />
                        <span>Labeled training data</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Target className="h-4 w-4 text-primary" />
                        <span>Input features (X)</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <ChartLine className="h-4 w-4 text-primary" />
                        <span>Output labels (y)</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-primary" />
                        <span>Learning algorithm</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <TestTube className="h-4 w-4 text-primary" />
                        <span>Evaluation metrics</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Categories */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Supervised Learning Categories</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {supervisedCategories.map((category, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <category.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{category.title}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    {category.description}
                  </p>
                  
                  <div className="space-y-3">
                    <div>
                      <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Common Algorithms:
                      </h4>
                      <ul className="space-y-1">
                        {category.algorithms.map((algo, aIdx) => (
                          <li key={aIdx} className="flex items-center gap-2 text-sm">
                            <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                            <span className="text-muted-foreground">{algo}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Examples:
                      </h4>
                      <ul className="space-y-1">
                        {category.examples.map((example, eIdx) => (
                          <li key={eIdx} className="text-xs text-muted-foreground">
                            • {example}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Regression Models */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Regression Algorithms</h2>
          <Tabs defaultValue="linear" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="linear">Linear Regression</TabsTrigger>
              <TabsTrigger value="ridge">Ridge Regression</TabsTrigger>
              <TabsTrigger value="tree">Decision Tree</TabsTrigger>
            </TabsList>
            
            {regressionModels.map((model, idx) => (
              <TabsContent key={idx} value={model.name.toLowerCase().split(' ')[0]}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                      <div className="lg:w-1/3">
                        <div className="flex items-center gap-3 mb-4">
                          <Badge variant="outline" className="font-mono">
                            {model.complexity}
                          </Badge>
                          <h3 className="font-semibold text-lg text-foreground">{model.name}</h3>
                        </div>
                        
                        <div className="space-y-4">
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-2">Key Assumptions:</h4>
                            <ul className="space-y-1">
                              {model.assumptions.map((assumption, aIdx) => (
                                <li key={aIdx} className="flex items-start gap-2 text-sm">
                                  <AlertCircle className="h-3 w-3 text-warning mt-0.5 shrink-0" />
                                  <span className="text-muted-foreground">{assumption}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-2">When to Use:</h4>
                            <p className="text-sm text-muted-foreground">
                              {model.name === "Linear Regression" && 
                                "When relationship between features and target is linear, for baseline models"}
                              {model.name === "Ridge Regression (L2 Regularization)" && 
                                "When features are correlated, to prevent overfitting"}
                              {model.name === "Decision Tree Regressor" && 
                                "For non-linear relationships, when interpretability is important"}
                            </p>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-2">Pros & Cons:</h4>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="bg-green-500/10 p-2 rounded">
                                <p className="text-xs font-semibold text-green-600">Pros</p>
                                <ul className="text-xs text-muted-foreground mt-1">
                                  {model.name === "Linear Regression" && (
                                    <>
                                      <li>• Simple & interpretable</li>
                                      <li>• Fast training & prediction</li>
                                      <li>• Works well with linear relationships</li>
                                    </>
                                  )}
                                  {model.name === "Ridge Regression (L2 Regularization)" && (
                                    <>
                                      <li>• Prevents overfitting</li>
                                      <li>• Handles multicollinearity</li>
                                      <li>• More stable than ordinary least squares</li>
                                    </>
                                  )}
                                  {model.name === "Decision Tree Regressor" && (
                                    <>
                                      <li>• No assumptions about data</li>
                                      <li>• Handles non-linear relationships</li>
                                      <li>• Easy to interpret</li>
                                    </>
                                  )}
                                </ul>
                              </div>
                              <div className="bg-red-500/10 p-2 rounded">
                                <p className="text-xs font-semibold text-red-600">Cons</p>
                                <ul className="text-xs text-muted-foreground mt-1">
                                  {model.name === "Linear Regression" && (
                                    <>
                                      <li>• Assumes linearity</li>
                                      <li>• Sensitive to outliers</li>
                                      <li>• Can't capture complex patterns</li>
                                    </>
                                  )}
                                  {model.name === "Ridge Regression (L2 Regularization)" && (
                                    <>
                                      <li>• Doesn't perform feature selection</li>
                                      <li>• Requires hyperparameter tuning</li>
                                      <li>• All coefficients remain non-zero</li>
                                    </>
                                  )}
                                  {model.name === "Decision Tree Regressor" && (
                                    <>
                                      <li>• Prone to overfitting</li>
                                      <li>• High variance</li>
                                      <li>• Poor extrapolation</li>
                                    </>
                                  )}
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="lg:w-2/3">
                        <CodeBlock
                          language="python"
                          code={model.code}
                          filename={`${model.name.toLowerCase().replace(' ', '_')}.py`}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Classification Models */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Classification Algorithms</h2>
          <Tabs defaultValue="logistic" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="logistic">Logistic Regression</TabsTrigger>
              <TabsTrigger value="random">Random Forest</TabsTrigger>
              <TabsTrigger value="svm">Support Vector Machine</TabsTrigger>
            </TabsList>
            
            {classificationModels.map((model, idx) => (
              <TabsContent key={idx} value={model.name.toLowerCase().split(' ')[0]}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                      <div className="lg:w-1/3">
                        <div className="flex items-center gap-3 mb-4">
                          <Badge variant="outline" className="font-mono">
                            {model.complexity}
                          </Badge>
                          <h3 className="font-semibold text-lg text-foreground">{model.name}</h3>
                        </div>
                        
                        <div className="space-y-4">
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-2">Key Assumptions:</h4>
                            <ul className="space-y-1">
                              {model.assumptions.map((assumption, aIdx) => (
                                <li key={aIdx} className="flex items-start gap-2 text-sm">
                                  <AlertCircle className="h-3 w-3 text-warning mt-0.5 shrink-0" />
                                  <span className="text-muted-foreground">{assumption}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-2">When to Use:</h4>
                            <p className="text-sm text-muted-foreground">
                              {model.name === "Logistic Regression" && 
                                "For binary classification problems, when you need probability outputs and interpretable coefficients"}
                              {model.name === "Random Forest Classifier" && 
                                "For complex classification tasks with non-linear relationships, robust to outliers"}
                              {model.name === "Support Vector Machine (SVM)" && 
                                "When classes are separable, especially in high-dimensional spaces"}
                            </p>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-2">Pros & Cons:</h4>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="bg-green-500/10 p-2 rounded">
                                <p className="text-xs font-semibold text-green-600">Pros</p>
                                <ul className="text-xs text-muted-foreground mt-1">
                                  {model.name === "Logistic Regression" && (
                                    <>
                                      <li>• Provides probabilities</li>
                                      <li>• Interpretable coefficients</li>
                                      <li>• Fast and efficient</li>
                                    </>
                                  )}
                                  {model.name === "Random Forest Classifier" && (
                                    <>
                                      <li>• High accuracy</li>
                                      <li>• Handles non-linearity well</li>
                                      <li>• Robust to outliers</li>
                                    </>
                                  )}
                                  {model.name === "Support Vector Machine (SVM)" && (
                                    <>
                                      <li>• Effective in high dimensions</li>
                                      <li>• Memory efficient</li>
                                      <li>• Versatile with different kernels</li>
                                    </>
                                  )}
                                </ul>
                              </div>
                              <div className="bg-red-500/10 p-2 rounded">
                                <p className="text-xs font-semibold text-red-600">Cons</p>
                                <ul className="text-xs text-muted-foreground mt-1">
                                  {model.name === "Logistic Regression" && (
                                    <>
                                      <li>• Assumes linear decision boundary</li>
                                      <li>• Sensitive to correlated features</li>
                                      <li>• Can't capture complex patterns</li>
                                    </>
                                  )}
                                  {model.name === "Random Forest Classifier" && (
                                    <>
                                      <li>• Black box model</li>
                                      <li>• Computationally expensive</li>
                                      <li>• Can overfit with noisy data</li>
                                    </>
                                  )}
                                  {model.name === "Support Vector Machine (SVM)" && (
                                    <>
                                      <li>• Doesn't scale well to large datasets</li>
                                      <li>• Requires careful tuning</li>
                                      <li>• Difficult to interpret</li>
                                    </>
                                  )}
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="lg:w-2/3">
                        <CodeBlock
                          language="python"
                          code={model.code}
                          filename={`${model.name.toLowerCase().replace(' ', '_').replace(/[()]/g, '')}.py`}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Key Concepts */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Key Theoretical Concepts</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {keyConcepts.map((concept, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <Brain className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-lg text-foreground">{concept.concept}</h3>
                  </div>
                  
                  <p className="text-sm text-muted-foreground mb-4">
                    {concept.description}
                  </p>
                  
                  {concept.formula && (
                    <div className="mb-4">
                      <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Formula:
                      </h4>
                      <div className="bg-muted/50 p-3 rounded font-mono text-sm">
                        {concept.formula}
                      </div>
                    </div>
                  )}
                  
                  {concept.types && (
                    <div className="mb-4">
                      <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Types:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {concept.types.map((type, tIdx) => (
                          <Badge key={tIdx} variant="secondary" className="text-xs">
                            {type}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {concept.variants && (
                    <div className="mb-4">
                      <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Variants:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {concept.variants.map((variant, vIdx) => (
                          <Badge key={vIdx} variant="outline" className="text-xs">
                            {variant}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {concept.effect && (
                    <div className="mt-4 p-3 bg-blue-500/5 rounded border border-blue-500/10">
                      <h4 className="text-xs font-semibold text-blue-600 mb-1">Effect:</h4>
                      <p className="text-xs text-muted-foreground">{concept.effect}</p>
                    </div>
                  )}
                  
                  {concept.learning_rate && (
                    <div className="mt-4 p-3 bg-green-500/5 rounded border border-green-500/10">
                      <h4 className="text-xs font-semibold text-green-600 mb-1">Learning Rate:</h4>
                      <p className="text-xs text-muted-foreground">{concept.learning_rate}</p>
                    </div>
                  )}
                  
                  {concept.methods && (
                    <div className="mt-4">
                      <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Methods:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {concept.methods.map((method, mIdx) => (
                          <Badge key={mIdx} variant="secondary" className="text-xs">
                            {method}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {concept.purpose && (
                    <div className="mt-4 p-3 bg-purple-500/5 rounded border border-purple-500/10">
                      <h4 className="text-xs font-semibold text-purple-600 mb-1">Purpose:</h4>
                      <p className="text-xs text-muted-foreground">{concept.purpose}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Evaluation Metrics */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Evaluation Metrics</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            {/* Regression Metrics */}
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-6">
                  <TrendingUp className="h-5 w-5 text-blue-500" />
                  <h3 className="font-semibold text-lg text-foreground">Regression Metrics</h3>
                </div>
                
                <div className="space-y-4">
                  {evaluationMetrics.regression.map((metric, idx) => (
                    <div key={idx} className="border-b border-border/50 pb-4 last:border-0 last:pb-0">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-medium text-sm text-foreground">{metric.metric}</h4>
                        <Badge variant="outline" className="text-xs">
                          {metric.formula}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{metric.interpretation}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Classification Metrics */}
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-6">
                  <Target className="h-5 w-5 text-green-500" />
                  <h3 className="font-semibold text-lg text-foreground">Classification Metrics</h3>
                </div>
                
                <div className="space-y-4">
                  {evaluationMetrics.classification.map((metric, idx) => (
                    <div key={idx} className="border-b border-border/50 pb-4 last:border-0 last:pb-0">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-medium text-sm text-foreground">{metric.metric}</h4>
                        <Badge variant="outline" className="text-xs">
                          {metric.formula}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{metric.interpretation}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Model Selection Guide */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Model Selection Guide</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {modelSelectionGuide.map((scenario, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-primary/10 rounded">
                      {idx === 0 && <Code className="h-4 w-4 text-primary" />}
                      {idx === 1 && <CpuIcon className="h-4 w-4 text-primary" />}
                      {idx === 2 && <Layers className="h-4 w-4 text-primary" />}
                      {idx === 3 && <Zap className="h-4 w-4 text-primary" />}
                      {idx === 4 && <Filter className="h-4 w-4 text-primary" />}
                    </div>
                    <h3 className="font-semibold text-sm text-foreground">Scenario {idx + 1}</h3>
                  </div>
                  
                  <h4 className="font-medium text-foreground mb-3">{scenario.scenario}</h4>
                  
                  <div className="space-y-4">
                    <div>
                      <h5 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Recommended Models:
                      </h5>
                      <div className="flex flex-wrap gap-2">
                        {scenario.recommended.map((model, mIdx) => (
                          <Badge key={mIdx} variant="secondary" className="text-xs">
                            {model}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="text-xs font-semibold text-foreground uppercase tracking-wide mb-2">
                        Why:
                      </h5>
                      <p className="text-sm text-muted-foreground">{scenario.reason}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Workflow Steps */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Supervised Learning Workflow</h2>
          
          <Card>
            <CardContent className="p-6">
              <div className="grid md:grid-cols-5 gap-4">
                {[
                  { step: 1, title: "Data Collection", icon: Database, description: "Gather labeled dataset with features and target variable" },
                  { step: 2, title: "Preprocessing", icon: Filter, description: "Handle missing values, outliers, and feature scaling" },
                  { step: 3, title: "Model Selection", icon: CpuIcon, description: "Choose appropriate algorithm based on problem type" },
                  { step: 4, title: "Training", icon: Brain, description: "Fit model to training data and learn patterns" },
                  { step: 5, title: "Evaluation", icon: ChartBar, description: "Test model on unseen data and measure performance" },
                ].map((item) => (
                  <div key={item.step} className="text-center">
                    <div className="relative mb-4">
                      <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                        <item.icon className="h-6 w-6 text-primary" />
                      </div>
                      <div className="absolute -top-2 -right-2 w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {item.step}
                      </div>
                    </div>
                    <h3 className="font-semibold text-foreground mb-2">{item.title}</h3>
                    <p className="text-xs text-muted-foreground">{item.description}</p>
                  </div>
                ))}
              </div>
              
              <Separator className="my-8" />
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-lg text-foreground mb-4">Best Practices</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <h4 className="font-medium text-sm text-foreground">Always split data</h4>
                        <p className="text-sm text-muted-foreground">Use train/test/validation splits (70/15/15 or 80/20)</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <h4 className="font-medium text-sm text-foreground">Feature scaling</h4>
                        <p className="text-sm text-muted-foreground">Normalize or standardize features for gradient-based algorithms</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                      <div>
                        <h4 className="font-medium text-sm text-foreground">Cross-validation</h4>
                        <p className="text-sm text-muted-foreground">Use k-fold CV for reliable performance estimates</p>
                      </div>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold text-lg text-foreground mb-4">Common Pitfalls</h3>
                  <ul className="space-y-3">
                    <li className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                      <div>
                        <h4 className="font-medium text-sm text-foreground">Data leakage</h4>
                        <p className="text-sm text-muted-foreground">Don't use test data during training or preprocessing</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                      <div>
                        <h4 className="font-medium text-sm text-foreground">Overfitting</h4>
                        <p className="text-sm text-muted-foreground">Monitor validation performance, use regularization</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                      <div>
                        <h4 className="font-medium text-sm text-foreground">Imbalanced data</h4>
                        <p className="text-sm text-muted-foreground">Use appropriate metrics and sampling techniques</p>
                      </div>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* FAQ Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Frequently Asked Questions</h2>
          
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
              <AccordionTrigger>What's the difference between regression and classification?</AccordionTrigger>
              <AccordionContent>
                Regression predicts continuous numerical values (e.g., house prices, temperature), while classification predicts discrete categorical labels (e.g., spam/not spam, disease/no disease). Regression uses algorithms like Linear Regression, while classification uses Logistic Regression, SVM, etc.
              </AccordionContent>
            </AccordionItem>
            
            <AccordionItem value="item-2">
              <AccordionTrigger>How do I choose between linear and non-linear models?</AccordionTrigger>
              <AccordionContent>
                Start with linear models for interpretability and baselines. If performance is poor, try non-linear models. Use feature transformations or polynomial features with linear models for simple non-linearity. Use tree-based models or neural networks for complex non-linear patterns.
              </AccordionContent>
            </AccordionItem>
            
            <AccordionItem value="item-3">
              <AccordionTrigger>What's the bias-variance tradeoff?</AccordionTrigger>
              <AccordionContent>
                Bias is error from overly simplistic assumptions; variance is error from sensitivity to fluctuations in training data. Simple models have high bias (underfit), complex models have high variance (overfit). The goal is to find the sweet spot where total error (bias² + variance) is minimized.
              </AccordionContent>
            </AccordionItem>
            
            <AccordionItem value="item-4">
              <AccordionTrigger>When should I use regularization?</AccordionTrigger>
              <AccordionContent>
                Use regularization when: 1) You have many features relative to samples, 2) Features are correlated, 3) Model shows signs of overfitting (good training performance, poor test performance). L1 (Lasso) for feature selection, L2 (Ridge) for general overfitting prevention, Elastic Net for both.
              </AccordionContent>
            </AccordionItem>
            
            <AccordionItem value="item-5">
              <AccordionTrigger>How much data do I need for supervised learning?</AccordionTrigger>
              <AccordionContent>
                It depends on problem complexity. Simple linear models: 10-100 samples per feature. Complex models (deep learning): thousands to millions of samples. More data generally improves performance, but quality matters more than quantity. Use techniques like data augmentation for small datasets.
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} />
        </section>

        {/* Next Steps */}
        <section className="mb-12">
          <Card className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-green-500/20">
            <CardContent className="p-8">
              <div className="flex items-center gap-4 mb-6">
                <Rocket className="h-8 w-8 text-green-500" />
                <h2 className="text-2xl font-bold text-foreground">Next Steps in Your Journey</h2>
              </div>
              
              <div className="grid md:grid-cols-3 gap-6">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <Shield className="h-5 w-5 text-blue-500" />
                      <h3 className="font-semibold text-foreground">Unsupervised Learning</h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      Learn clustering, dimensionality reduction, and association rules for unlabeled data
                    </p>
                    <Badge variant="outline" className="text-xs">Clustering • PCA • Anomaly Detection</Badge>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <Bot className="h-5 w-5 text-purple-500" />
                      <h3 className="font-semibold text-foreground">Deep Learning</h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      Master neural networks, CNNs for images, RNNs for sequences, and transformers for NLP
                    </p>
                    <Badge variant="outline" className="text-xs">Neural Networks • TensorFlow • PyTorch</Badge>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <Cloud className="h-5 w-5 text-orange-500" />
                      <h3 className="font-semibold text-foreground">MLOps & Deployment</h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      Learn to deploy models, create ML pipelines, monitor performance, and scale systems
                    </p>
                    <Badge variant="outline" className="text-xs">Docker • Kubernetes • CI/CD</Badge>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}