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
  GraduationCap
} from "lucide-react";

export default function SupervisedLearningPage() {
  const result = getSubtopicBySlug("machine-learning", "supervised");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-linear",
      label: "Linear Regression",
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")`,
    },
    {
      language: "python-logistic",
      label: "Logistic Regression",
      code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))`,
    },
    {
      language: "python-decision",
      label: "Decision Tree",
      code: `from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\\nFeature Importance:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")`,
    },
    {
      language: "python-random",
      label: "Random Forest",
      code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Mean Accuracy: {cv_scores.mean():.3f}")`,
    },
    {
      language: "python-svm",
      label: "SVM",
      code: `from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))`,
    },
    {
      language: "python-xgboost",
      label: "XGBoost",
      code: `import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Generate data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=20
)

# Predict and evaluate
y_pred = model.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Supervised Learning?",
      options: [
        "Learning without labeled data",
        "Learning with labeled data where inputs have known outputs",
        "Learning through rewards and punishments",
        "Learning from unlabeled data"
      ],
      correctAnswer: 1,
      explanation: "Supervised learning uses labeled data where each input (X) has a known output (y) to train the model.",
    },
    {
      id: 2,
      question: "What is the difference between Regression and Classification?",
      options: [
        "Regression predicts continuous values, Classification predicts categorical labels",
        "Regression predicts categorical labels, Classification predicts continuous values",
        "They are the same thing",
        "Regression uses deep learning, Classification uses traditional ML"
      ],
      correctAnswer: 0,
      explanation: "Regression predicts continuous values (e.g., house prices), while Classification predicts categorical labels (e.g., spam/not spam).",
    },
    {
      id: 3,
      question: "What is overfitting in Supervised Learning?",
      options: [
        "Model performs well on all data",
        "Model performs well on training data but poorly on new data",
        "Model is too simple to capture patterns",
        "Model has high bias"
      ],
      correctAnswer: 1,
      explanation: "Overfitting occurs when a model learns training data too well, capturing noise instead of general patterns.",
    },
    {
      id: 4,
      question: "Which algorithm is best for binary classification with linear boundaries?",
      options: [
        "Random Forest",
        "Logistic Regression",
        "KNN",
        "Decision Tree"
      ],
      correctAnswer: 1,
      explanation: "Logistic Regression is ideal for binary classification with linear decision boundaries.",
    },
    {
      id: 5,
      question: "What is the Bias-Variance Tradeoff?",
      options: [
        "Choosing between two algorithms",
        "Balancing model simplicity and complexity",
        "Choosing between training and test data",
        "A technique for feature selection"
      ],
      correctAnswer: 1,
      explanation: "The Bias-Variance tradeoff balances underfitting (high bias) against overfitting (high variance).",
    },
    {
      id: 6,
      question: "What does L1 regularization (Lasso) do?",
      options: [
        "Adds squared penalty to weights",
        "Adds absolute value penalty to weights",
        "Adds no penalty",
        "Removes all features"
      ],
      correctAnswer: 1,
      explanation: "L1 regularization adds an absolute value penalty to weights, which can drive some weights to zero for feature selection.",
    },
    {
      id: 7,
      question: "Which metric is NOT used for classification evaluation?",
      options: [
        "Accuracy",
        "F1 Score",
        "Mean Squared Error (MSE)",
        "Precision"
      ],
      correctAnswer: 2,
      explanation: "MSE is a regression metric, not a classification metric.",
    },
    {
      id: 8,
      question: "What is the purpose of Cross-Validation?",
      options: [
        "To increase training data",
        "To better evaluate model performance and prevent overfitting",
        "To make training faster",
        "To reduce the number of features"
      ],
      correctAnswer: 1,
      explanation: "Cross-validation provides a more robust evaluation by testing on multiple validation sets.",
    },
    {
      id: 9,
      question: "Which algorithm is an ensemble method?",
      options: [
        "Linear Regression",
        "Random Forest",
        "Logistic Regression",
        "Naive Bayes"
      ],
      correctAnswer: 1,
      explanation: "Random Forest is an ensemble method that combines multiple decision trees.",
    },
    {
      id: 10,
      question: "What is the difference between parameters and hyperparameters?",
      options: [
        "They are the same thing",
        "Parameters are learned during training, hyperparameters are set before training",
        "Hyperparameters are learned during training, parameters are set before training",
        "Parameters are for classification, hyperparameters are for regression"
      ],
      correctAnswer: 1,
      explanation: "Parameters are learned from training data (e.g., weights), while hyperparameters are set before training (e.g., learning rate, max depth).",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Supervised Machine Learning — Complete Guide
          </h1>
          <p className="text-muted-foreground text-lg">
            A comprehensive guide to understanding supervised learning, algorithms, evaluation metrics, and best practices
          </p>
        </div>

        {/* 1. What is Supervised Learning? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            1. What is Supervised Learning?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Supervised learning is a type of machine learning where a model is trained using <span className="font-semibold text-foreground">labeled data</span>, meaning each input (X) has a known output (y).
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-lg text-primary">X → y</p>
                <p className="text-xs text-muted-foreground">Mapping function learned from data</p>
              </div>
              <p className="text-muted-foreground mt-3">
                The goal is to learn a mapping function from inputs to outputs that can predict for unseen data.
              </p>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Analogy</h4>
                  <p className="text-sm text-muted-foreground">
                    Think of supervised learning like a student learning with a teacher:
                  </p>
                  <ul className="text-sm text-muted-foreground list-disc list-inside mt-2">
                    <li><span className="font-medium text-foreground">Teacher</span> provides examples (labeled data)</li>
                    <li><span className="font-medium text-foreground">Student</span> learns patterns (model training)</li>
                    <li><span className="font-medium text-foreground">Exam</span> tests on new questions (prediction)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Training vs Prediction */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            2. Training vs Prediction
          </h2>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <GraduationCap className="h-4 w-4 text-primary" />
                <h4 className="font-semibold text-foreground">Training Phase</h4>
              </div>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• Model learns patterns from labeled data</li>
                <li>• Input: Features (X)</li>
                <li>• Output: Labels (y)</li>
                <li>• Goal: Minimize loss function</li>
              </ul>
              <div className="bg-muted p-2 rounded mt-2 text-center">
                <p className="font-mono text-xs">(X_train, y_train) → Model</p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <h4 className="font-semibold text-foreground">Prediction Phase</h4>
              </div>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• Model predicts output for new unseen inputs</li>
                <li>• Input: New features (X_test)</li>
                <li>• Output: Predictions (y_pred)</li>
                <li>• Goal: Generalize to new data</li>
              </ul>
              <div className="bg-muted p-2 rounded mt-2 text-center">
                <p className="font-mono text-xs">X_test → Model → y_pred</p>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Key Components */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            3. Key Components
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• <span className="font-medium text-foreground">Features (X)</span>: Input variables</li>
                <li>• <span className="font-medium text-foreground">Labels (y)</span>: Target variables</li>
                <li>• <span className="font-medium text-foreground">Dataset</span>: Training/Test/Validation</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Model</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• <span className="font-medium text-foreground">Algorithm</span>: Learning method</li>
                <li>• <span className="font-medium text-foreground">Parameters</span>: Learned from data</li>
                <li>• <span className="font-medium text-foreground">Loss Function</span>: Measures error</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Evaluation</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• <span className="font-medium text-foreground">Metrics</span>: Performance measures</li>
                <li>• <span className="font-medium text-foreground">Cross-Validation</span>: Robust evaluation</li>
                <li>• <span className="font-medium text-foreground">Testing</span>: Unseen data evaluation</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 4. Types of Supervised Learning */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            4. Types of Supervised Learning
          </h2>

          <div className="space-y-6">
            {/* Regression */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3 flex items-center gap-2">
                <LineChart className="h-5 w-5 text-primary" />
                4.1 Regression
              </h3>
              <p className="text-muted-foreground mb-2">
                Predicts <span className="font-medium text-foreground">continuous values</span>
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Examples</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>House price prediction</li>
                    <li>Temperature forecasting</li>
                    <li>Stock price prediction</li>
                  </ul>
                </div>
                <div className="md:col-span-2">
                  <h4 className="font-semibold text-foreground text-sm">Algorithms</h4>
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="bg-muted px-2 py-1 rounded text-xs">Linear Regression</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Polynomial Regression</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Ridge / Lasso</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Decision Tree Regressor</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Random Forest Regressor</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Classification */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3 flex items-center gap-2">
                <PieChart className="h-5 w-5 text-primary" />
                4.2 Classification
              </h3>
              <p className="text-muted-foreground mb-2">
                Predicts <span className="font-medium text-foreground">categorical labels</span>
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Examples</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Spam detection</li>
                    <li>Disease diagnosis</li>
                    <li>Image classification</li>
                  </ul>
                </div>
                <div className="md:col-span-2">
                  <h4 className="font-semibold text-foreground text-sm">Algorithms</h4>
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="bg-muted px-2 py-1 rounded text-xs">Logistic Regression</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Decision Trees</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Random Forest</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">SVM</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">KNN</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Naive Bayes</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Ensemble Methods */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3 flex items-center gap-2">
                <Layers className="h-5 w-5 text-primary" />
                4.3 Ensemble Methods
              </h3>
              <p className="text-muted-foreground mb-2">
                Combine multiple models for <span className="font-medium text-foreground">better accuracy</span>
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Algorithms</h4>
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="bg-muted px-2 py-1 rounded text-xs">Random Forest</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">Gradient Boosting</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">XGBoost</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">AdaBoost</span>
                    <span className="bg-muted px-2 py-1 rounded text-xs">LightGBM</span>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Use Cases</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Fraud detection</li>
                    <li>Kaggle competitions</li>
                    <li>Credit scoring</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Neural Networks */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3 flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                4.4 Neural Networks
              </h3>
              <p className="text-muted-foreground mb-2">
                Used for <span className="font-medium text-foreground">complex pattern learning</span>
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Types</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>MLP (Tabular data)</li>
                    <li>CNN (Images)</li>
                    <li>RNN (Sequences)</li>
                    <li>Transformers (NLP)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Regression Algorithms */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <LineChart className="h-5 w-5 text-primary" />
            5. Regression Algorithms
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="font-semibold text-foreground text-lg mb-3">Linear Regression</h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-foreground text-sm">Assumptions</h4>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Linear relationship</li>
                  <li>Independence of errors</li>
                  <li>Constant variance (homoscedasticity)</li>
                </ul>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                  <p className="font-semibold text-foreground text-xs">Pros</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Simple and fast</li>
                    <li>Highly interpretable</li>
                  </ul>
                </div>
                <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                  <p className="font-semibold text-foreground text-xs">Cons</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Sensitive to outliers</li>
                    <li>Cannot capture complex patterns</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Classification Algorithms */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <PieChart className="h-5 w-5 text-primary" />
            6. Classification Algorithms
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <h3 className="font-semibold text-foreground text-lg mb-3">Logistic Regression</h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-foreground text-sm">Assumptions</h4>
                <ul className="text-xs text-muted-foreground list-disc list-inside">
                  <li>Binary classification</li>
                  <li>Linear decision boundary</li>
                </ul>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                  <p className="font-semibold text-foreground text-xs">Pros</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Outputs probabilities</li>
                    <li>Easy to interpret</li>
                  </ul>
                </div>
                <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                  <p className="font-semibold text-foreground text-xs">Cons</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Limited to linear boundaries</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Key Concepts */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            7. Key Concepts in Supervised Learning
          </h2>

          <div className="space-y-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Bias-Variance Tradeoff</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-3 bg-destructive/10 border border-destructive/20 rounded">
                  <p className="font-semibold text-foreground text-sm">High Bias → Underfitting</p>
                  <p className="text-xs text-muted-foreground">Model is too simple, misses patterns</p>
                </div>
                <div className="p-3 bg-destructive/10 border border-destructive/20 rounded">
                  <p className="font-semibold text-foreground text-sm">High Variance → Overfitting</p>
                  <p className="text-xs text-muted-foreground">Model is too complex, captures noise</p>
                </div>
              </div>
              <div className="mt-3 bg-muted p-3 rounded">
                <p className="text-sm font-mono text-center">Error = Bias² + Variance + Noise</p>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Regularization</h4>
                <p className="text-xs text-muted-foreground">Prevents overfitting:</p>
                <ul className="text-xs text-muted-foreground list-disc list-inside mt-1">
                  <li>L1 (Lasso)</li>
                  <li>L2 (Ridge)</li>
                  <li>Elastic Net</li>
                </ul>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Gradient Descent</h4>
                <p className="text-xs text-muted-foreground">Optimization technique:</p>
                <ul className="text-xs text-muted-foreground list-disc list-inside mt-1">
                  <li>Batch GD</li>
                  <li>Stochastic GD</li>
                  <li>Mini-batch GD</li>
                </ul>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <h4 className="font-semibold text-foreground mb-2">Cross Validation</h4>
                <p className="text-xs text-muted-foreground">Better evaluation:</p>
                <ul className="text-xs text-muted-foreground list-disc list-inside mt-1">
                  <li>K-Fold CV</li>
                  <li>Stratified CV</li>
                  <li>Time Series CV</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Evaluation Metrics */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            8. Evaluation Metrics
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Regression Metrics</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li><span className="font-medium text-foreground">MAE</span>: Mean Absolute Error</li>
                <li><span className="font-medium text-foreground">MSE</span>: Mean Squared Error</li>
                <li><span className="font-medium text-foreground">RMSE</span>: Root Mean Squared Error</li>
                <li><span className="font-medium text-foreground">R² Score</span>: Coefficient of determination</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Classification Metrics</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li><span className="font-medium text-foreground">Accuracy</span>: Overall correctness</li>
                <li><span className="font-medium text-foreground">Precision</span>: Quality of positive predictions</li>
                <li><span className="font-medium text-foreground">Recall</span>: Coverage of positive predictions</li>
                <li><span className="font-medium text-foreground">F1 Score</span>: Harmonic mean of precision/recall</li>
                <li><span className="font-medium text-foreground">ROC-AUC</span>: Area under ROC curve</li>
                <li><span className="font-medium text-foreground">Log Loss</span>: Probability-based loss</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 9. Model Selection Guide */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            9. Model Selection Guide
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-medium text-foreground">Small dataset</span>
                    <p className="text-muted-foreground text-xs">→ Linear / Logistic Regression</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-medium text-foreground">Large dataset</span>
                    <p className="text-muted-foreground text-xs">→ Random Forest / XGBoost</p>
                  </div>
                </li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-medium text-foreground">Images/Text</span>
                    <p className="text-muted-foreground text-xs">→ Deep Learning</p>
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-medium text-foreground">Real-time</span>
                    <p className="text-muted-foreground text-xs">→ Simple models (Logistic, Naive Bayes)</p>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* 10. Common Problems */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            10. Common Problems
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground text-sm">Overfitting</h4>
                <p className="text-xs text-muted-foreground">Model too complex</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground text-sm">Underfitting</h4>
                <p className="text-xs text-muted-foreground">Model too simple</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground text-sm">Data Leakage</h4>
                <p className="text-xs text-muted-foreground">Test info in training</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground text-sm">Imbalanced Data</h4>
                <p className="text-xs text-muted-foreground">Unequal class distribution</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground text-sm">Missing Data</h4>
                <p className="text-xs text-muted-foreground">Incomplete records</p>
              </div>
            </div>
          </div>
        </section>

        {/* 11. Supervised Learning Workflow */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            11. Supervised Learning Workflow
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">1</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Collection</h4>
                    <p className="text-xs text-muted-foreground">Gather relevant data</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">2</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Cleaning</h4>
                    <p className="text-xs text-muted-foreground">Handle missing values and outliers</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">3</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Engineering</h4>
                    <p className="text-xs text-muted-foreground">Create and transform features</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">4</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Train-Test Split</h4>
                    <p className="text-xs text-muted-foreground">Separate data for evaluation</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">5</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Model Training</h4>
                    <p className="text-xs text-muted-foreground">Train algorithm on training data</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">6</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Evaluation</h4>
                    <p className="text-xs text-muted-foreground">Assess performance on test data</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">7</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Hyperparameter Tuning</h4>
                    <p className="text-xs text-muted-foreground">Optimize model configuration</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">8</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Deployment</h4>
                    <p className="text-xs text-muted-foreground">Deploy model to production</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 12. Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            12. Best Practices
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Always split dataset properly</h4>
                <p className="text-sm text-muted-foreground">Use 70-80% for training, 20-30% for testing</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Normalize features when needed</h4>
                <p className="text-sm text-muted-foreground">Scale features for distance-based algorithms</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use cross-validation</h4>
                <p className="text-sm text-muted-foreground">K-Fold CV for robust evaluation</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Monitor overfitting</h4>
                <p className="text-sm text-muted-foreground">Compare train vs validation performance</p>
              </div>
            </div>
          </div>
        </section>

        {/* 13. Applications */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            13. Applications
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• <span className="font-medium text-foreground">Fraud detection</span></li>
                <li>• <span className="font-medium text-foreground">Medical diagnosis</span></li>
                <li>• <span className="font-medium text-foreground">Recommendation systems</span></li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• <span className="font-medium text-foreground">Spam filtering</span></li>
                <li>• <span className="font-medium text-foreground">Image recognition</span></li>
                <li>• <span className="font-medium text-foreground">Sentiment analysis</span></li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• <span className="font-medium text-foreground">Stock prediction</span></li>
                <li>• <span className="font-medium text-foreground">Customer churn</span></li>
                <li>• <span className="font-medium text-foreground">Credit scoring</span></li>
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
            See supervised learning algorithms in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Supervised Learning Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}