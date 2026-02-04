// app/learn/machine-learning/introduction/page.tsx
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Target, Zap, Network, Layers, Hash, Flame, Brain, Cpu, BarChart, LineChart, PieChart, Database, GitBranch, Code, TestTube, Rocket, Shield, Cloud, TrendingUp, Bot, CpuIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export default function MachineLearningIntroPage() {
  const mlCategories = [
    {
      title: "Supervised Learning",
      icon: Target,
      description: "Learn from labeled data with input-output pairs",
      algorithms: ["Linear Regression", "Logistic Regression", "Decision Trees", "SVM", "Neural Networks"]
    },
    {
      title: "Unsupervised Learning",
      icon: Network,
      description: "Find patterns in unlabeled data",
      algorithms: ["Clustering (K-Means)", "Dimensionality Reduction (PCA)", "Anomaly Detection", "Association Rules"]
    },
    {
      title: "Reinforcement Learning",
      icon: Zap,
      description: "Learn through trial and error with rewards",
      algorithms: ["Q-Learning", "Deep Q Networks", "Policy Gradient", "Actor-Critic"]
    },
    {
      title: "Deep Learning",
      icon: Brain,
      description: "Multi-layer neural networks for complex patterns",
      algorithms: ["CNNs", "RNNs", "Transformers", "GANs", "Autoencoders"]
    }
  ];

  const mlWorkflow = [
    {
      step: 1,
      title: "Problem Definition",
      description: "Define the business problem and success metrics",
      tasks: ["Identify objectives", "Define success metrics", "Determine feasibility"],
      icon: Target
    },
    {
      step: 2,
      title: "Data Collection",
      description: "Gather and aggregate data from various sources",
      tasks: ["Collect datasets", "Merge data sources", "Initial data exploration"],
      icon: Database
    },
    {
      step: 3,
      title: "Data Preparation",
      description: "Clean, transform, and preprocess the data",
      tasks: ["Handle missing values", "Feature engineering", "Normalization/Scaling"],
      icon: Cpu
    },
    {
      step: 4,
      title: "Model Selection",
      description: "Choose appropriate algorithms for the problem",
      tasks: ["Algorithm selection", "Baseline models", "Architecture design"],
      icon: GitBranch
    },
    {
      step: 5,
      title: "Model Training",
      description: "Train models using training data",
      tasks: ["Split data", "Train models", "Hyperparameter tuning"],
      icon: Rocket
    },
    {
      step: 6,
      title: "Model Evaluation",
      description: "Evaluate model performance on test data",
      tasks: ["Performance metrics", "Error analysis", "Model comparison"],
      icon: TestTube
    },
    {
      step: 7,
      title: "Model Deployment",
      description: "Deploy model to production environment",
      tasks: ["API development", "Monitoring", "Maintenance"],
      icon: Cloud
    },
    {
      step: 8,
      title: "Monitoring & Maintenance",
      description: "Monitor performance and update models",
      tasks: ["Performance tracking", "Model retraining", "Drift detection"],
      icon: Shield
    }
  ];

  const mlAlgorithms = [
    {
      name: "Linear Regression",
      type: "Supervised",
      task: "Regression",
      complexity: "O(n²p) where n=samples, p=features",
      code: `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Feature
y = np.array([2, 4, 5, 4, 5])            # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")`
    },
    {
      name: "Logistic Regression",
      type: "Supervised",
      task: "Classification",
      complexity: "O(n*p) per iteration",
      code: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data: 2 features, binary classification
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
y = np.array([0, 0, 0, 1, 1, 1])  # Binary labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\\nConfusion Matrix:")
print(conf_matrix)
print("\\nClassification Report:")
print(report)
print("\\nPredicted probabilities for first test sample:")
print(y_pred_proba[0])`
    },
    {
      name: "K-Means Clustering",
      type: "Unsupervised",
      task: "Clustering",
      complexity: "O(n*k*p*iterations)",
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Determine optimal k using elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Apply K-Means with optimal k (let's say k=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Evaluate clustering
silhouette_avg = silhouette_score(X, y_pred)

# Plot clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.8, marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.tight_layout()
plt.show()

print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Centroids:\\n{centroids}")`
    },
    {
      name: "Decision Tree Classifier",
      type: "Supervised",
      task: "Classification",
      complexity: "O(n*p*log(n)) for balanced tree",
      code: `import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = DecisionTreeClassifier(
    max_depth=3,          # Limit tree depth
    min_samples_split=2,  # Minimum samples to split node
    min_samples_leaf=1,   # Minimum samples in leaf
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(report)
print("\\nFeature Importances:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")`
    }
  ];

  const essentialConcepts = [
    {
      concept: "Training, Validation, Test Split",
      description: "Splitting data to avoid overfitting and evaluate model performance",
      formula: "Typically 70% train, 15% validation, 15% test",
      importance: "Critical for proper model evaluation"
    },
    {
      concept: "Overfitting vs Underfitting",
      description: "Overfitting: Model learns noise. Underfitting: Model too simple.",
      formula: "Bias-Variance Tradeoff",
      importance: "Key to model generalization"
    },
    {
      concept: "Cross-Validation",
      description: "k-fold validation to get robust performance estimates",
      formula: "k-fold CV, Stratified k-fold for classification",
      importance: "Better utilization of data"
    },
    {
      concept: "Feature Engineering",
      description: "Creating new features from existing data",
      formula: "Domain knowledge + Data transformation",
      importance: "Often more important than algorithm choice"
    },
    {
      concept: "Hyperparameter Tuning",
      description: "Optimizing model parameters that aren't learned",
      formula: "Grid Search, Random Search, Bayesian Optimization",
      importance: "Critical for model performance"
    }
  ];

  const evaluationMetrics = [
    {
      type: "Regression",
      metrics: [
        { name: "Mean Absolute Error (MAE)", formula: "Σ|yᵢ - ŷᵢ|/n", interpretation: "Average absolute error" },
        { name: "Mean Squared Error (MSE)", formula: "Σ(yᵢ - ŷᵢ)²/n", interpretation: "Penalizes large errors" },
        { name: "R² Score", formula: "1 - (SS_res/SS_tot)", interpretation: "Variance explained" },
        { name: "Root Mean Squared Error (RMSE)", formula: "√MSE", interpretation: "In original units" }
      ]
    },
    {
      type: "Classification",
      metrics: [
        { name: "Accuracy", formula: "(TP+TN)/(TP+TN+FP+FN)", interpretation: "Overall correctness" },
        { name: "Precision", formula: "TP/(TP+FP)", interpretation: "Correct positive predictions" },
        { name: "Recall", formula: "TP/(TP+FN)", interpretation: "Actual positives identified" },
        { name: "F1-Score", formula: "2*(Precision*Recall)/(Precision+Recall)", interpretation: "Harmonic mean" },
        { name: "ROC-AUC", formula: "Area under ROC curve", interpretation: "Overall performance" }
      ]
    },
    {
      type: "Clustering",
      metrics: [
        { name: "Silhouette Score", formula: "(b-a)/max(a,b)", interpretation: "Cohesion vs separation" },
        { name: "Davies-Bouldin Index", formula: "Average similarity", interpretation: "Lower is better" },
        { name: "Calinski-Harabasz Index", formula: "Between variance/Within variance", interpretation: "Higher is better" }
      ]
    }
  ];

  const realWorldApplications = [
    {
      industry: "Healthcare",
      icon: Shield,
      applications: ["Disease diagnosis", "Drug discovery", "Medical imaging analysis", "Personalized treatment"],
      example: "CNN for detecting tumors in MRI scans"
    },
    {
      industry: "Finance",
      icon: TrendingUp,
      applications: ["Fraud detection", "Algorithmic trading", "Credit scoring", "Risk assessment"],
      example: "Anomaly detection for credit card fraud"
    },
    {
      industry: "E-commerce",
      icon: ShoppingCart,
      applications: ["Recommendation systems", "Customer segmentation", "Price optimization", "Demand forecasting"],
      example: "Collaborative filtering for product recommendations"
    },
    {
      industry: "Autonomous Vehicles",
      icon: Car,
      applications: ["Object detection", "Path planning", "Traffic prediction", "Driver monitoring"],
      example: "YOLO for real-time object detection"
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the main difference between supervised and unsupervised learning?",
      options: [
        "Supervised uses GPUs, unsupervised uses CPUs",
        "Supervised uses labeled data, unsupervised finds patterns in unlabeled data",
        "Supervised is faster, unsupervised is more accurate",
        "Supervised is for regression, unsupervised is for classification"
      ],
      correctAnswer: 1,
      explanation: "Supervised learning uses labeled data with known outputs, while unsupervised learning finds patterns in data without predefined labels.",
    },
    {
      id: 2,
      question: "Which of these is NOT a common evaluation metric for classification problems?",
      options: ["Accuracy", "Mean Squared Error", "Precision", "F1-Score"],
      correctAnswer: 1,
      explanation: "Mean Squared Error is primarily used for regression problems, not classification.",
    },
    {
      id: 3,
      question: "What does overfitting mean in machine learning?",
      options: [
        "The model is too simple and cannot learn patterns",
        "The model learns the training data too well, including noise",
        "The model takes too long to train",
        "The model uses too many features"
      ],
      correctAnswer: 1,
      explanation: "Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor generalization to new data.",
    },
    {
      id: 4,
      question: "What is the purpose of a validation set?",
      options: [
        "To test the final model performance",
        "To tune hyperparameters during training",
        "To train the initial model",
        "To visualize the data"
      ],
      correctAnswer: 1,
      explanation: "The validation set is used to tune hyperparameters and make decisions during model development, while the test set is reserved for final evaluation.",
    },
    {
      id: 5,
      question: "Which algorithm would you choose for a recommendation system?",
      options: ["Linear Regression", "K-Means", "Collaborative Filtering", "Decision Tree"],
      correctAnswer: 2,
      explanation: "Collaborative filtering is commonly used in recommendation systems to predict user preferences based on similar users' behavior.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Introduction to Machine Learning
          </h1>
          <p className="text-lg text-muted-foreground">
            Complete beginner's guide to machine learning concepts, algorithms, and real-world applications. Learn how computers learn from data.
          </p>
        </header>

        {/* Hero Section */}
        <section className="mb-12">
          <Card className="bg-linear-to-r from-primary/10 to-purple-500/10 border-primary/20">
            <CardContent className="p-8">
              <div className="flex flex-col md:flex-row items-center gap-8">
                <div className="md:w-2/3">
                  <h2 className="text-2xl font-bold mb-4 text-foreground">
                    What is Machine Learning?
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary" className="gap-1">
                      <Brain className="h-3 w-3" />
                      Pattern Recognition
                    </Badge>
                    <Badge variant="secondary" className="gap-1">
                      <Database className="h-3 w-3" />
                      Data-Driven
                    </Badge>
                    <Badge variant="secondary" className="gap-1">
                      <TrendingUp className="h-3 w-3" />
                      Predictive Analytics
                    </Badge>
                    <Badge variant="secondary" className="gap-1">
                      <Bot className="h-3 w-3" />
                      AI Foundation
                    </Badge>
                  </div>
                </div>
                <div className="md:w-1/3">
                  <div className="bg-primary/5 p-6 rounded-lg">
                    <h3 className="font-semibold mb-3 text-foreground">Key Characteristics</h3>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                        <span>Learns from data</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                        <span>Improves with experience</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                        <span>Makes data-driven predictions</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                        <span>Automates decision making</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* ML Categories */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Types of Machine Learning</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {mlCategories.map((category, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <category.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{category.title}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    {category.description}
                  </p>
                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide">Common Algorithms:</h4>
                    <ul className="space-y-1">
                      {category.algorithms.map((algo, aIdx) => (
                        <li key={aIdx} className="flex items-center gap-2 text-sm">
                          <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                          <span className="text-muted-foreground">{algo}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* ML Workflow */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Machine Learning Workflow</h2>
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-8 md:left-1/2 transform md:-translate-x-1/2 h-full w-1 bg-primary/20"></div>
            
            <div className="space-y-12">
              {mlWorkflow.map((step, idx) => (
                <div key={idx} className={`relative flex ${idx % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'} items-start gap-6`}>
                  {/* Step number */}
                  <div className="absolute left-6 md:left-1/2 transform md:-translate-x-1/2 z-10">
                    <div className="h-12 w-12 rounded-full bg-primary flex items-center justify-center border-4 border-background">
                      <span className="font-bold text-primary-foreground">{step.step}</span>
                    </div>
                  </div>
                  
                  {/* Content */}
                  <div className={`ml-20 md:ml-0 md:w-5/12 ${idx % 2 === 0 ? 'md:pr-12' : 'md:pl-12'}`}>
                    <Card className="hover:shadow-lg transition-shadow">
                      <CardContent className="p-6">
                        <div className="flex items-center gap-3 mb-3">
                          <step.icon className="h-5 w-5 text-primary" />
                          <h3 className="font-semibold text-lg text-foreground">{step.title}</h3>
                        </div>
                        <p className="text-sm text-muted-foreground mb-4">
                          {step.description}
                        </p>
                        <div>
                          <h4 className="text-xs font-semibold text-foreground mb-2 uppercase tracking-wide">Key Tasks:</h4>
                          <ul className="space-y-1">
                            {step.tasks.map((task, tIdx) => (
                              <li key={tIdx} className="flex items-start gap-2 text-sm">
                                <CheckCircle2 className="h-3 w-3 text-green-500 mt-0.5 shrink-0" />
                                <span className="text-muted-foreground">{task}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Algorithm Implementations */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Essential ML Algorithms</h2>
          <Tabs defaultValue="linear" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="linear">Linear Regression</TabsTrigger>
              <TabsTrigger value="logistic">Logistic Regression</TabsTrigger>
              <TabsTrigger value="kmeans">K-Means</TabsTrigger>
              <TabsTrigger value="decision">Decision Tree</TabsTrigger>
            </TabsList>
            
            {mlAlgorithms.map((algo, idx) => (
              <TabsContent key={idx} value={algo.name.toLowerCase().replace(' ', '')}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                      <div className="lg:w-1/3">
                        <div className="flex items-center gap-3 mb-4">
                          <Badge variant="outline">{algo.type}</Badge>
                          <Badge variant="secondary">{algo.task}</Badge>
                        </div>
                        <h3 className="font-semibold text-lg text-foreground mb-2">{algo.name}</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                          {algo.name === "Linear Regression" && "Predicts continuous values using linear relationship between features and target"}
                          {algo.name === "Logistic Regression" && "Predicts probability of categorical outcomes using logistic function"}
                          {algo.name === "K-Means Clustering" && "Partitions data into k clusters based on feature similarity"}
                          {algo.name === "Decision Tree Classifier" && "Tree-like model for classification using feature thresholds"}
                        </p>
                        <div className="space-y-3">
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-1">Time Complexity:</h4>
                            <code className="text-xs bg-muted px-2 py-1 rounded">{algo.complexity}</code>
                          </div>
                          <div>
                            <h4 className="font-medium text-sm text-foreground mb-1">Best For:</h4>
                            <p className="text-sm text-muted-foreground">
                              {algo.name === "Linear Regression" && "Numerical prediction, trend analysis"}
                              {algo.name === "Logistic Regression" && "Binary/multiclass classification, probability estimation"}
                              {algo.name === "K-Means Clustering" && "Customer segmentation, image compression, anomaly detection"}
                              {algo.name === "Decision Tree Classifier" && "Interpretable models, non-linear relationships"}
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="lg:w-2/3">
                        <CodeBlock
                          language="python"
                          code={algo.code}
                          filename={`${algo.name.replace(' ', '_').toLowerCase()}.py`}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Essential Concepts */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Essential ML Concepts</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {essentialConcepts.map((concept, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                      <span className="font-bold text-primary">{idx + 1}</span>
                    </div>
                    <h3 className="font-semibold text-foreground">{concept.concept}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    {concept.description}
                  </p>
                  <div className="space-y-2">
                    <div className="bg-muted p-3 rounded">
                      <p className="text-xs font-medium text-foreground">Formula/Method:</p>
                      <p className="text-sm font-mono mt-1">{concept.formula}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-warning" />
                      <span className="text-xs text-muted-foreground">{concept.importance}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Evaluation Metrics */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Evaluation Metrics</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {evaluationMetrics.map((category, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <h3 className="font-semibold mb-4 text-foreground">{category.type} Metrics</h3>
                  <div className="space-y-4">
                    {category.metrics.map((metric, mIdx) => (
                      <div key={mIdx} className="border-b pb-3 last:border-0 last:pb-0">
                        <div className="flex justify-between items-start mb-1">
                          <h4 className="font-medium text-sm text-foreground">{metric.name}</h4>
                          <Badge variant="outline" className="text-xs">Formula</Badge>
                        </div>
                        <code className="text-xs bg-muted block p-2 rounded mb-2">{metric.formula}</code>
                        <p className="text-xs text-muted-foreground">{metric.interpretation}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Real-world Applications */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Real-world Applications</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {realWorldApplications.map((app, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <app.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{app.industry}</h3>
                  </div>
                  <div className="space-y-3">
                    <div>
                      <h4 className="text-xs font-semibold text-foreground mb-2 uppercase tracking-wide">Applications:</h4>
                      <ul className="space-y-1">
                        {app.applications.map((application, aIdx) => (
                          <li key={aIdx} className="flex items-center gap-2 text-sm">
                            <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                            <span className="text-muted-foreground">{application}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div className="bg-primary/5 border border-primary/20 p-3 rounded">
                      <p className="text-xs font-medium text-foreground">Example:</p>
                      <p className="text-xs text-muted-foreground mt-1">{app.example}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Tools & Libraries */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Essential ML Tools & Libraries</h2>
          <Card>
            <CardContent className="p-6">
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div>
                  <h3 className="font-semibold mb-3 text-foreground">Python Libraries</h3>
                  <ul className="space-y-2">
                    <li className="flex items-center justify-between">
                      <span className="text-sm">scikit-learn</span>
                      <Badge variant="outline">Classical ML</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">TensorFlow</span>
                      <Badge variant="outline">Deep Learning</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">PyTorch</span>
                      <Badge variant="outline">Research DL</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">XGBoost</span>
                      <Badge variant="outline">Gradient Boosting</Badge>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-3 text-foreground">Data Processing</h3>
                  <ul className="space-y-2">
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Pandas</span>
                      <Badge variant="outline">DataFrames</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">NumPy</span>
                      <Badge variant="outline">Numerical</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Matplotlib</span>
                      <Badge variant="outline">Plotting</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Seaborn</span>
                      <Badge variant="outline">Statistics</Badge>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-3 text-foreground">Deployment</h3>
                  <ul className="space-y-2">
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Flask/FastAPI</span>
                      <Badge variant="outline">APIs</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Docker</span>
                      <Badge variant="outline">Containers</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">MLflow</span>
                      <Badge variant="outline">Tracking</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Kubernetes</span>
                      <Badge variant="outline">Orchestration</Badge>
                    </li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-3 text-foreground">Cloud Platforms</h3>
                  <ul className="space-y-2">
                    <li className="flex items-center justify-between">
                      <span className="text-sm">AWS SageMaker</span>
                      <Badge variant="outline">AWS</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Azure ML</span>
                      <Badge variant="outline">Azure</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">GCP Vertex AI</span>
                      <Badge variant="outline">Google</Badge>
                    </li>
                    <li className="flex items-center justify-between">
                      <span className="text-sm">Databricks</span>
                      <Badge variant="outline">Spark</Badge>
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Test Your ML Knowledge</h2>
          <Quiz questions={quizQuestions} title="Machine Learning Fundamentals Quiz" />
        </section>

        {/* Getting Started */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Getting Started with ML</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Learning Path</h3>
                <ol className="space-y-3">
                  <li className="flex items-start gap-3">
                    <div className="h-6 w-6 rounded-full bg-primary flex items-center justify-center shrink-0">
                      <span className="text-xs font-bold text-primary-foreground">1</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Python & Statistics</h4>
                      <p className="text-xs text-muted-foreground">Learn Python, NumPy, Pandas, basic statistics</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="h-6 w-6 rounded-full bg-primary flex items-center justify-center shrink-0">
                      <span className="text-xs font-bold text-primary-foreground">2</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">scikit-learn Basics</h4>
                      <p className="text-xs text-muted-foreground">Start with Linear/Logistic Regression, Decision Trees</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="h-6 w-6 rounded-full bg-primary flex items-center justify-center shrink-0">
                      <span className="text-xs font-bold text-primary-foreground">3</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Intermediate Concepts</h4>
                      <p className="text-xs text-muted-foreground">Cross-validation, hyperparameter tuning, pipelines</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="h-6 w-6 rounded-full bg-primary flex items-center justify-center shrink-0">
                      <span className="text-xs font-bold text-primary-foreground">4</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Deep Learning</h4>
                      <p className="text-xs text-muted-foreground">Neural Networks, CNNs, RNNs with TensorFlow/PyTorch</p>
                    </div>
                  </li>
                </ol>
              </CardContent>
            </Card>
            
            <Card className="bg-green-500/5 border-green-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Project Ideas for Beginners</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <div>
                      <h4 className="font-medium text-sm text-foreground">House Price Prediction</h4>
                      <p className="text-xs text-muted-foreground">Use Linear Regression with real estate data</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Iris Flower Classification</h4>
                      <p className="text-xs text-muted-foreground">Classify flower species with scikit-learn</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Spam Email Detection</h4>
                      <p className="text-xs text-muted-foreground">Build a spam filter using Naive Bayes</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Customer Segmentation</h4>
                      <p className="text-xs text-muted-foreground">Use K-Means for market segmentation</p>
                    </div>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Common Mistakes & Tips */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Common Mistakes & Best Practices</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-warning/5 border-warning/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <AlertCircle className="h-5 w-5 text-warning" />
                  <h3 className="font-semibold text-foreground">Common Mistakes</h3>
                </div>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-warning mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Data Leakage</h4>
                      <p className="text-xs text-muted-foreground">Using test data during training or preprocessing</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-warning mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Ignoring Class Imbalance</h4>
                      <p className="text-xs text-muted-foreground">Not handling imbalanced datasets in classification</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-warning mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Over-reliance on Accuracy</h4>
                      <p className="text-xs text-muted-foreground">Using accuracy for imbalanced classification problems</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-warning mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Not Scaling Features</h4>
                      <p className="text-xs text-muted-foreground">Forgetting to scale features for distance-based algorithms</p>
                    </div>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-blue-500/5 border-blue-500/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle2 className="h-5 w-5 text-blue-500" />
                  <h3 className="font-semibold text-foreground">Best Practices</h3>
                </div>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-blue-500 mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Always Use Cross-Validation</h4>
                      <p className="text-xs text-muted-foreground">k-fold CV provides more reliable performance estimates</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-blue-500 mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Start Simple</h4>
                      <p className="text-xs text-muted-foreground">Begin with simple models before trying complex ones</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-blue-500 mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Feature Engineering  Algorithm</h4>
                      <p className="text-xs text-muted-foreground">Good features often matter more than algorithm choice</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-blue-500 mt-1.5 shrink-0"></div>
                    <div>
                      <h4 className="font-medium text-sm text-foreground">Monitor for Drift</h4>
                      <p className="text-xs text-muted-foreground">Monitor model performance and retrain as data changes</p>
                    </div>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Quick Reference */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">ML Quick Reference</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Algorithm Selection Guide</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between items-center">
                    <span>Linear/Logistic Regression</span>
                    <Badge variant="outline">Baseline</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Decision Trees/Random Forest</span>
                    <Badge variant="outline">Interpretable</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>XGBoost/LightGBM</span>
                    <Badge variant="outline">Tabular Data</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Neural Networks</span>
                    <Badge variant="outline">Complex Patterns</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>K-Means</span>
                    <Badge variant="outline">Clustering</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">When to Use What</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-green-500"></div>
                    <span><strong>Structured data:</strong> XGBoost, Random Forest</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-blue-500"></div>
                    <span><strong>Images:</strong> CNNs (ResNet, VGG)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-purple-500"></div>
                    <span><strong>Text/NLP:</strong> Transformers, RNNs</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-orange-500"></div>
                    <span><strong>Time Series:</strong> LSTM, ARIMA</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-red-500"></div>
                    <span><strong>Recommendations:</strong> Collaborative Filtering</span>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Essential Math Concepts</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span>Linear Algebra</span>
                    <Badge variant="outline">Matrices, Vectors</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Calculus</span>
                    <Badge variant="outline">Derivatives, Gradients</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Probability</span>
                    <Badge variant="outline">Distributions, Bayes</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Statistics</span>
                    <Badge variant="outline">Hypothesis Testing</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Optimization</span>
                    <Badge variant="outline">Gradient Descent</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </div>
  );
}


const ShoppingCart = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
  </svg>
);

const Car = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
  </svg>
);