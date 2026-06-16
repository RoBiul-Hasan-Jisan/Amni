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
  Database,
  BarChart3,
  TrendingUp,
  Target,
  Layers,
  Cpu,
  Shield,
  Sparkles,
  GitBranch,
  Globe,
  Activity,
  PieChart,
  FileText,
  Image,
  Mic,
  Server,
  Code2
} from "lucide-react";

export default function MlIntroPage() {
  const result = getSubtopicBySlug("machine-learning", "ml-intro");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-sklearn",
      label: "Traditional Programming vs ML",
      code: `# Traditional Programming
def is_spam(email):
    # Explicit rules
    if "win" in email and "free" in email:
        return True
    if "click" in email and "prize" in email:
        return True
    return False

# Machine Learning Approach
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data (features and labels)
emails = ["win free prize", "meeting at 3pm", "click here to win"]
labels = [1, 0, 1]  # 1 = spam, 0 = not spam

# Learn patterns from data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# Predict new email
new_email = ["free money"]
X_new = vectorizer.transform(new_email)
prediction = model.predict(X_new)
print(f"Prediction: {'Spam' if prediction[0] else 'Not Spam'}")`,
    },
    {
      language: "python-tensorflow",
      label: "Classification Example",
      code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: species of iris

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Predicted species: {iris.target_names[y_pred]}")
print(f"Actual species: {iris.target_names[y_test]}")`,
    },
    {
      language: "python-pytorch",
      label: "Regression Example",
      code: `from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()
X = diabetes.data  # Features
y = diabetes.target  # Target: disease progression

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Coefficients: {model.coef_[:3]}...")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the main difference between traditional programming and Machine Learning?",
      options: [
        "ML is faster",
        "ML learns patterns from data, traditional programming uses explicit rules",
        "ML uses more memory",
        "They are the same thing"
      ],
      correctAnswer: 1,
      explanation: "Traditional programming involves writing explicit rules, while ML learns patterns from data automatically.",
    },
    {
      id: 2,
      question: "Which type of ML uses labeled data?",
      options: [
        "Unsupervised Learning",
        "Supervised Learning",
        "Reinforcement Learning",
        "All of the above"
      ],
      correctAnswer: 1,
      explanation: "Supervised Learning uses labeled data where both input features and correct outputs are provided.",
    },
    {
      id: 3,
      question: "What is the difference between features and labels?",
      options: [
        "They are the same thing",
        "Features are inputs, labels are outputs",
        "Labels are inputs, features are outputs",
        "Features are numbers, labels are text"
      ],
      correctAnswer: 1,
      explanation: "Features are the input variables used to make predictions, and labels are the target variables we want to predict.",
    },
    {
      id: 4,
      question: "What is overfitting in ML?",
      options: [
        "Model performs well on all data",
        "Model performs well on training data but poorly on new data",
        "Model performs poorly on training data",
        "Model is too simple"
      ],
      correctAnswer: 1,
      explanation: "Overfitting occurs when a model learns training data too well, capturing noise instead of general patterns.",
    },
    {
      id: 5,
      question: "Which metric is used for evaluating classification models?",
      options: [
        "Mean Squared Error (MSE)",
        "Accuracy",
        "R² Score",
        "All of the above"
      ],
      correctAnswer: 1,
      explanation: "Accuracy is a primary metric for classification, while MSE and R² are regression metrics.",
    },
    {
      id: 6,
      question: "What is the purpose of train-test split?",
      options: [
        "To reduce file size",
        "To evaluate model performance on unseen data",
        "To make training faster",
        "To visualize data"
      ],
      correctAnswer: 1,
      explanation: "Train-test split helps evaluate how well the model generalizes to new, unseen data.",
    },
    {
      id: 7,
      question: "Which ML type is used for clustering?",
      options: [
        "Supervised Learning",
        "Unsupervised Learning",
        "Reinforcement Learning",
        "Deep Learning"
      ],
      correctAnswer: 1,
      explanation: "Clustering is an unsupervised learning technique used to find patterns in unlabeled data.",
    },
    {
      id: 8,
      question: "What is the bias-variance tradeoff?",
      options: [
        "Choosing between two models",
        "Balancing model simplicity and complexity",
        "Choosing between two datasets",
        "A programming technique"
      ],
      correctAnswer: 1,
      explanation: "The bias-variance tradeoff balances model complexity (variance) against model simplicity (bias).",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
       

        {/* 1. What is Machine Learning? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            1. What is Machine Learning?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Definition of Machine Learning</h3>
              <p className="text-muted-foreground text-sm">
                Machine Learning is a subset of Artificial Intelligence that enables systems to learn 
                and improve from experience without being explicitly programmed. It focuses on developing 
                computer programs that can access data and use it to learn for themselves.
              </p>
              <div className="mt-3 p-3 bg-muted rounded text-xs">
                <p className="font-semibold text-foreground">Arthur Samuel's Definition (1959):</p>
                <p className="text-muted-foreground">"Field of study that gives computers the ability to learn without being explicitly programmed."</p>
              </div>
            </div>
            
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">History of Machine Learning</h3>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li><span className="font-medium text-foreground">1950s:</span> Alan Turing proposes the Turing Test</li>
                <li><span className="font-medium text-foreground">1959:</span> Arthur Samuel creates first ML program</li>
                <li><span className="font-medium text-foreground">1980s:</span> Neural networks gain popularity</li>
                <li><span className="font-medium text-foreground">1990s:</span> Support Vector Machines (SVM) developed</li>
                <li><span className="font-medium text-foreground">2010s:</span> Deep learning revolution</li>
                <li><span className="font-medium text-foreground">2020s:</span> Generative AI and LLMs</li>
              </ul>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Brain className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground text-sm">AI vs ML vs Deep Learning</h4>
                  <p className="text-xs text-muted-foreground">
                    <span className="font-medium">AI</span>: Broad concept of smart machines<br/>
                    <span className="font-medium">ML</span>: Subset of AI, learns from data<br/>
                    <span className="font-medium">Deep Learning</span>: Subset of ML, uses neural networks
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Globe className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Real-World Applications</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Recommendation systems</li>
                    <li>Fraud detection</li>
                    <li>Medical diagnosis</li>
                    <li>Self-driving cars</li>
                    <li>Voice assistants</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Activity className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Key Impact Areas</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Healthcare</li>
                    <li>Finance</li>
                    <li>Manufacturing</li>
                    <li>Retail</li>
                    <li>Transportation</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Why Machine Learning? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            2. Why Machine Learning?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Traditional Programming vs ML</h3>
              <div className="space-y-2 text-sm">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-primary">Traditional:</p>
                  <p className="text-muted-foreground">Data + Rules → Output</p>
                </div>
                <div className="bg-primary/10 p-2 rounded">
                  <p className="font-mono text-primary">Machine Learning:</p>
                  <p className="text-muted-foreground">Data + Output → Rules</p>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  ML automatically discovers patterns and rules from data, eliminating manual rule creation.
                </p>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Benefits & Limitations</h3>
              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground text-sm">Benefits</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Automates decision-making</li>
                      <li>Handles complex patterns</li>
                      <li>Improves over time</li>
                      <li>Processes large datasets</li>
                    </ul>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground text-sm">Limitations</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Needs quality data</li>
                      <li>Can be biased</li>
                      <li>Black-box models</li>
                      <li>Computationally expensive</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">When to Use ML?</h4>
                <ul className="text-sm text-muted-foreground grid md:grid-cols-2 gap-2">
                  <li>• Too complex for explicit rules</li>
                  <li>• Patterns change over time</li>
                  <li>• Need to handle large-scale data</li>
                  <li>• Want adaptive solutions</li>
                  <li>• Complex pattern recognition</li>
                  <li>• Predictive analytics needed</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Key Terminology */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            3. Key Terminology
          </h2>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-3">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Dataset</h4>
                  <p className="text-xs text-muted-foreground">Collection of data used for training and testing ML models</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Features</h4>
                  <p className="text-xs text-muted-foreground">Input variables used to make predictions (X)</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Labels</h4>
                  <p className="text-xs text-muted-foreground">Output variables we want to predict (y)</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Samples</h4>
                  <p className="text-xs text-muted-foreground">Individual data points in the dataset</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-3">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Training Data</h4>
                  <p className="text-xs text-muted-foreground">Data used to train the model</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Test Data</h4>
                  <p className="text-xs text-muted-foreground">Data used to evaluate final model</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Validation Data</h4>
                  <p className="text-xs text-muted-foreground">Data used to tune hyperparameters</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Model</h4>
                  <p className="text-xs text-muted-foreground">Mathematical representation of learned patterns</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 col-span-2">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Prediction</h4>
                  <p className="text-xs text-muted-foreground">Model's output for new inputs</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Parameters</h4>
                  <p className="text-xs text-muted-foreground">Internal model variables learned from training data</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Hyperparameters</h4>
                  <p className="text-xs text-muted-foreground">Configuration settings set before training (e.g., learning rate)</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Types of Machine Learning */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            4. Types of Machine Learning
          </h2>

          <div className="space-y-6">
            {/* Supervised Learning */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3">Supervised Learning</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Classification</h4>
                  <p className="text-xs text-muted-foreground">Predicting discrete categories</p>
                  <div className="bg-muted p-2 rounded mt-2">
                    <p className="text-xs text-muted-foreground">Examples:</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Spam detection</li>
                      <li>Image recognition</li>
                      <li>Disease diagnosis</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Regression</h4>
                  <p className="text-xs text-muted-foreground">Predicting continuous values</p>
                  <div className="bg-muted p-2 rounded mt-2">
                    <p className="text-xs text-muted-foreground">Examples:</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>House price prediction</li>
                      <li>Stock forecasting</li>
                      <li>Temperature prediction</li>
                    </ul>
                  </div>
                </div>
              </div>
              <div className="mt-3">
                <h4 className="font-semibold text-foreground text-sm">Common Algorithms</h4>
                <div className="flex flex-wrap gap-2 mt-1">
                  <span className="bg-muted px-2 py-1 rounded text-xs">Linear Regression</span>
                  <span className="bg-muted px-2 py-1 rounded text-xs">Logistic Regression</span>
                  <span className="bg-muted px-2 py-1 rounded text-xs">Decision Trees</span>
                  <span className="bg-muted px-2 py-1 rounded text-xs">Random Forest</span>
                  <span className="bg-muted px-2 py-1 rounded text-xs">SVM</span>
                  <span className="bg-muted px-2 py-1 rounded text-xs">KNN</span>
                  <span className="bg-muted px-2 py-1 rounded text-xs">Naive Bayes</span>
                </div>
              </div>
            </div>

            {/* Unsupervised Learning */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3">Unsupervised Learning</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Clustering</h4>
                  <p className="text-xs text-muted-foreground">Group similar data points</p>
                  <div className="bg-muted p-2 rounded mt-1">
                    <p className="text-xs text-muted-foreground">Examples:</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Customer segmentation</li>
                      <li>Anomaly detection</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Association Rules</h4>
                  <p className="text-xs text-muted-foreground">Find relationships in data</p>
                  <div className="bg-muted p-2 rounded mt-1">
                    <p className="text-xs text-muted-foreground">Examples:</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Market basket analysis</li>
                      <li>Recommendation systems</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Dimensionality Reduction</h4>
                  <p className="text-xs text-muted-foreground">Reduce feature space</p>
                  <div className="bg-muted p-2 rounded mt-1">
                    <p className="text-xs text-muted-foreground">Examples:</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>PCA</li>
                      <li>t-SNE</li>
                      <li>Feature visualization</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Reinforcement Learning */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-3">Reinforcement Learning</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="space-y-2">
                    <div>
                      <h4 className="font-semibold text-foreground text-sm">Agent</h4>
                      <p className="text-xs text-muted-foreground">The learner/decision maker</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground text-sm">Environment</h4>
                      <p className="text-xs text-muted-foreground">Everything the agent interacts with</p>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="space-y-2">
                    <div>
                      <h4 className="font-semibold text-foreground text-sm">Reward</h4>
                      <p className="text-xs text-muted-foreground">Feedback signal to guide learning</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground text-sm">Policy</h4>
                      <p className="text-xs text-muted-foreground">Strategy for selecting actions</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-3 p-3 bg-muted rounded">
                <p className="text-xs text-muted-foreground">Examples: Game playing (AlphaGo), Robotics, Autonomous driving</p>
              </div>
            </div>

            {/* Semi-Supervised & Self-Supervised */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-card border border-border rounded-lg p-4">
                <h3 className="font-semibold text-foreground mb-2">Semi-Supervised Learning</h3>
                <p className="text-sm text-muted-foreground">
                  Combines small amount of labeled data with large amount of unlabeled data.
                  <br/>
                  <span className="text-xs">Example: Image classification with few labeled images</span>
                </p>
              </div>
              <div className="bg-card border border-border rounded-lg p-4">
                <h3 className="font-semibold text-foreground mb-2">Self-Supervised Learning</h3>
                <p className="text-sm text-muted-foreground">
                  Learns from data without explicit labels by creating proxy tasks.
                  <br/>
                  <span className="text-xs">Example: Predicting masked words in text (BERT)</span>
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Machine Learning Workflow */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" />
            5. Machine Learning Workflow
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">1</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Collection</h4>
                    <p className="text-xs text-muted-foreground">Gather relevant data from various sources</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">2</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Data Cleaning</h4>
                    <p className="text-xs text-muted-foreground">Handle missing values, outliers, and inconsistencies</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">3</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Feature Engineering</h4>
                    <p className="text-xs text-muted-foreground">Create and transform features for better performance</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">4</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Train-Test Split</h4>
                    <p className="text-xs text-muted-foreground">Separate data for training and evaluation</p>
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
                    <h4 className="font-semibold text-foreground text-sm">Model Evaluation</h4>
                    <p className="text-xs text-muted-foreground">Assess performance on validation data</p>
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
                    <h4 className="font-semibold text-foreground text-sm">Deployment & Monitoring</h4>
                    <p className="text-xs text-muted-foreground">Deploy to production and track performance</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Data Fundamentals */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            6. Data Fundamentals
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Types</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Structured Data</p>
                  <p className="text-xs text-muted-foreground">Organized in tables (e.g., CSV, SQL)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Unstructured Data</p>
                  <p className="text-xs text-muted-foreground">No predefined format (e.g., text, images)</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Categories</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Numerical</p>
                  <p className="text-xs text-muted-foreground">Continuous (age, price) or discrete (count)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Categorical</p>
                  <p className="text-xs text-muted-foreground">Nominal (colors) or ordinal (ratings)</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Formats</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Time Series</p>
                  <p className="text-xs text-muted-foreground">Data over time (stock prices)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Image & Audio</p>
                  <p className="text-xs text-muted-foreground">Pixel data or waveform data</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Text</p>
                  <p className="text-xs text-muted-foreground">Natural language data</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Common ML Algorithms Overview */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Code2 className="h-5 w-5 text-primary" />
            7. Common ML Algorithms Overview
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Regression</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Linear Regression</p>
                  <p className="text-xs text-muted-foreground">Simple linear relationships</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Polynomial Regression</p>
                  <p className="text-xs text-muted-foreground">Non-linear relationships</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Classification</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Logistic Regression</p>
                  <p className="text-xs text-muted-foreground">Probability-based classification</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Decision Trees</p>
                  <p className="text-xs text-muted-foreground">Tree-based decision making</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Random Forest</p>
                  <p className="text-xs text-muted-foreground">Ensemble of decision trees</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">SVM</p>
                  <p className="text-xs text-muted-foreground">Margin-based classification</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">KNN</p>
                  <p className="text-xs text-muted-foreground">Nearest neighbor-based</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Naive Bayes</p>
                  <p className="text-xs text-muted-foreground">Probability-based classification</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 md:col-span-2">
              <h4 className="font-semibold text-foreground mb-2">Clustering</h4>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <p className="font-medium text-foreground text-sm">K-Means</p>
                  <p className="text-xs text-muted-foreground">Centroid-based clustering</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Hierarchical</p>
                  <p className="text-xs text-muted-foreground">Tree-based clustering</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">DBSCAN</p>
                  <p className="text-xs text-muted-foreground">Density-based clustering</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Model Evaluation Basics */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <PieChart className="h-5 w-5 text-primary" />
            8. Model Evaluation Basics
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Classification Metrics</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Accuracy</p>
                  <p className="text-xs text-muted-foreground">Correct predictions / Total predictions</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Precision</p>
                  <p className="text-xs text-muted-foreground">True positives / (True + False positives)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Recall</p>
                  <p className="text-xs text-muted-foreground">True positives / (True + False negatives)</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">F1 Score</p>
                  <p className="text-xs text-muted-foreground">Harmonic mean of precision and recall</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Confusion Matrix</p>
                  <p className="text-xs text-muted-foreground">Visualizing TP, TN, FP, FN counts</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Regression Metrics</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">MAE</p>
                  <p className="text-xs text-muted-foreground">Mean Absolute Error</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">MSE</p>
                  <p className="text-xs text-muted-foreground">Mean Squared Error</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">RMSE</p>
                  <p className="text-xs text-muted-foreground">Root Mean Squared Error</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">R² Score</p>
                  <p className="text-xs text-muted-foreground">Coefficient of determination</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Challenges in Machine Learning */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            9. Challenges in Machine Learning
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Overfitting</h4>
                  <p className="text-sm text-muted-foreground">
                    Model learns training data too well, capturing noise instead of patterns.
                    <br/>
                    <span className="text-xs">Solution: Cross-validation, regularization, more data</span>
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Underfitting</h4>
                  <p className="text-sm text-muted-foreground">
                    Model is too simple to capture the underlying patterns.
                    <br/>
                    <span className="text-xs">Solution: Use more complex model, add features</span>
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Bias & Variance</h4>
                  <p className="text-sm text-muted-foreground">
                    Tradeoff between model simplicity (bias) and complexity (variance).
                    <br/>
                    <span className="text-xs">Goal: Find the sweet spot</span>
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Data Leakage</h4>
                  <p className="text-sm text-muted-foreground">
                    Information from test set leaks into training.
                    <br/>
                    <span className="text-xs">Solution: Always split data before preprocessing</span>
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Class Imbalance</h4>
                  <p className="text-sm text-muted-foreground">
                    One class dominates the dataset.
                    <br/>
                    <span className="text-xs">Solution: Resampling, weighted loss functions</span>
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Missing Data</h4>
                  <p className="text-sm text-muted-foreground">
                    Missing values in the dataset.
                    <br/>
                    <span className="text-xs">Solution: Imputation, deletion, or prediction</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 10. ML Tools & Libraries */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Server className="h-5 w-5 text-primary" />
            10. ML Tools & Libraries
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Python Libraries</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">NumPy</p>
                  <p className="text-xs text-muted-foreground">Numerical computing</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Pandas</p>
                  <p className="text-xs text-muted-foreground">Data manipulation</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Matplotlib</p>
                  <p className="text-xs text-muted-foreground">Data visualization</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Seaborn</p>
                  <p className="text-xs text-muted-foreground">Statistical visualization</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Scikit-Learn</p>
                  <p className="text-xs text-muted-foreground">Machine learning algorithms</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Deep Learning Libraries</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">TensorFlow</p>
                  <p className="text-xs text-muted-foreground">Google's DL framework</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">PyTorch</p>
                  <p className="text-xs text-muted-foreground">Facebook's DL framework</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Keras</p>
                  <p className="text-xs text-muted-foreground">High-level API for TensorFlow</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Other Tools</h4>
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Jupyter</p>
                  <p className="text-xs text-muted-foreground">Interactive notebooks</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Git</p>
                  <p className="text-xs text-muted-foreground">Version control</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Docker</p>
                  <p className="text-xs text-muted-foreground">Containerization</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 11. Machine Learning Applications */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Globe className="h-5 w-5 text-primary" />
            11. Machine Learning Applications
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Recommendation Systems</p>
                  <p className="text-xs text-muted-foreground">Netflix, Amazon, Spotify recommendations</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Spam Detection</p>
                  <p className="text-xs text-muted-foreground">Email spam filtering</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Fraud Detection</p>
                  <p className="text-xs text-muted-foreground">Banking and financial fraud</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Medical Diagnosis</p>
                  <p className="text-xs text-muted-foreground">Disease detection from medical images</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-2">
                <div>
                  <p className="font-medium text-foreground text-sm">Self-Driving Cars</p>
                  <p className="text-xs text-muted-foreground">Autonomous vehicles</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Computer Vision</p>
                  <p className="text-xs text-muted-foreground">Image recognition, object detection</p>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">Natural Language Processing</p>
                  <p className="text-xs text-muted-foreground">Chatbots, translation, sentiment analysis</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 12. Future of Machine Learning */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            12. Future of Machine Learning
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Emerging Trends</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><span className="font-medium text-foreground">Generative AI:</span> Creating content (text, images, video)</li>
                <li><span className="font-medium text-foreground">LLMs:</span> Large Language Models (GPT, Claude)</li>
                <li><span className="font-medium text-foreground">AI Agents:</span> Autonomous decision-making systems</li>
                <li><span className="font-medium text-foreground">Multimodal AI:</span> Combining text, image, audio</li>
                <li><span className="font-medium text-foreground">Edge AI:</span> ML on edge devices</li>
              </ul>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Impact on Industries</h4>
                  <ul className="text-sm text-muted-foreground">
                    <li>• Healthcare: Personalized medicine</li>
                    <li>• Education: Adaptive learning</li>
                    <li>• Finance: AI-powered trading</li>
                    <li>• Manufacturing: Smart factories</li>
                    <li>• Transportation: Autonomous systems</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 13. Prerequisites for ML */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            13. Prerequisites for Machine Learning
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Core Skills</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Python Programming</li>
                <li>• Statistics</li>
                <li>• Probability</li>
                <li>• Linear Algebra</li>
                <li>• Calculus (Basics)</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Technical Skills</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Data Structures</li>
                <li>• Algorithms</li>
                <li>• SQL</li>
                <li>• Git</li>
                <li>• Linux/Command Line</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Soft Skills</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Problem-solving</li>
                <li>• Critical thinking</li>
                <li>• Communication</li>
                <li>• Business acumen</li>
                <li>• Curiosity</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 14. Mini Projects */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            14. Mini Projects
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4 hover:border-primary transition-colors">
              <h4 className="font-semibold text-foreground mb-2">House Price Prediction</h4>
              <p className="text-sm text-muted-foreground">
                Use regression to predict house prices based on features like size, location, and number of rooms.
                <br/>
                <span className="text-xs text-primary">Skills: Regression, Feature Engineering</span>
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 hover:border-primary transition-colors">
              <h4 className="font-semibold text-foreground mb-2">Student Score Prediction</h4>
              <p className="text-sm text-muted-foreground">
                Predict student exam scores based on study hours, previous grades, and attendance.
                <br/>
                <span className="text-xs text-primary">Skills: Regression, Data Analysis</span>
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 hover:border-primary transition-colors">
              <h4 className="font-semibold text-foreground mb-2">Spam Email Classifier</h4>
              <p className="text-sm text-muted-foreground">
                Build a classifier to detect spam emails using text features and NLP techniques.
                <br/>
                <span className="text-xs text-primary">Skills: Classification, NLP</span>
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 hover:border-primary transition-colors">
              <h4 className="font-semibold text-foreground mb-2">Movie Recommendation System</h4>
              <p className="text-sm text-muted-foreground">
                Create a movie recommendation engine using collaborative filtering or content-based approaches.
                <br/>
                <span className="text-xs text-primary">Skills: Recommendation Systems, Collaborative Filtering</span>
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 hover:border-primary transition-colors">
              <h4 className="font-semibold text-foreground mb-2">Customer Segmentation</h4>
              <p className="text-sm text-muted-foreground">
                Use clustering to segment customers based on purchasing behavior and demographics.
                <br/>
                <span className="text-xs text-primary">Skills: Clustering, Marketing Analytics</span>
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-4 hover:border-primary transition-colors">
              <h4 className="font-semibold text-foreground mb-2">Image Classifier</h4>
              <p className="text-sm text-muted-foreground">
                Build a classifier to identify objects or animals in images using CNN.
                <br/>
                <span className="text-xs text-primary">Skills: Deep Learning, Computer Vision</span>
              </p>
            </div>
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Code Examples
          </h2>
          <p className="text-muted-foreground mb-4">
            See Machine Learning in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Machine Learning Introduction Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}