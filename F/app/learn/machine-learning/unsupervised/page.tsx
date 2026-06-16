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
  Grid
} from "lucide-react";

export default function UnsupervisedLearningPage() {
  const result = getSubtopicBySlug("machine-learning", "unsupervised");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-kmeans",
      label: "K-Means Clustering",
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
y_pred = kmeans.fit_predict(X_scaled)

# Get cluster centers
centers = kmeans.cluster_centers_

print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Cluster centers shape: {centers.shape}")

# Find optimal K using Elbow Method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

print("\\nInertia for different K values:")
for k, inertia in enumerate(inertias, 1):
    print(f"K={k}: {inertia:.2f}")`,
    },
    {
      language: "python-hierarchical",
      label: "Hierarchical Clustering",
      code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Generate data
X, _ = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=0.60,
    random_state=42
)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Agglomerative Clustering
model = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)
y_pred = model.fit_predict(X_scaled)

print(f"Number of clusters: {model.n_clusters}")
print(f"Labels: {np.unique(y_pred)}")
print(f"Cluster sizes:")
for i in range(model.n_clusters):
    print(f"Cluster {i}: {np.sum(y_pred == i)} samples")

# Compute linkage matrix for dendrogram
linkage_matrix = linkage(X_scaled, method='ward')

print("\\nLinkage matrix shape:", linkage_matrix.shape)`,
    },
    {
      language: "python-dbscan",
      label: "DBSCAN",
      code: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate moons dataset (non-linear clusters)
X, _ = make_moons(
    n_samples=200,
    noise=0.05,
    random_state=42
)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)
y_pred = dbscan.fit_predict(X_scaled)

# Get number of clusters (excluding noise points labeled -1)
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = list(y_pred).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Unique labels: {np.unique(y_pred)}")

# Calculate silhouette score (if more than 1 cluster)
if n_clusters > 1:
    score = silhouette_score(X_scaled, y_pred)
    print(f"Silhouette Score: {score:.3f}")`,
    },
    {
      language: "python-pca",
      label: "PCA - Dimensionality Reduction",
      code: `import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X = digits.data
y = digits.target

print(f"Original data shape: {X.shape}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced data shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")

# Show principal components
print(f"\\nPrincipal components shape: {pca.components_.shape}")
print(f"First component explains {pca.explained_variance_ratio_[0]:.3f} of variance")
print(f"Second component explains {pca.explained_variance_ratio_[1]:.3f} of variance")`,
    },
    {
      language: "python-tsne",
      label: "t-SNE - Visualization",
      code: `import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X = digits.data
y = digits.target

print(f"Original data shape: {X.shape}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    n_iter=1000
)
X_tsne = tsne.fit_transform(X_scaled)

print(f"Reduced data shape: {X_tsne.shape}")
print(f"KL divergence: {tsne.kl_divergence_:.3f}")

# Show unique digits in dataset
print(f"Unique digits: {np.unique(y)}")
print(f"Number of samples per digit:")
for digit in range(10):
    print(f"Digit {digit}: {np.sum(y == digit)} samples")`,
    },
    {
      language: "python-rule",
      label: "Association Rules (Apriori)",
      code: `from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np

# Create sample transaction data
dataset = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['bread', 'eggs'],
    ['milk', 'eggs'],
    ['milk', 'bread', 'eggs', 'butter'],
    ['bread', 'butter'],
    ['milk', 'butter'],
    ['milk', 'bread', 'butter']
]

# One-hot encode transactions
# Create a list of all unique items
items = set()
for transaction in dataset:
    items.update(transaction)
items = sorted(list(items))

# Create one-hot encoded DataFrame
one_hot = []
for transaction in dataset:
    row = [1 if item in transaction else 0 for item in items]
    one_hot.append(row)

df = pd.DataFrame(one_hot, columns=items)

print("Transaction Data (One-hot encoded):")
print(df)

# Find frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

print("\\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Unsupervised Learning?",
      options: [
        "Learning with labeled data",
        "Learning from unlabeled data to find patterns",
        "Learning through rewards and punishments",
        "Learning from structured data only"
      ],
      correctAnswer: 1,
      explanation: "Unsupervised learning works with unlabeled data to discover hidden patterns and structures without predefined outputs.",
    },
    {
      id: 2,
      question: "What is the main goal of clustering?",
      options: [
        "To predict future values",
        "To group similar data points together",
        "To reduce the dataset size",
        "To classify data into categories"
      ],
      correctAnswer: 1,
      explanation: "Clustering aims to group similar data points based on their features, discovering natural groupings in the data.",
    },
    {
      id: 3,
      question: "What is the difference between K-Means and Hierarchical Clustering?",
      options: [
        "They are the same algorithm",
        "K-Means requires specifying k, Hierarchical creates a tree of clusters",
        "K-Means is for classification, Hierarchical is for regression",
        "Hierarchical requires labeled data"
      ],
      correctAnswer: 1,
      explanation: "K-Means requires the number of clusters (k) to be specified, while Hierarchical clustering creates a dendrogram tree without pre-specifying k.",
    },
    {
      id: 4,
      question: "What is the Elbow Method used for?",
      options: [
        "To evaluate model accuracy",
        "To find the optimal number of clusters in K-Means",
        "To reduce dimensionality",
        "To detect outliers"
      ],
      correctAnswer: 1,
      explanation: "The Elbow Method plots inertia vs. number of clusters to find the 'elbow' point where adding more clusters yields diminishing returns.",
    },
    {
      id: 5,
      question: "What is PCA used for?",
      options: [
        "Clustering data",
        "Dimensionality reduction while preserving variance",
        "Classification of data",
        "Association rule mining"
      ],
      correctAnswer: 1,
      explanation: "PCA (Principal Component Analysis) reduces dimensionality by transforming data to new axes that maximize variance.",
    },
    {
      id: 6,
      question: "What is the main advantage of DBSCAN over K-Means?",
      options: [
        "It's faster",
        "It can find clusters of arbitrary shape and handle noise",
        "It requires fewer parameters",
        "It works with labeled data"
      ],
      correctAnswer: 1,
      explanation: "DBSCAN can identify clusters of any shape and automatically handles noise points, unlike K-Means which assumes spherical clusters.",
    },
    {
      id: 7,
      question: "What is the purpose of Association Rule Mining?",
      options: [
        "To cluster data points",
        "To discover relationships between items in transactions",
        "To reduce dimensionality",
        "To classify data"
      ],
      correctAnswer: 1,
      explanation: "Association Rule Mining discovers interesting relationships and patterns in large datasets, often used in market basket analysis.",
    },
    {
      id: 8,
      question: "What is the silhouette score used for?",
      options: [
        "To measure model accuracy",
        "To evaluate clustering quality",
        "To determine the number of features",
        "To test data normality"
      ],
      correctAnswer: 1,
      explanation: "Silhouette score measures how similar each point is to its own cluster compared to other clusters, ranging from -1 to 1.",
    },
    {
      id: 9,
      question: "What is the difference between PCA and t-SNE?",
      options: [
        "They are the same algorithm",
        "PCA is linear, t-SNE is non-linear for visualization",
        "t-SNE is faster",
        "PCA works only with categorical data"
      ],
      correctAnswer: 1,
      explanation: "PCA is a linear dimensionality reduction technique, while t-SNE is non-linear and specifically designed for visualizing high-dimensional data.",
    },
    {
      id: 10,
      question: "What is a dendrogram?",
      options: [
        "A type of neural network",
        "A tree diagram showing hierarchical clustering relationships",
        "A visualization of PCA results",
        "A type of database"
      ],
      correctAnswer: 1,
      explanation: "A dendrogram is a tree diagram that shows the hierarchical relationships between clusters, used in hierarchical clustering.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Unsupervised Machine Learning — Complete Guide
          </h1>
          <p className="text-muted-foreground text-lg">
            A comprehensive guide to understanding unsupervised learning, clustering, dimensionality reduction, and association rules
          </p>
        </div>

        {/* 1. What is Unsupervised Learning? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            1. What is Unsupervised Learning?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Unsupervised learning is a type of machine learning where a model is trained using <span className="font-semibold text-foreground">unlabeled data</span>, meaning there are no predefined outputs (y).
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-lg text-primary">X → ???</p>
                <p className="text-xs text-muted-foreground">Discover hidden patterns without labels</p>
              </div>
              <p className="text-muted-foreground mt-3">
                The goal is to discover hidden patterns, structures, or relationships in the data.
              </p>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Analogy</h4>
                  <p className="text-sm text-muted-foreground">
                    Think of unsupervised learning like exploring without a map:
                  </p>
                  <ul className="text-sm text-muted-foreground list-disc list-inside mt-2">
                    <li><span className="font-medium text-foreground">Data</span>: Uncharted territory</li>
                    <li><span className="font-medium text-foreground">Model</span>: Explorer discovering patterns</li>
                    <li><span className="font-medium text-foreground">Discovery</span>: Finding natural groupings or structures</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Types of Unsupervised Learning */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            2. Types of Unsupervised Learning
          </h2>

          <div className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-card border border-border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Network className="h-4 w-4 text-primary" />
                  <h4 className="font-semibold text-foreground">Clustering</h4>
                </div>
                <p className="text-xs text-muted-foreground">
                  Group similar data points together
                </p>
                <div className="mt-2 bg-muted p-2 rounded">
                  <p className="text-xs font-medium text-foreground">Examples:</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Customer segmentation</li>
                    <li>Image segmentation</li>
                    <li>Anomaly detection</li>
                  </ul>
                </div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Grid className="h-4 w-4 text-primary" />
                  <h4 className="font-semibold text-foreground">Association Rules</h4>
                </div>
                <p className="text-xs text-muted-foreground">
                  Discover relationships between items
                </p>
                <div className="mt-2 bg-muted p-2 rounded">
                  <p className="text-xs font-medium text-foreground">Examples:</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Market basket analysis</li>
                    <li>Recommendation systems</li>
                    <li>Pattern discovery</li>
                  </ul>
                </div>
              </div>

              <div className="bg-card border border-border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <ScatterChart className="h-4 w-4 text-primary" />
                  <h4 className="font-semibold text-foreground">Dimensionality Reduction</h4>
                </div>
                <p className="text-xs text-muted-foreground">
                  Reduce number of features
                </p>
                <div className="mt-2 bg-muted p-2 rounded">
                  <p className="text-xs font-medium text-foreground">Examples:</p>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Data visualization</li>
                    <li>Feature extraction</li>
                    <li>Noise reduction</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Clustering Algorithms */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Network className="h-5 w-5 text-primary" />
            3. Clustering Algorithms
          </h2>

          <div className="space-y-4">
            {/* K-Means */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">K-Means Clustering</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">How it works</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Initialize K centroids randomly</li>
                    <li>Assign points to nearest centroid</li>
                    <li>Update centroids</li>
                    <li>Repeat until convergence</li>
                  </ul>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                    <p className="font-semibold text-foreground text-xs">Pros</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Simple and fast</li>
                      <li>Scalable</li>
                    </ul>
                  </div>
                  <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                    <p className="font-semibold text-foreground text-xs">Cons</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Requires specifying K</li>
                      <li>Sensitive to initial centroids</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Hierarchical Clustering */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">Hierarchical Clustering</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">How it works</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Start with each point as its own cluster</li>
                    <li>Merge closest clusters</li>
                    <li>Repeat until all points in one cluster</li>
                    <li>Result: Dendrogram tree</li>
                  </ul>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                    <p className="font-semibold text-foreground text-xs">Pros</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>No need to specify K</li>
                      <li>Hierarchical structure</li>
                    </ul>
                  </div>
                  <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                    <p className="font-semibold text-foreground text-xs">Cons</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Computationally expensive</li>
                      <li>Sensitive to noise</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* DBSCAN */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">DBSCAN</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">How it works</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Groups points based on density</li>
                    <li>Uses eps (neighborhood radius)</li>
                    <li>Uses min_samples (minimum points)</li>
                    <li>Identifies noise points</li>
                  </ul>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                    <p className="font-semibold text-foreground text-xs">Pros</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Can find arbitrary shapes</li>
                      <li>Handles noise well</li>
                    </ul>
                  </div>
                  <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                    <p className="font-semibold text-foreground text-xs">Cons</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Requires tuning parameters</li>
                      <li>Sensitive to density variation</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Common Algorithms Summary */}
            <div className="bg-muted/50 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Common Clustering Algorithms</h4>
              <div className="grid md:grid-cols-3 gap-2 text-xs">
                <div className="bg-card p-2 rounded">
                  <p className="font-medium text-foreground">K-Means</p>
                  <p className="text-muted-foreground">Partition-based, centroid</p>
                </div>
                <div className="bg-card p-2 rounded">
                  <p className="font-medium text-foreground">Hierarchical</p>
                  <p className="text-muted-foreground">Agglomerative/Divisive</p>
                </div>
                <div className="bg-card p-2 rounded">
                  <p className="font-medium text-foreground">DBSCAN</p>
                  <p className="text-muted-foreground">Density-based</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Dimensionality Reduction */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <ScatterChart className="h-5 w-5 text-primary" />
            4. Dimensionality Reduction
          </h2>

          <div className="space-y-4">
            {/* PCA */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">PCA (Principal Component Analysis)</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">How it works</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Finds directions of maximum variance</li>
                    <li>Projects data onto principal components</li>
                    <li>Preserves most important information</li>
                  </ul>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                    <p className="font-semibold text-foreground text-xs">Pros</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Linear and fast</li>
                      <li>Preserves variance</li>
                    </ul>
                  </div>
                  <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                    <p className="font-semibold text-foreground text-xs">Cons</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Assumes linear relationships</li>
                      <li>Less interpretable</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* t-SNE */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">t-SNE (t-Distributed Stochastic Neighbor Embedding)</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">How it works</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Preserves local structure</li>
                    <li>Non-linear dimensionality reduction</li>
                    <li>Great for visualization</li>
                  </ul>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-green-50 border border-green-200 rounded p-2 dark:bg-green-950 dark:border-green-800">
                    <p className="font-semibold text-foreground text-xs">Pros</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Excellent for visualization</li>
                      <li>Handles non-linear patterns</li>
                    </ul>
                  </div>
                  <div className="bg-destructive/10 border border-destructive/20 rounded p-2">
                    <p className="font-semibold text-foreground text-xs">Cons</p>
                    <ul className="text-xs text-muted-foreground list-disc list-inside">
                      <li>Computationally intensive</li>
                      <li>Stochastic results</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Association Rules */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Grid className="h-5 w-5 text-primary" />
            5. Association Rules
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold text-foreground text-lg mb-2">Apriori Algorithm</h3>
                <p className="text-sm text-muted-foreground">
                  Finds frequent itemsets and association rules in transactional data.
                </p>
                <div className="mt-3">
                  <h4 className="font-semibold text-foreground text-sm">Key Metrics</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium text-foreground">Support</span>: How frequent an itemset is</li>
                    <li><span className="font-medium text-foreground">Confidence</span>: Conditional probability</li>
                    <li><span className="font-medium text-foreground">Lift</span>: Strength of association</li>
                  </ul>
                </div>
              </div>
              <div className="bg-muted p-3 rounded">
                <p className="text-sm font-semibold text-foreground">Example Rule</p>
                <div className="mt-2 text-center">
                  <p className="text-sm text-muted-foreground">
                    <span className="text-primary font-medium">Milk</span> → <span className="text-primary font-medium">Bread</span>
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Support: 0.4 | Confidence: 0.8 | Lift: 1.2
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Customers who buy milk are 1.2x more likely to buy bread
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Key Concepts */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            6. Key Concepts in Unsupervised Learning
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Distance Metrics</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Euclidean</span>: Straight-line distance</li>
                <li><span className="font-medium text-foreground">Manhattan</span>: City-block distance</li>
                <li><span className="font-medium text-foreground">Cosine</span>: Similarity for text</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Clustering Evaluation</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Silhouette Score</span>: Cluster quality</li>
                <li><span className="font-medium text-foreground">Elbow Method</span>: Optimal K</li>
                <li><span className="font-medium text-foreground">Dunn Index</span>: Compactness</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Dimensionality Metrics</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li><span className="font-medium text-foreground">Explained Variance</span>: PCA metric</li>
                <li><span className="font-medium text-foreground">KL Divergence</span>: t-SNE metric</li>
                <li><span className="font-medium text-foreground">Stress</span>: MDS metric</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 7. Challenges and Solutions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            7. Challenges and Solutions
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Choosing Number of Clusters</h4>
                  <p className="text-sm text-muted-foreground">
                    Use Elbow Method, Silhouette Analysis, or Gap Statistic.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">High Dimensionality</h4>
                  <p className="text-sm text-muted-foreground">
                    Apply PCA first, then cluster. Use t-SNE for visualization.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Noise and Outliers</h4>
                  <p className="text-sm text-muted-foreground">
                    Use DBSCAN for robust clustering with noise handling.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Interpretability</h4>
                  <p className="text-sm text-muted-foreground">
                    Use explainable techniques like decision trees on clusters.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Applications */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            8. Applications
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Customer Segmentation</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Group customers by behavior</li>
                <li>Targeted marketing campaigns</li>
                <li>Personalized recommendations</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Anomaly Detection</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Fraud detection</li>
                <li>Network intrusion</li>
                <li>Equipment failure</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Data Visualization</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>High-dimensional visualization</li>
                <li>Exploratory data analysis</li>
                <li>Pattern discovery</li>
              </ul>
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
                <h4 className="font-semibold text-foreground mb-1">Standardize data</h4>
                <p className="text-sm text-muted-foreground">Always scale features for distance-based algorithms</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Try multiple algorithms</h4>
                <p className="text-sm text-muted-foreground">Different algorithms reveal different patterns</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Validate results</h4>
                <p className="text-sm text-muted-foreground">Use multiple evaluation metrics and domain knowledge</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Visualize when possible</h4>
                <p className="text-sm text-muted-foreground">Use t-SNE, PCA, or dendrograms for insights</p>
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
            See unsupervised learning algorithms in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Unsupervised Learning Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}