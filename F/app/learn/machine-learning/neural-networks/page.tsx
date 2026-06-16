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
  Cog
} from "lucide-react";

export default function NeuralNetworksPage() {
  const result = getSubtopicBySlug("machine-learning", "neural-networks");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-mlp",
      label: "MLP - TensorFlow/Keras",
      code: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate synthetic data
X = np.random.random((1000, 20))
y = np.random.randint(0, 2, (1000, 1))

# Build MLP model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()

# Train model
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")`,
    },
    {
      language: "python-cnn",
      label: "CNN - TensorFlow/Keras",
      code: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate synthetic image data
X = np.random.random((1000, 28, 28, 1))
y = np.random.randint(0, 10, (1000,))

# Build CNN model
model = keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()

# Train model
history = model.fit(
    X, y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")`,
    },
    {
      language: "python-rnn",
      label: "RNN/LSTM - TensorFlow/Keras",
      code: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate synthetic sequence data
timesteps = 50
features = 10
X = np.random.random((1000, timesteps, features))
y = np.random.randint(0, 2, (1000, 1))

# Build LSTM model
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()

# Train model
history = model.fit(
    X, y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")`,
    },
    {
      language: "python-pytorch",
      label: "PyTorch - Basic NN",
      code: `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Generate synthetic data
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000, 1)).float()

# Create model, loss, optimizer
model = NeuralNetwork(20, 64, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
batch_size = 32

for epoch in range(epochs):
    # Shuffle data
    indices = torch.randperm(X.size(0))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Mini-batch training
    for i in range(0, len(X), batch_size):
        batch_X = X_shuffled[i:i+batch_size]
        batch_y = y_shuffled[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training complete!")`,
    },
    {
      language: "python-transfer",
      label: "Transfer Learning",
      code: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import numpy as np

# Load pre-trained VGG16 model (without top layers)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display architecture
model.summary()

# Generate synthetic data (for demonstration)
X = np.random.random((100, 224, 224, 3))
y = keras.utils.to_categorical(np.random.randint(0, 10, (100,)), 10)

# Train model
history = model.fit(
    X, y,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("Transfer learning complete!")
print(f"Final accuracy: {history.history['accuracy'][-1]:.3f}")`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is a neural network?",
      options: [
        "A network of computers",
        "A computing system inspired by biological neural networks",
        "A type of database",
        "A programming language"
      ],
      correctAnswer: 1,
      explanation: "Neural networks are computing systems inspired by biological neural networks that constitute animal brains, designed to recognize patterns.",
    },
    {
      id: 2,
      question: "What is a perceptron?",
      options: [
        "A type of neural network with multiple layers",
        "The simplest type of neural network with a single layer",
        "A type of activation function",
        "A training algorithm"
      ],
      correctAnswer: 1,
      explanation: "A perceptron is the simplest neural network architecture, consisting of a single layer of neurons.",
    },
    {
      id: 3,
      question: "What is the role of activation functions?",
      options: [
        "To initialize weights",
        "To introduce non-linearity into the network",
        "To reduce overfitting",
        "To speed up training"
      ],
      correctAnswer: 1,
      explanation: "Activation functions introduce non-linearity, allowing neural networks to learn complex patterns.",
    },
    {
      id: 4,
      question: "What is backpropagation?",
      options: [
        "A way to initialize weights",
        "An algorithm to train neural networks by propagating errors backward",
        "A type of activation function",
        "A regularization technique"
      ],
      correctAnswer: 1,
      explanation: "Backpropagation is the core training algorithm that computes gradients and updates weights by propagating errors backward through the network.",
    },
    {
      id: 5,
      question: "What is the difference between CNN and RNN?",
      options: [
        "CNN is for images, RNN is for sequences",
        "CNN is faster, RNN is slower",
        "CNN has more layers, RNN has fewer layers",
        "They are the same thing"
      ],
      correctAnswer: 0,
      explanation: "CNNs (Convolutional Neural Networks) are designed for spatial data like images, while RNNs (Recurrent Neural Networks) are designed for sequential data like text and time series.",
    },
    {
      id: 6,
      question: "What is a dropout layer used for?",
      options: [
        "To increase model complexity",
        "To prevent overfitting by randomly dropping neurons",
        "To speed up training",
        "To improve accuracy"
      ],
      correctAnswer: 1,
      explanation: "Dropout randomly drops neurons during training to prevent overfitting and improve generalization.",
    },
    {
      id: 7,
      question: "What is the vanishing gradient problem?",
      options: [
        "Gradients become very large during training",
        "Gradients become very small and slow down learning in deep networks",
        "Gradients disappear completely",
        "Gradients oscillate wildly"
      ],
      correctAnswer: 1,
      explanation: "The vanishing gradient problem occurs when gradients become extremely small, making deep networks difficult to train.",
    },
    {
      id: 8,
      question: "What is transfer learning?",
      options: [
        "Transferring data between models",
        "Using pre-trained models for new tasks",
        "Transferring weights between layers",
        "A type of activation function"
      ],
      correctAnswer: 1,
      explanation: "Transfer learning uses knowledge from a pre-trained model on a large dataset and applies it to a new, related task.",
    },
    {
      id: 9,
      question: "What is the purpose of a loss function?",
      options: [
        "To measure model accuracy",
        "To measure the error between predictions and true values",
        "To initialize weights",
        "To reduce overfitting"
      ],
      correctAnswer: 1,
      explanation: "A loss function quantifies how far the model's predictions are from the true values, guiding the training process.",
    },
    {
      id: 10,
      question: "What is an optimizer in neural networks?",
      options: [
        "A function that evaluates model performance",
        "An algorithm that updates weights to minimize loss",
        "A type of activation function",
        "A regularization technique"
      ],
      correctAnswer: 1,
      explanation: "An optimizer is an algorithm that updates network weights to minimize the loss function, such as SGD or Adam.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
      

        {/* 1. What is a Neural Network? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            1. What is a Neural Network?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                A neural network is a computing system inspired by biological neural networks that constitute animal brains. It consists of interconnected nodes (neurons) that process information.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <div className="flex justify-center items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-xs">Input</div>
                  <span className="text-muted-foreground">→</span>
                  <div className="flex gap-1">
                    <div className="w-8 h-8 rounded-full bg-primary/30 flex items-center justify-center text-xs">H</div>
                    <div className="w-8 h-8 rounded-full bg-primary/30 flex items-center justify-center text-xs">H</div>
                    <div className="w-8 h-8 rounded-full bg-primary/30 flex items-center justify-center text-xs">H</div>
                  </div>
                  <span className="text-muted-foreground">→</span>
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-xs">Output</div>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Input → Hidden Layers → Output</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Analogy</h4>
                  <p className="text-sm text-muted-foreground">
                    Think of a neural network like a brain:
                  </p>
                  <ul className="text-sm text-muted-foreground list-disc list-inside mt-2">
                    <li><span className="font-medium text-foreground">Neurons</span>: Processing units</li>
                    <li><span className="font-medium text-foreground">Synapses</span>: Weighted connections</li>
                    <li><span className="font-medium text-foreground">Learning</span>: Strengthening/weakening connections</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. History and Evolution */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Clock className="h-5 w-5 text-primary" />
            2. History and Evolution of Neural Networks
          </h2>

          <div className="grid md:grid-cols-4 gap-3">
            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-2">
                <span className="text-xs font-bold text-primary">1958</span>
              </div>
              <h4 className="font-semibold text-foreground text-xs">Perceptron</h4>
              <p className="text-xs text-muted-foreground">First neural network model</p>
            </div>

            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-2">
                <span className="text-xs font-bold text-primary">1986</span>
              </div>
              <h4 className="font-semibold text-foreground text-xs">Backpropagation</h4>
              <p className="text-xs text-muted-foreground">Training algorithm for multi-layer networks</p>
            </div>

            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-2">
                <span className="text-xs font-bold text-primary">2012</span>
              </div>
              <h4 className="font-semibold text-foreground text-xs">Deep Learning</h4>
              <p className="text-xs text-muted-foreground">ImageNet breakthrough with CNNs</p>
            </div>

            <div className="bg-card border border-border rounded-lg p-3 text-center">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-2">
                <span className="text-xs font-bold text-primary">2020+</span>
              </div>
              <h4 className="font-semibold text-foreground text-xs">Generative AI</h4>
              <p className="text-xs text-muted-foreground">GPT, Diffusion models, Transformers</p>
            </div>
          </div>
        </section>

        {/* 3. Basic Architecture */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            3. Basic Architecture
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Components</h3>
              <div className="space-y-3">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Input Layer</h4>
                  <p className="text-xs text-muted-foreground">Receives input data (features)</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Hidden Layers</h4>
                  <p className="text-xs text-muted-foreground">Process information through weighted connections</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Output Layer</h4>
                  <p className="text-xs text-muted-foreground">Produces final prediction</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Weights & Biases</h4>
                  <p className="text-xs text-muted-foreground">Parameters learned during training</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Neuron Calculation</h3>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-sm text-center">z = Σ(wᵢ × xᵢ) + b</p>
                <p className="font-mono text-sm text-center mt-2">a = f(z)</p>
                <div className="mt-2 text-xs text-muted-foreground">
                  <p>Where:</p>
                  <ul className="list-disc list-inside">
                    <li>wᵢ = weights</li>
                    <li>xᵢ = inputs</li>
                    <li>b = bias</li>
                    <li>f = activation function</li>
                    <li>a = output (activation)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Activation Functions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            4. Activation Functions
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-3">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Sigmoid</h4>
                  <p className="text-xs text-muted-foreground">σ(x) = 1 / (1 + e⁻ˣ)</p>
                  <p className="text-xs text-muted-foreground mt-1">Range: (0, 1) - Good for binary classification</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Tanh</h4>
                  <p className="text-xs text-muted-foreground">tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)</p>
                  <p className="text-xs text-muted-foreground mt-1">Range: (-1, 1) - Zero-centered</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">ReLU</h4>
                  <p className="text-xs text-muted-foreground">f(x) = max(0, x)</p>
                  <p className="text-xs text-muted-foreground mt-1">Range: [0, ∞) - Most popular for hidden layers</p>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Leaky ReLU</h4>
                  <p className="text-xs text-muted-foreground">f(x) = max(αx, x) where α is small</p>
                  <p className="text-xs text-muted-foreground mt-1">Fixes dead neurons problem</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Softmax</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-sm text-center">σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Outputs probability distribution over multiple classes
                </p>
              </div>
              
              <h4 className="font-semibold text-foreground mt-3 mb-2">Activation Function Guide</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside space-y-1">
                <li><span className="font-medium text-foreground">Hidden Layers:</span> ReLU (default), Leaky ReLU, Tanh</li>
                <li><span className="font-medium text-foreground">Binary Classification:</span> Sigmoid</li>
                <li><span className="font-medium text-foreground">Multi-class Classification:</span> Softmax</li>
                <li><span className="font-medium text-foreground">Regression:</span> Linear (no activation)</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 5. Types of Neural Networks */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Network className="h-5 w-5 text-primary" />
            5. Types of Neural Networks
          </h2>

          <div className="space-y-4">
            {/* MLP */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">MLP (Multilayer Perceptron)</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">What it does</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Fully connected layers</li>
                    <li>Works with tabular data</li>
                    <li>Basic building block</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Use Cases</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Classification</li>
                    <li>Regression</li>
                    <li>Pattern recognition</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* CNN */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">CNN (Convolutional Neural Network)</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">What it does</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Uses convolutional layers</li>
                    <li>Detects spatial patterns</li>
                    <li>Translation invariant</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Use Cases</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Image classification</li>
                    <li>Object detection</li>
                    <li>Video analysis</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* RNN/LSTM */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">RNN / LSTM (Recurrent Neural Networks)</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">What it does</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Processes sequences</li>
                    <li>Has memory of previous inputs</li>
                    <li>LSTM handles long-term dependencies</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Use Cases</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Natural Language Processing</li>
                    <li>Time series prediction</li>
                    <li>Speech recognition</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Transformers */}
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground text-lg mb-2">Transformers</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-foreground text-sm">What it does</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Uses attention mechanism</li>
                    <li>Handles long-range dependencies</li>
                    <li>Parallel processing</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-foreground text-sm">Use Cases</h4>
                  <ul className="text-xs text-muted-foreground list-disc list-inside">
                    <li>Language models (GPT, BERT)</li>
                    <li>Machine translation</li>
                    <li>Text generation</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Training Process */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            6. Training Process
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="space-y-3">
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">1</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Forward Propagation</h4>
                    <p className="text-xs text-muted-foreground">Input passes through network to produce output</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">2</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Loss Calculation</h4>
                    <p className="text-xs text-muted-foreground">Compute error between prediction and target</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">3</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Backpropagation</h4>
                    <p className="text-xs text-muted-foreground">Calculate gradients of loss w.r.t. weights</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="bg-primary text-primary-foreground rounded-full w-5 h-5 flex items-center justify-center text-xs shrink-0">4</span>
                  <div>
                    <h4 className="font-semibold text-foreground text-sm">Weight Update</h4>
                    <p className="text-xs text-muted-foreground">Update weights using optimizer (e.g., SGD, Adam)</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Key Training Concepts</h4>
              <div className="space-y-2">
                <div>
                  <h5 className="font-medium text-foreground text-sm">Learning Rate</h5>
                  <p className="text-xs text-muted-foreground">Step size for weight updates</p>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Batch Size</h5>
                  <p className="text-xs text-muted-foreground">Number of samples per gradient update</p>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Epochs</h5>
                  <p className="text-xs text-muted-foreground">Number of complete passes through dataset</p>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Early Stopping</h5>
                  <p className="text-xs text-muted-foreground">Stop when validation performance plateaus</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 7. Optimization Algorithms */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Cog className="h-5 w-5 text-primary" />
            7. Optimization Algorithms
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Gradient Descent Variants</h4>
              <ul className="space-y-2 text-sm">
                <li>
                  <span className="font-medium text-foreground">Batch GD</span>
                  <p className="text-xs text-muted-foreground">Uses entire dataset per update</p>
                </li>
                <li>
                  <span className="font-medium text-foreground">Stochastic GD</span>
                  <p className="text-xs text-muted-foreground">Uses one sample per update</p>
                </li>
                <li>
                  <span className="font-medium text-foreground">Mini-batch GD</span>
                  <p className="text-xs text-muted-foreground">Uses a batch of samples (most common)</p>
                </li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Popular Optimizers</h4>
              <ul className="space-y-2 text-sm">
                <li>
                  <span className="font-medium text-foreground">SGD + Momentum</span>
                  <p className="text-xs text-muted-foreground">Accelerates convergence</p>
                </li>
                <li>
                  <span className="font-medium text-foreground">Adam</span>
                  <p className="text-xs text-muted-foreground">Adaptive learning rate (most popular)</p>
                </li>
                <li>
                  <span className="font-medium text-foreground">RMSprop</span>
                  <p className="text-xs text-muted-foreground">Adaptive learning rate for non-stationary problems</p>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* 8. Regularization Techniques */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            8. Regularization Techniques
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Techniques</h4>
              <div className="space-y-3">
                <div>
                  <h5 className="font-medium text-foreground text-sm">L1/L2 Regularization</h5>
                  <p className="text-xs text-muted-foreground">Adds penalty to weights to prevent overfitting</p>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Dropout</h5>
                  <p className="text-xs text-muted-foreground">Randomly drops neurons during training</p>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Batch Normalization</h5>
                  <p className="text-xs text-muted-foreground">Normalizes layer inputs to improve training</p>
                </div>
                <div>
                  <h5 className="font-medium text-foreground text-sm">Data Augmentation</h5>
                  <p className="text-xs text-muted-foreground">Creates variations of training data</p>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">When to Use</h4>
              <div className="space-y-2">
                <div className="bg-muted p-2 rounded">
                  <p className="text-xs text-foreground font-medium">High Variance (Overfitting)</p>
                  <p className="text-xs text-muted-foreground">Use: Dropout, L2 regularization</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="text-xs text-foreground font-medium">High Bias (Underfitting)</p>
                  <p className="text-xs text-muted-foreground">Use: More layers, less regularization</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="text-xs text-foreground font-medium">Slow Convergence</p>
                  <p className="text-xs text-muted-foreground">Use: Batch Normalization</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 9. Common Problems and Solutions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            9. Common Problems and Solutions
          </h2>

          <div className="space-y-3">
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Vanishing Gradients</h4>
                  <p className="text-sm text-muted-foreground">
                    Use ReLU activation, skip connections, or residual networks.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Exploding Gradients</h4>
                  <p className="text-sm text-muted-foreground">
                    Use gradient clipping, weight initialization techniques.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Overfitting</h4>
                  <p className="text-sm text-muted-foreground">
                    Use dropout, regularization, early stopping, or more data.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Slow Training</h4>
                  <p className="text-sm text-muted-foreground">
                    Use batch normalization, better optimizers (Adam), or reduce layers.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 10. Applications */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            10. Applications
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Computer Vision</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Image classification</li>
                <li>Object detection</li>
                <li>Face recognition</li>
                <li>Autonomous driving</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Natural Language Processing</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Machine translation</li>
                <li>Sentiment analysis</li>
                <li>Text generation</li>
                <li>Chatbots</li>
              </ul>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Other Areas</h4>
              <ul className="text-xs text-muted-foreground list-disc list-inside">
                <li>Speech recognition</li>
                <li>Recommendation systems</li>
                <li>Game playing (AlphaGo)</li>
                <li>Drug discovery</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 11. Best Practices */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            11. Best Practices
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Start simple</h4>
                <p className="text-sm text-muted-foreground">Begin with a basic architecture and add complexity gradually</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Monitor training</h4>
                <p className="text-sm text-muted-foreground">Track loss, accuracy, and gradients during training</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use validation data</h4>
                <p className="text-sm text-muted-foreground">Always have separate validation data for tuning</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Document experiments</h4>
                <p className="text-sm text-muted-foreground">Keep track of hyperparameters and results</p>
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
            See neural network implementations in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Neural Networks Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}