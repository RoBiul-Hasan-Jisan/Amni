"use client";

import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb } from "lucide-react";
import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Cell,
} from "recharts";

// Types for our data structures
interface ComparisonData {
  aspect: string;
  machineLearning: number;
  deepLearning: number;
}

interface AccuracyData {
  year: number;
  accuracy: number;
  model: string;
}

interface ArchitectureCard {
  name: string;
  keyStrength: string;
  dataType: string;
  memory: string;
  difficulty: string;
  color: string;
  applications: string[];
}

export default function DeepLearningPage() {
  const result = getSubtopicBySlug("deep-learning", "dl-intro");
  if (!result) return null;

  const { topic, subtopic } = result;

  // State for window width to handle responsive charts
  const [isMounted, setIsMounted] = useState(false);
  const [windowWidth, setWindowWidth] = useState(1024);

  useEffect(() => {
    setIsMounted(true);
    setWindowWidth(window.innerWidth);
    
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getResponsiveFontSize = () => {
    if (!isMounted) return 12;
    return windowWidth < 640 ? 10 : 12;
  };

  // Data for visualizations
  const accuracyEvolutionData: AccuracyData[] = [
    { year: 2010, accuracy: 72, model: "Traditional ML" },
    { year: 2012, accuracy: 84.7, model: "AlexNet" },
    { year: 2014, accuracy: 92.7, model: "VGGNet" },
    { year: 2015, accuracy: 96.4, model: "ResNet" },
    { year: 2017, accuracy: 97.7, model: "SENet" },
    { year: 2019, accuracy: 98.7, model: "EfficientNet" },
    { year: 2021, accuracy: 99.2, model: "Advanced Models" },
  ];

  const comparisonData: ComparisonData[] = [
    { aspect: "Data Dependency", machineLearning: 40, deepLearning: 90 },
    { aspect: "Hardware Needs", machineLearning: 30, deepLearning: 95 },
    { aspect: "Training Time", machineLearning: 25, deepLearning: 90 },
    { aspect: "Feature Learning", machineLearning: 20, deepLearning: 95 },
    { aspect: "Interpretability", machineLearning: 85, deepLearning: 25 },
    { aspect: "Accuracy Ceiling", machineLearning: 60, deepLearning: 95 },
  ];

  const architectureData: ArchitectureCard[] = [
    {
      name: "ANN",
      keyStrength: "Versatility",
      dataType: "Tabular data",
      memory: "No",
      difficulty: "Easier",
      color: "from-blue-500 to-cyan-500",
      applications: ["Classification", "Regression", "Pattern Recognition"],
    },
    {
      name: "CNN",
      keyStrength: "Spatial patterns",
      dataType: "Images",
      memory: "No",
      difficulty: "Moderate",
      color: "from-green-500 to-emerald-500",
      applications: ["Image Recognition", "Object Detection", "Medical Imaging"],
    },
    {
      name: "RNN",
      keyStrength: "Sequential data",
      dataType: "Text/Time series",
      memory: "Yes",
      difficulty: "Harder",
      color: "from-purple-500 to-pink-500",
      applications: ["Speech Recognition", "Text Generation", "Time Series Prediction"],
    },
    {
      name: "GAN",
      keyStrength: "Generation",
      dataType: "Images/Text",
      memory: "No",
      difficulty: "Very challenging",
      color: "from-red-500 to-orange-500",
      applications: ["Image Generation", "Data Augmentation", "Super-Resolution"],
    },
  ];

  const timelineData = [
    { year: "1950s", event: "Foundations", impact: 10, category: "Theory" },
    { year: "1980s", event: "Backpropagation", impact: 30, category: "Theory" },
    { year: "2000s", event: "Machine Learning", impact: 50, category: "Practice" },
    { year: "2012", event: "AlexNet", impact: 80, category: "Practice" },
    { year: "2015", event: "TensorFlow", impact: 85, category: "Tools" },
    { year: "2018", event: "BERT/GPT", impact: 95, category: "Practice" },
    { year: "2024", event: "Generative AI", impact: 100, category: "Practice" },
  ];

  const industryImpactData = [
    { industry: "Healthcare", impact: 92, adoption: 75 },
    { industry: "Finance", impact: 85, adoption: 65 },
    { industry: "Automotive", impact: 95, adoption: 70 },
    { industry: "Retail", impact: 78, adoption: 80 },
    { industry: "Manufacturing", impact: 88, adoption: 60 },
    { industry: "Entertainment", impact: 90, adoption: 85 },
  ];

  if (!isMounted) {
    return (
      <TopicContent topic={topic} subtopic={subtopic}>
        <div className="space-y-8">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
            <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded mb-6"></div>
          </div>
        </div>
      </TopicContent>
    );
  }

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is Deep Learning?</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Deep Learning</strong> is a specialized subfield of Machine Learning 
            that uses neural networks with multiple layers to progressively extract higher-level features from raw input.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Think of deep learning like a human brain processing visual information: 
                  low-level neurons detect edges, mid-level neurons recognize shapes, and 
                  high-level neurons identify complete objects like faces or cars.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* AI Hierarchy */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">AI Hierarchy: The Relationship</h2>
          <p className="text-muted-foreground mb-4">
            Deep Learning exists within the broader domains of Artificial Intelligence and Machine Learning.
          </p>
          <div className="flex justify-center items-center py-8 bg-card rounded-lg border border-border">
            <svg width="400" height="300" viewBox="0 0 500 400" className="mx-auto">
              <circle cx="250" cy="200" r="160" fill="#E0E7FF" opacity="0.6" />
              <text x="250" y="50" textAnchor="middle" className="text-lg font-bold fill-gray-800 dark:fill-gray-200">
                Artificial Intelligence
              </text>
              <circle cx="250" cy="220" r="110" fill="#BFDBFE" opacity="0.7" />
              <text x="250" y="160" textAnchor="middle" className="text-base font-semibold fill-gray-800 dark:fill-gray-200">
                Machine Learning
              </text>
              <circle cx="250" cy="240" r="70" fill="#93C5FD" opacity="0.8" />
              <text x="250" y="245" textAnchor="middle" className="text-sm font-bold fill-gray-800 dark:fill-gray-200">
                Deep Learning
              </text>
            </svg>
          </div>
          <div className="grid md:grid-cols-3 gap-4 mt-4">
            <div className="p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg border border-border">
              <h3 className="font-semibold text-foreground mb-1">Artificial Intelligence</h3>
              <p className="text-sm text-muted-foreground">Broadest field - intelligent machines</p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-800/30 rounded-lg border border-border">
              <h3 className="font-semibold text-foreground mb-1">Machine Learning</h3>
              <p className="text-sm text-muted-foreground">Learn from data without explicit programming</p>
            </div>
            <div className="p-3 bg-blue-200 dark:bg-blue-700/30 rounded-lg border border-border">
              <h3 className="font-semibold text-foreground mb-1">Deep Learning</h3>
              <p className="text-sm text-muted-foreground">Neural networks with multiple layers</p>
            </div>
          </div>
        </section>

        {/* ML vs DL Comparison */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Machine Learning vs Deep Learning</h2>
          <div className="space-y-4 mb-6">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Machine Learning</h4>
              <p className="text-muted-foreground text-sm">
                A subset of AI that enables systems to learn from data and improve from experience 
                without being explicitly programmed. ML algorithms use statistical methods to find patterns.
              </p>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold text-foreground mb-2">Deep Learning</h4>
              <p className="text-muted-foreground text-sm">
                A specialized subset of ML based on artificial neural networks with multiple layers 
                that progressively extract higher-level features from raw input.
              </p>
            </div>
          </div>

          {/* Comparison Chart */}
          <div className="h-[400px] w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="aspect" 
                  angle={-45} 
                  textAnchor="end" 
                  height={80} 
                  interval={0} 
                  tick={{ fontSize: getResponsiveFontSize() }}
                  stroke="#9CA3AF"
                />
                <YAxis 
                  label={{ value: "Intensity (%)", angle: -90, position: "insideLeft", style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() } }}
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#1F2937", border: "none", borderRadius: "8px", fontSize: getResponsiveFontSize() }}
                />
                <Legend wrapperStyle={{ fontSize: getResponsiveFontSize() }} />
                <Bar dataKey="machineLearning" name="Machine Learning" fill="#8884d8" radius={[6, 6, 0, 0]} />
                <Bar dataKey="deepLearning" name="Deep Learning" fill="#82ca9d" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mt-6">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2">✓ ML is Ideal For:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>Smaller datasets with clear features</li>
                <li>Projects with limited computational resources</li>
                <li>Applications requiring model interpretability</li>
                <li>Structured data problems</li>
              </ul>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold text-foreground mb-2">✓ DL Shines With:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>Complex pattern recognition tasks</li>
                <li>Unstructured data (images, audio, text)</li>
                <li>Abundant data and computing power</li>
                <li>State-of-the-art performance needs</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Neural Network Architecture */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Neural Network Architecture</h2>
          <p className="text-muted-foreground mb-4">
            A <strong className="text-foreground">Neural Network</strong> is a computational model inspired by the human brain, 
            learning patterns through interconnected layers of artificial neurons.
          </p>
          
          <div className="bg-card border border-border rounded-lg p-6 mb-4">
            <div className="flex flex-wrap items-center justify-center gap-3 mb-6">
              <span className="px-4 py-2 rounded-full bg-yellow-100 dark:bg-yellow-900/40 text-yellow-800 dark:text-yellow-300 font-semibold text-sm">
                Input Layer
              </span>
              <span className="text-muted-foreground text-xl">→</span>
              <span className="px-4 py-2 rounded-full bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-300 font-semibold text-sm">
                Hidden Layers
              </span>
              <span className="text-muted-foreground text-xl">→</span>
              <span className="px-4 py-2 rounded-full bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-300 font-semibold text-sm">
                Output Layer
              </span>
            </div>

            {/* Neural Network SVG */}
            <div className="flex justify-center py-4">
              <svg width="600" height="280" viewBox="0 0 720 340" className="mx-auto">
                {/* Connection Lines */}
                <g stroke="#94A3B8" strokeWidth="1.5" opacity="0.5">
                  <line x1="110" y1="120" x2="220" y2="90" />
                  <line x1="110" y1="120" x2="220" y2="150" />
                  <line x1="110" y1="120" x2="220" y2="210" />
                  {[90,150,210].map((y1) =>
                    [70,130,190,250].map((y2, i) => (
                      <line key={`${y1}-${y2}-${i}`} x1="235" y1={y1} x2="360" y2={y2} />
                    ))
                  )}
                  {[70,130,190,250].map((y1) =>
                    [90,150,210].map((y2, i) => (
                      <line key={`${y1}-${y2}-${i}`} x1="375" y1={y1} x2="500" y2={y2} />
                    ))
                  )}
                  <line x1="515" y1="90" x2="620" y2="150" />
                  <line x1="515" y1="150" x2="620" y2="150" />
                  <line x1="515" y1="210" x2="620" y2="150" />
                </g>

                {/* Input Layer */}
                <circle cx="90" cy="120" r="26" fill="#FACC15" />
                <text x="90" y="55" textAnchor="middle" className="text-sm fill-gray-700 dark:fill-gray-200">Input</text>

                {/* Hidden Layers */}
                {[90,150,210].map((y, i) => (<circle key={i} cx="220" cy={y} r="18" fill="#22C55E" />))}
                {[70,130,190,250].map((y, i) => (<circle key={i} cx="360" cy={y} r="18" fill="#3B82F6" />))}
                {[90,150,210].map((y, i) => (<circle key={i} cx="500" cy={y} r="18" fill="#8B5CF6" />))}

                {/* Output Layer */}
                <circle cx="630" cy="150" r="26" fill="#EF4444" />
                <text x="630" y="295" textAnchor="middle" className="text-sm fill-gray-700 dark:fill-gray-200">Output</text>

                <text x="360" y="25" textAnchor="middle" className="text-base fill-gray-600 dark:fill-gray-300">Hidden Layers</text>
                <line x1="220" y1="35" x2="500" y2="35" stroke="#64748B" strokeWidth="2" />
              </svg>
            </div>
          </div>

          <div className="grid md:grid-cols-4 gap-4 mt-4">
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-1">Input Layer</h4>
              <p className="text-sm text-muted-foreground">Receives raw data (images, text, audio, numbers)</p>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-1">Hidden Layers</h4>
              <p className="text-sm text-muted-foreground">Extract features and learn complex patterns</p>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-1">Weights & Biases</h4>
              <p className="text-sm text-muted-foreground">Adjustable parameters optimized during training</p>
            </div>
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-1">Output Layer</h4>
              <p className="text-sm text-muted-foreground">Produces predictions or decisions</p>
            </div>
          </div>
        </section>

        {/* Neural Network Types */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Types of Neural Networks</h2>
          <p className="text-muted-foreground mb-4">
            Different architectures are designed for different types of data and problems.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            {architectureData.map((arch) => (
              <div key={arch.name} className={`bg-gradient-to-br ${arch.color} rounded-lg p-5 text-white transform transition-all hover:scale-[1.02]`}>
                <h3 className="text-xl font-bold mb-2">{arch.name}</h3>
                <p className="text-sm opacity-90 mb-3">
                  <strong>Key Strength:</strong> {arch.keyStrength}
                </p>
                <div className="space-y-1 text-sm">
                  <p><strong>Primary Data:</strong> {arch.dataType}</p>
                  <p><strong>Memory:</strong> {arch.memory}</p>
                  <p><strong>Difficulty:</strong> {arch.difficulty}</p>
                  <div className="mt-2">
                    <strong>Applications:</strong>
                    <ul className="list-disc list-inside mt-1 text-xs">
                      {arch.applications.map((app) => <li key={app}>{app}</li>)}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Accuracy Evolution */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Accuracy Evolution on ImageNet</h2>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={accuracyEvolutionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="year" stroke="#9CA3AF" tick={{ fontSize: getResponsiveFontSize() }} />
                <YAxis domain={[60, 100]} label={{ value: "Top-5 Accuracy (%)", angle: -90, position: "insideLeft", style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() } }} stroke="#9CA3AF" tick={{ fontSize: getResponsiveFontSize() }} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "none", borderRadius: "8px", fontSize: getResponsiveFontSize() }} />
                <Legend wrapperStyle={{ fontSize: getResponsiveFontSize() }} />
                <Line type="monotone" dataKey="accuracy" stroke="#8884d8" strokeWidth={3} dot={{ r: windowWidth < 640 ? 4 : 6 }} name="Top-5 Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Key Milestones</h4>
                <div className="grid grid-cols-4 gap-3 text-sm">
                  <div><span className="font-semibold">2010:</span> ML: 72%</div>
                  <div><span className="font-semibold">2012:</span> AlexNet: 84.7%</div>
                  <div><span className="font-semibold">2015:</span> ResNet: 96.4%</div>
                  <div><span className="font-semibold">2021+:</span> 99%+</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Industry Impact */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Industry Impact & Adoption</h2>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={industryImpactData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="industry" stroke="#9CA3AF" tick={{ fontSize: getResponsiveFontSize() }} />
                <YAxis yAxisId="left" label={{ value: "Impact (%)", angle: -90, position: "insideLeft", style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() } }} stroke="#9CA3AF" tick={{ fontSize: getResponsiveFontSize() }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: "Adoption (%)", angle: 90, position: "insideRight", style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() } }} stroke="#9CA3AF" tick={{ fontSize: getResponsiveFontSize() }} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "none", borderRadius: "8px", fontSize: getResponsiveFontSize() }} />
                <Legend wrapperStyle={{ fontSize: getResponsiveFontSize() }} />
                <Bar yAxisId="left" dataKey="impact" name="Business Impact" fill="#8884d8" />
                <Line yAxisId="right" type="monotone" dataKey="adoption" name="Adoption Rate" stroke="#82ca9d" strokeWidth={3} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Common Challenges */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Challenges</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Data Requirements</h4>
                <p className="text-sm text-muted-foreground">
                  Deep learning models typically require massive amounts of labeled data to perform well.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Computational Cost</h4>
                <p className="text-sm text-muted-foreground">
                  Training deep networks requires significant GPU/TPU resources and time.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Interpretability</h4>
                <p className="text-sm text-muted-foreground">
                  Deep learning models are often "black boxes" - understanding why they make decisions is difficult.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Key Takeaways</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Deep Learning = Neural Networks with Multiple Layers</h4>
                <p className="text-sm text-muted-foreground">Automatically learn features from data without manual engineering</p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Success Drivers</h4>
                <p className="text-sm text-muted-foreground">Big Data + Powerful Hardware + Better Algorithms + Open-source</p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">When to Use DL</h4>
                <p className="text-sm text-muted-foreground">Large datasets, unstructured data, maximum accuracy needed</p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Considerations</h4>
                <p className="text-sm text-muted-foreground">Data requirements, interpretability, computational cost</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </TopicContent>
  );
}