"use client";

import Link from "next/link";
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

const DeepLearningPage: React.FC = () => {
  // State for window width to handle responsive charts
  const [isMounted, setIsMounted] = useState(false);
  const [windowWidth, setWindowWidth] = useState(1024); // Default desktop width

  useEffect(() => {
    setIsMounted(true);
    setWindowWidth(window.innerWidth);
    
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Helper function to get responsive font size
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
      applications: [
        "Image Recognition",
        "Object Detection",
        "Medical Imaging",
      ],
    },
    {
      name: "RNN",
      keyStrength: "Sequential data",
      dataType: "Text/Time series",
      memory: "Yes",
      difficulty: "Harder",
      color: "from-purple-500 to-pink-500",
      applications: [
        "Speech Recognition",
        "Text Generation",
        "Time Series Prediction",
      ],
    },
    {
      name: "GAN",
      keyStrength: "Generation",
      dataType: "Images/Text",
      memory: "No",
      difficulty: "Very challenging",
      color: "from-red-500 to-orange-500",
      applications: [
        "Image Generation",
        "Data Augmentation",
        "Super-Resolution",
      ],
    },
  ];

  const timelineData = [
    { year: "1950s", event: "Foundations", impact: 10, category: "Theory" },
    { year: "1980s", event: "Backpropagation", impact: 30, category: "Theory" },
    {
      year: "2000s",
      event: "Machine Learning",
      impact: 50,
      category: "Practice",
    },
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

  // Don't render charts during SSR to avoid hydration mismatches
  if (!isMounted) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded w-3/4 mb-4"></div>
            <div className="h-64 bg-gray-200 rounded mb-6"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation Bar */}
      <nav className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/learn" className="text-blue-600 dark:text-blue-400 hover:underline text-sm sm:text-base">
                ← Back to Learning Path
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10">
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Deep Learning: Introduction and different between ML and DL
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-base sm:text-lg">
            Deep Learning is a specialized subfield within Artificial Intelligence and Machine Learning, 
            using neural networks with multiple layers to automatically learn features from data.
          </p>
        </div>

        {/* AI Hierarchy Venn Diagram */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            AI Hierarchy: The Relationship
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4 sm:mb-6 text-sm sm:text-base">
            Deep Learning is a specialized subfield that exists within the broader domains of Artificial Intelligence and Machine Learning.
          </p>
          <div className="flex justify-center items-center min-h-75 sm:min-h-100 overflow-x-auto">
            <svg
              width="100%"
              height="300"
              viewBox="0 0 500 400"
              className="mx-auto max-w-125"
            >
              <circle cx="250" cy="200" r="160" fill="#E0E7FF" opacity="0.6" />
              <text
                x="250"
                y="50"
                textAnchor="middle"
                className="text-sm sm:text-xl font-bold fill-gray-800 dark:fill-gray-200"
              >
                Artificial Intelligence
              </text>

              <circle cx="250" cy="220" r="110" fill="#BFDBFE" opacity="0.7" />
              <text
                x="250"
                y="160"
                textAnchor="middle"
                className="text-xs sm:text-lg font-semibold fill-gray-800 dark:fill-gray-200"
              >
                Machine Learning
              </text>

              <circle cx="250" cy="240" r="70" fill="#93C5FD" opacity="0.8" />
              <text
                x="250"
                y="245"
                textAnchor="middle"
                className="text-xs sm:text-base font-bold fill-gray-800 dark:fill-gray-200"
              >
                Deep Learning
              </text>
            </svg>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 mt-4 text-center">
            <div className="p-2 sm:p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
              <h3 className="font-bold text-blue-800 dark:text-blue-300 text-sm sm:text-base">Artificial Intelligence</h3>
              <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                Broadest field - intelligent machines
              </p>
            </div>
            <div className="p-2 sm:p-3 bg-blue-100 dark:bg-blue-800/30 rounded-lg">
              <h3 className="font-bold text-blue-800 dark:text-blue-300 text-sm sm:text-base">Machine Learning</h3>
              <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                Learn from data without explicit programming
              </p>
            </div>
            <div className="p-2 sm:p-3 bg-blue-200 dark:bg-blue-700/30 rounded-lg">
              <h3 className="font-bold text-blue-800 dark:text-blue-300 text-sm sm:text-base">Deep Learning</h3>
              <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                Neural networks with multiple layers
              </p>
            </div>
          </div>
        </div>

        {/* ML vs DL Comparison Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            Machine Learning vs Deep Learning: Comprehensive Comparison
          </h2>

          {/* Definitions */}
          <div className="mb-4 sm:mb-6 space-y-2 sm:space-y-3 text-gray-700 dark:text-gray-300 leading-relaxed text-sm sm:text-base">
            <p>
              <span className="font-semibold text-gray-900 dark:text-white">Machine Learning</span> is a subset of artificial intelligence that enables systems to learn from data and improve from experience without being explicitly programmed. ML algorithms use statistical methods to find patterns in data and make predictions or decisions.
            </p>
            <p>
              <span className="font-semibold text-gray-900 dark:text-white">Deep Learning</span> is a specialized subset of machine learning based on artificial neural networks with multiple layers that progressively extract higher-level features from raw input. In deep learning, there is minimal or no need for manual feature engineering.
            </p>
          </div>

          {/* Chart */}
          <div className="h-100 sm:h-125 w-full pb-8 sm:pb-12">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparisonData}
                margin={{
                  top: 20,
                  right: 20,
                  left: 10,
                  bottom: 80,
                }}
              >
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
                  label={{
                    value: "Intensity (%)",
                    angle: -90,
                    position: "insideLeft",
                    style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() }
                  }}
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1F2937",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: getResponsiveFontSize()
                  }}
                />
                <Legend
                  verticalAlign="bottom"
                  wrapperStyle={{
                    paddingTop: "20px",
                    fontSize: getResponsiveFontSize()
                  }}
                />
                <Bar
                  dataKey="machineLearning"
                  name="Machine Learning"
                  fill="#8884d8"
                  radius={[6, 6, 0, 0]}
                />
                <Bar
                  dataKey="deepLearning"
                  name="Deep Learning"
                  fill="#82ca9d"
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Bottom Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 mt-4 sm:mt-6">
            <div className="p-3 sm:p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <h3 className="font-bold text-base sm:text-lg text-green-800 dark:text-green-300 mb-2">
                ✓ Machine Learning is Ideal For:
              </h3>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 text-xs sm:text-sm">
                <li>Smaller datasets with clear features</li>
                <li>Projects with limited computational resources</li>
                <li>Applications requiring model interpretability</li>
                <li>Structured data problems (tabular data)</li>
                <li>Quick deployment with reasonable accuracy</li>
              </ul>
            </div>
            <div className="p-3 sm:p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <h3 className="font-bold text-base sm:text-lg text-blue-800 dark:text-blue-300 mb-2">
                ✓ Deep Learning Shines With:
              </h3>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 text-xs sm:text-sm">
                <li>Complex pattern recognition tasks</li>
                <li>Unstructured data (images, audio, text)</li>
                <li>Projects with abundant data and computing power</li>
                <li>Problems where feature engineering is difficult</li>
                <li>State-of-the-art performance applications</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Neural Network Structure */}
        <div className="bg-linear-to-br from-white to-slate-50 dark:from-gray-900 dark:to-gray-800 rounded-3xl shadow-2xl p-4 sm:p-6 md:p-8 mb-6 sm:mb-8 md:mb-10 border border-gray-200 dark:border-gray-700 overflow-hidden relative">
          {/* Decorative Glow */}
          <div className="absolute top-0 right-0 w-48 h-48 sm:w-72 sm:h-72 bg-blue-500/10 blur-3xl rounded-full"></div>
          <div className="absolute bottom-0 left-0 w-48 h-48 sm:w-72 sm:h-72 bg-purple-500/10 blur-3xl rounded-full"></div>

          {/* Header */}
          <div className="relative z-10">
            <div className="flex items-center gap-2 sm:gap-3 mb-3 sm:mb-4">
              <div>
                <h2 className="text-xl sm:text-2xl md:text-3xl font-extrabold bg-linear-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Neural Network Architecture
                </h2>
                <p className="text-gray-500 dark:text-gray-400 text-xs sm:text-sm">
                  Understanding the building blocks of Deep Learning
                </p>
              </div>
            </div>

            {/* Description */}
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-sm sm:text-base md:text-lg mb-4 sm:mb-6 md:mb-8 max-w-4xl">
              A <span className="font-semibold text-blue-600 dark:text-blue-400">Neural Network</span> is a computational model inspired by the human brain. 
              It learns patterns from data through interconnected layers of artificial neurons:
            </p>

            {/* Flow */}
            <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs sm:text-sm md:text-base font-semibold mb-6 sm:mb-8">
              <span className="px-3 py-1.5 sm:px-4 sm:py-2 rounded-full bg-yellow-100 dark:bg-yellow-900/40 text-yellow-800 dark:text-yellow-300 shadow">
                Input Layer
              </span>
              <span className="text-gray-400 text-lg sm:text-xl">→</span>
              <span className="px-3 py-1.5 sm:px-4 sm:py-2 rounded-full bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-300 shadow">
                Hidden Layers
              </span>
              <span className="text-gray-400 text-lg sm:text-xl">→</span>
              <span className="px-3 py-1.5 sm:px-4 sm:py-2 rounded-full bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-300 shadow">
                Output Layer
              </span>
            </div>
          </div>

          {/* Neural Network SVG - Responsive */}
          <div className="relative z-10 flex justify-center overflow-x-auto py-2 sm:py-4">
            <svg
              width="100%"
              height="auto"
              viewBox="0 0 720 340"
              className="drop-shadow-xl max-w-180"
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Connection Lines */}
              <g stroke="#94A3B8" strokeWidth="1.5" opacity="0.5">
                {/* Input -> Hidden 1 */}
                <line x1="110" y1="120" x2="220" y2="90" />
                <line x1="110" y1="120" x2="220" y2="150" />
                <line x1="110" y1="120" x2="220" y2="210" />

                {/* Hidden 1 -> Hidden 2 */}
                {[90,150,210].map((y1) =>
                  [70,130,190,250].map((y2, i) => (
                    <line
                      key={`${y1}-${y2}-${i}`}
                      x1="235"
                      y1={y1}
                      x2="360"
                      y2={y2}
                    />
                  ))
                )}

                {/* Hidden 2 -> Hidden 3 */}
                {[70,130,190,250].map((y1) =>
                  [90,150,210].map((y2, i) => (
                    <line
                      key={`${y1}-${y2}-${i}`}
                      x1="375"
                      y1={y1}
                      x2="500"
                      y2={y2}
                    />
                  ))
                )}

                {/* Hidden 3 -> Output */}
                <line x1="515" y1="90" x2="620" y2="150" />
                <line x1="515" y1="150" x2="620" y2="150" />
                <line x1="515" y1="210" x2="620" y2="150" />
              </g>

              {/* Input Layer */}
              <circle cx="90" cy="120" r="26" fill="#FACC15" />
              <text
                x="90"
                y="55"
                textAnchor="middle"
                className="text-xs sm:text-sm fill-gray-700 dark:fill-gray-200"
                fontSize="12"
              >
                Input
              </text>

              {/* Hidden Layer 1 */}
              {[90,150,210].map((y, i) => (
                <circle key={i} cx="220" cy={y} r="18" fill="#22C55E" />
              ))}

              {/* Hidden Layer 2 */}
              {[70,130,190,250].map((y, i) => (
                <circle key={i} cx="360" cy={y} r="18" fill="#3B82F6" />
              ))}

              {/* Hidden Layer 3 */}
              {[90,150,210].map((y, i) => (
                <circle key={i} cx="500" cy={y} r="18" fill="#8B5CF6" />
              ))}

              {/* Output */}
              <circle cx="630" cy="150" r="26" fill="#EF4444" />
              <text
                x="630"
                y="295"
                textAnchor="middle"
                className="text-xs sm:text-sm fill-gray-700 dark:fill-gray-200"
                fontSize="12"
              >
                Output
              </text>

              {/* Labels */}
              <text
                x="360"
                y="25"
                textAnchor="middle"
                className="text-base sm:text-lg fill-gray-600 dark:fill-gray-300"
                fontSize="14"
              >
                Hidden Layers
              </text>
              <line x1="220" y1="35" x2="500" y2="35" stroke="#64748B" strokeWidth="2" />
            </svg>
          </div>

          {/* Information Cards */}
          <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 md:gap-5 mt-4 sm:mt-6 md:mt-8">
            <div className="p-3 sm:p-4 md:p-5 rounded-2xl bg-yellow-100 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 shadow-md hover:scale-105 transition">
              <h3 className="font-bold text-yellow-800 dark:text-yellow-300 mb-1 sm:mb-2 text-sm sm:text-base">
                Input Layer
              </h3>
              <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300">
                Receives raw data such as images, text, audio, or numbers.
              </p>
            </div>
            <div className="p-3 sm:p-4 md:p-5 rounded-2xl bg-green-100 dark:bg-green-900/20 border border-green-200 dark:border-green-700 shadow-md hover:scale-105 transition">
              <h3 className="font-bold text-green-800 dark:text-green-300 mb-1 sm:mb-2 text-sm sm:text-base">
                Hidden Layers
              </h3>
              <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300">
                Perform feature extraction and learn complex patterns.
              </p>
            </div>
            <div className="p-3 sm:p-4 md:p-5 rounded-2xl bg-blue-100 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 shadow-md hover:scale-105 transition">
              <h3 className="font-bold text-blue-800 dark:text-blue-300 mb-1 sm:mb-2 text-sm sm:text-base">
                Weights & Biases
              </h3>
              <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300">
                Adjustable parameters optimized during training.
              </p>
            </div>
            <div className="p-3 sm:p-4 md:p-5 rounded-2xl bg-red-100 dark:bg-red-900/20 border border-red-200 dark:border-red-700 shadow-md hover:scale-105 transition">
              <h3 className="font-bold text-red-800 dark:text-red-300 mb-1 sm:mb-2 text-sm sm:text-base">
                Output Layer
              </h3>
              <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300">
                Produces predictions, classifications, or decisions.
              </p>
            </div>
          </div>
        </div>

        {/* Biological Inspiration */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            Biological Inspiration
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm sm:text-base">
            Deep Learning is fundamentally inspired by the structure and function of the human brain:
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
            <div className="text-center p-2 sm:p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="font-semibold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Interconnected Neurons</div>
            </div>
            <div className="text-center p-2 sm:p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="font-semibold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Layer-wise Processing</div>
            </div>
            <div className="text-center p-2 sm:p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="font-semibold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Pattern Recognition</div>
            </div>
            <div className="text-center p-2 sm:p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="font-semibold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">Improves with Experience</div>
            </div>
          </div>
        </div>

        {/* Architectures Section */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 mb-6 sm:mb-8">
          {architectureData.map((arch) => (
            <div
              key={arch.name}
              className={`bg-linear-to-br ${arch.color} rounded-xl shadow-lg p-4 sm:p-6 text-white transform transition-all hover:scale-105`}
            >
              <h3 className="text-xl sm:text-2xl font-bold mb-2">{arch.name}</h3>
              <p className="text-base sm:text-lg opacity-90 mb-3 sm:mb-4">
                <strong>Key Strength:</strong> {arch.keyStrength}
              </p>
              <div className="space-y-1 sm:space-y-2 text-sm sm:text-base">
                <p>
                  <strong>Primary Data:</strong> {arch.dataType}
                </p>
                <p>
                  <strong>Memory:</strong> {arch.memory}
                </p>
                <p>
                  <strong>Training Difficulty:</strong> {arch.difficulty}
                </p>
                <div className="mt-2 sm:mt-3">
                  <strong>Applications:</strong>
                  <ul className="list-disc list-inside mt-1 text-xs sm:text-sm">
                    {arch.applications.map((app) => (
                      <li key={app}>{app}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Performance Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            Accuracy Evolution on ImageNet
          </h2>
          <div className="h-75 sm:h-100 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={accuracyEvolutionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="year" 
                  stroke="#9CA3AF" 
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <YAxis 
                  domain={[60, 100]} 
                  label={{ 
                    value: "Top-5 Accuracy (%)", 
                    angle: -90, 
                    position: "insideLeft", 
                    style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() }
                  }} 
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: "#1F2937", 
                    border: "none", 
                    borderRadius: "8px",
                    fontSize: getResponsiveFontSize()
                  }} 
                />
                <Legend wrapperStyle={{ fontSize: getResponsiveFontSize() }} />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#8884d8"
                  strokeWidth={3}
                  dot={{ r: windowWidth < 640 ? 4 : 6 }}
                  name="Top-5 Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 sm:mt-4 p-3 sm:p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h3 className="font-bold text-blue-800 dark:text-blue-300 mb-2 text-sm sm:text-base">Key Milestones</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
              <div className="text-center">
                <div className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">2010</div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Traditional ML: 72%</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">2012</div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">AlexNet: 84.7%</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">2015</div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">ResNet: 96.4%</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-gray-800 dark:text-gray-200 text-xs sm:text-sm">2021+</div>
                <div className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Advanced: 99%</div>
              </div>
            </div>
          </div>
        </div>

        {/* Industry Impact Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            Industry Impact & Adoption
          </h2>
          <div className="h-75 sm:h-100 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={industryImpactData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="industry" 
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                  angle={windowWidth < 640 ? -30 : 0}
                  textAnchor={windowWidth < 640 ? "end" : "middle"}
                  height={windowWidth < 640 ? 60 : 30}
                />
                <YAxis 
                  yAxisId="left" 
                  label={{ 
                    value: "Impact (%)", 
                    angle: -90, 
                    position: "insideLeft", 
                    style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() }
                  }} 
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <YAxis 
                  yAxisId="right" 
                  orientation="right" 
                  label={{ 
                    value: "Adoption (%)", 
                    angle: 90, 
                    position: "insideRight", 
                    style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() }
                  }} 
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: "#1F2937", 
                    border: "none", 
                    borderRadius: "8px",
                    fontSize: getResponsiveFontSize()
                  }} 
                />
                <Legend wrapperStyle={{ fontSize: getResponsiveFontSize() }} />
                <Bar yAxisId="left" dataKey="impact" name="Business Impact" fill="#8884d8" />
                <Line yAxisId="right" type="monotone" dataKey="adoption" name="Adoption Rate" stroke="#82ca9d" strokeWidth={3} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Timeline Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 mb-6 sm:mb-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">
            The Deep Learning Revolution: Timeline
          </h2>
          <div className="h-62.5 sm:h-75 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="year" 
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                  angle={windowWidth < 640 ? -30 : 0}
                  textAnchor={windowWidth < 640 ? "end" : "middle"}
                  height={windowWidth < 640 ? 50 : 30}
                />
                <YAxis 
                  label={{ 
                    value: "Impact →", 
                    angle: -90, 
                    position: "insideLeft", 
                    style: { fill: "#9CA3AF", fontSize: getResponsiveFontSize() }
                  }} 
                  stroke="#9CA3AF"
                  tick={{ fontSize: getResponsiveFontSize() }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: "#1F2937", 
                    border: "none", 
                    borderRadius: "8px",
                    fontSize: getResponsiveFontSize()
                  }} 
                />
                <Bar dataKey="impact" fill="#82ca9d">
                  {timelineData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.category === "Theory" ? "#8884d8" : entry.category === "Practice" ? "#82ca9d" : "#ffc658"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-3 sm:gap-4 mt-3 sm:mt-4">
            <div className="flex items-center">
              <div className="w-3 h-3 sm:w-4 sm:h-4 bg-[#8884d8] rounded mr-1 sm:mr-2"></div>
              <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Theory</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 sm:w-4 sm:h-4 bg-[#82ca9d] rounded mr-1 sm:mr-2"></div>
              <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Practice</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 sm:w-4 sm:h-4 bg-[#ffc658] rounded mr-1 sm:mr-2"></div>
              <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Tools</span>
            </div>
          </div>
          <p className="text-center text-gray-600 dark:text-gray-400 mt-3 sm:mt-4 text-xs sm:text-sm">
            From Turing to Transformers: Deep learning emerged when big data, powerful hardware, and better algorithms converged
          </p>
        </div>

        {/* Key Takeaways */}
        <div className="mt-6 sm:mt-8 p-4 sm:p-6 bg-linear-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/30 dark:to-purple-950/30 rounded-xl border border-indigo-200 dark:border-indigo-800">
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-900 dark:text-white">Key Takeaways</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <div className="flex items-start space-x-2 sm:space-x-3">
              <div>
                <strong className="text-gray-800 dark:text-gray-200 text-sm sm:text-base">Deep Learning = Neural networks with multiple layers</strong>
                <br />
                <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                  Automatically learn features from data
                </span>
              </div>
            </div>
            <div className="flex items-start space-x-2 sm:space-x-3">
              <div>
                <strong className="text-gray-800 dark:text-gray-200 text-sm sm:text-base">Success driven by:</strong>
                <br />
                <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                  Big Data + Powerful Hardware + Better Algorithms + Open-source
                </span>
              </div>
            </div>
            <div className="flex items-start space-x-2 sm:space-x-3">
              <div>
                <strong className="text-gray-800 dark:text-gray-200 text-sm sm:text-base">When to use DL:</strong>
                <br />
                <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                  Large datasets, unstructured data, maximum accuracy needed
                </span>
              </div>
            </div>
            <div className="flex items-start space-x-2 sm:space-x-3">
              <div>
                <strong className="text-gray-800 dark:text-gray-200 text-sm sm:text-base">Challenges:</strong>
                <br />
                <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                  Data requirements, interpretability, computational cost
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="flex justify-between items-center pt-4 sm:pt-6 mt-4 sm:mt-6 border-t border-gray-200 dark:border-gray-700">
          <Link
            href="/learn"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 sm:px-6 rounded-lg transition duration-200 text-sm sm:text-base"
          >
            ← Back to Learning Path
          </Link>
          <div className="text-gray-500 dark:text-gray-400 text-xs sm:text-sm">
            Deep Learning vs Machine Learning
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeepLearningPage;