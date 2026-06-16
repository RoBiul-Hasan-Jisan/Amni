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

  Minus,
  Plus,
  Equal,
  X,
  
  ArrowRight,
  Scale,
  Maximize,
  Minimize,
  Move,

  AlignCenter,
  AlignLeft,
  AlignRight,
  Binary,
  Hash,
  Type,
  List,
  Grid3x3,
  Tag,
  Columns,
  SquareSplitVertical,
  SquareSplitHorizontal,
  ArrowUp,
  ArrowDown,
  SortAsc,
  SortDesc,
  ListOrdered
} from "lucide-react";

export default function OrdinalEncodingPage() {
  const result = getSubtopicBySlug("machine-learning", "ordinal-encoding");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python-basic",
      label: "Basic Ordinal Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Create ordinal data (has natural order)
data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large', 'Medium', 'Extra Large'],
    'Experience': ['Junior', 'Mid', 'Senior', 'Lead', 'Mid', 'Senior', 'Junior', 'Lead'],
    'Salary': [45000, 55000, 70000, 90000, 52000, 68000, 48000, 85000]
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL ORDINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. DEFINING ORDINAL ORDER")
print("=" * 50)

# Define the order of categories (from lowest to highest)
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
size_order = ['Small', 'Medium', 'Large', 'Extra Large']
experience_order = ['Junior', 'Mid', 'Senior', 'Lead']

print("Education order:", education_order)
print("Size order:", size_order)
print("Experience order:", experience_order)

print("\\n" + "=" * 50)
print("3. ORDINAL ENCODING (SKLEARN)")
print("=" * 50)

# Create encoder with specified categories order
encoder = OrdinalEncoder(
    categories=[education_order, size_order, experience_order]
)

# Fit and transform
df_encoded = pd.DataFrame(
    encoder.fit_transform(df[['Education', 'Size', 'Experience']]),
    columns=['Education_Encoded', 'Size_Encoded', 'Experience_Encoded']
)

# Combine with original data
df_result = pd.concat([df, df_encoded], axis=1)

print("Encoded data:")
print(df_result)

print("\\n" + "=" * 50)
print("4. MAPPING VERIFICATION")
print("=" * 50)

print("\\nEducation mapping:")
for i, level in enumerate(education_order):
    print(f"  {level} → {i}")

print("\\nSize mapping:")
for i, level in enumerate(size_order):
    print(f"  {level} → {i}")

print("\\nExperience mapping:")
for i, level in enumerate(experience_order):
    print(f"  {level} → {i}")`,
    },
    {
      language: "python-pandas",
      label: "Ordinal Encoding with Pandas",
      code: `import pandas as pd
import numpy as np

# Create ordinal data
data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large', 'Medium', 'Extra Large'],
    'Experience': ['Junior', 'Mid', 'Senior', 'Lead', 'Mid', 'Senior', 'Junior', 'Lead']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. ORDINAL ENCODING USING MAP")
print("=" * 50)

# Define mapping dictionaries
education_mapping = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}

size_mapping = {
    'Small': 0,
    'Medium': 1,
    'Large': 2,
    'Extra Large': 3
}

experience_mapping = {
    'Junior': 0,
    'Mid': 1,
    'Senior': 2,
    'Lead': 3
}

print("Education mapping:", education_mapping)
print("Size mapping:", size_mapping)
print("Experience mapping:", experience_mapping)

# Apply mapping
df['Education_Encoded'] = df['Education'].map(education_mapping)
df['Size_Encoded'] = df['Size'].map(size_mapping)
df['Experience_Encoded'] = df['Experience'].map(experience_mapping)

print("\\nEncoded data:")
print(df)

print("\\n" + "=" * 50)
print("3. ORDINAL ENCODING USING CATEGORY TYPE")
print("=" * 50)

# Convert to categorical with order
df['Education_Cat'] = pd.Categorical(
    df['Education'],
    categories=education_mapping.keys(),
    ordered=True
)
df['Size_Cat'] = pd.Categorical(
    df['Size'],
    categories=size_mapping.keys(),
    ordered=True
)

print("Categorical types:")
print(df[['Education_Cat', 'Size_Cat']].head())

print("\\nCodes (ordinal values):")
df['Education_Codes'] = df['Education_Cat'].cat.codes
df['Size_Codes'] = df['Size_Cat'].cat.codes
print(df[['Education_Cat', 'Education_Codes', 'Size_Cat', 'Size_Codes']].head())`,
    },
    {
      language: "python-compare",
      label: "Ordinal vs Label Encoding",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Create ordinal data
data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large', 'Medium', 'Extra Large']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

print("\\n" + "=" * 50)
print("2. LABEL ENCODING (Alphabetical Order)")
print("=" * 50)

# Label Encoding (alphabetical by default)
le_education = LabelEncoder()
le_size = LabelEncoder()

df['Education_Label'] = le_education.fit_transform(df['Education'])
df['Size_Label'] = le_size.fit_transform(df['Size'])

print("Label Encoded:")
print(df[['Education', 'Education_Label', 'Size', 'Size_Label']])

print("\\nLabel Encoding Mapping:")
print("Education:", dict(zip(le_education.classes_, le_education.transform(le_education.classes_))))
print("Size:", dict(zip(le_size.classes_, le_size.transform(le_size.classes_))))

print("\\n" + "=" * 50)
print("3. ORDINAL ENCODING (Custom Order)")
print("=" * 50)

# Ordinal Encoding with custom order
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
size_order = ['Small', 'Medium', 'Large', 'Extra Large']

encoder = OrdinalEncoder(
    categories=[education_order, size_order]
)

df_ordinal = pd.DataFrame(
    encoder.fit_transform(df[['Education', 'Size']]),
    columns=['Education_Ordinal', 'Size_Ordinal']
)

df['Education_Ordinal'] = df_ordinal['Education_Ordinal']
df['Size_Ordinal'] = df_ordinal['Size_Ordinal']

print("Ordinal Encoded:")
print(df[['Education', 'Education_Label', 'Education_Ordinal', 'Size', 'Size_Label', 'Size_Ordinal']])

print("\\n" + "=" * 50)
print("4. COMPARISON")
print("=" * 50)
print("\\nEducation Comparison:")
print(f"  Label Encoding: {dict(zip(le_education.classes_, le_education.transform(le_education.classes_)))}")
print(f"  Ordinal Encoding: {dict(zip(education_order, range(len(education_order))))}")

print("\\nSize Comparison:")
print(f"  Label Encoding: {dict(zip(le_size.classes_, le_size.transform(le_size.classes_)))}")
print(f"  Ordinal Encoding: {dict(zip(size_order, range(len(size_order))))}")

print("\\n" + "=" * 50)
print("5. KEY DIFFERENCE")
print("=" * 50)
print("Label Encoding assigns integers alphabetically (Bachelor→0, High School→1, Master→2, PhD→3)")
print("This creates incorrect order (Bachelor < High School < Master < PhD)")
print("\\nOrdinal Encoding respects actual order (High School→0, Bachelor→1, Master→2, PhD→3)")`,
    },
    {
      language: "python-inverse",
      label: "Inverse Transformation",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Create ordinal data
data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'High School', 'PhD'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large', 'Medium', 'Extra Large'],
    'Experience': ['Junior', 'Mid', 'Senior', 'Lead', 'Mid', 'Senior', 'Junior', 'Lead']
}
df = pd.DataFrame(data)

print("=" * 50)
print("1. ORIGINAL DATA")
print("=" * 50)
print(df)

# Define order
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
size_order = ['Small', 'Medium', 'Large', 'Extra Large']
experience_order = ['Junior', 'Mid', 'Senior', 'Lead']

# Encode
encoder = OrdinalEncoder(
    categories=[education_order, size_order, experience_order]
)

df_encoded = pd.DataFrame(
    encoder.fit_transform(df[['Education', 'Size', 'Experience']]),
    columns=['Education_Encoded', 'Size_Encoded', 'Experience_Encoded']
)

df_result = pd.concat([df, df_encoded], axis=1)

print("\\n" + "=" * 50)
print("2. ENCODED DATA")
print("=" * 50)
print(df_result)

print("\\n" + "=" * 50)
print("3. INVERSE TRANSFORMATION")
print("=" * 50)

# Decode back to original
df_decoded = pd.DataFrame(
    encoder.inverse_transform(df_encoded.values),
    columns=['Education_Decoded', 'Size_Decoded', 'Experience_Decoded']
)

df_final = pd.concat([df_result, df_decoded], axis=1)

print("Decoded data:")
print(df_final[['Education', 'Education_Encoded', 'Education_Decoded', 
                'Size', 'Size_Encoded', 'Size_Decoded']])

print("\\n" + "=" * 50)
print("4. VERIFICATION")
print("=" * 50)
print("Education matches:", (df['Education'] == df_final['Education_Decoded']).all())
print("Size matches:", (df['Size'] == df_final['Size_Decoded']).all())
print("Experience matches:", (df['Experience'] == df_final['Experience_Decoded']).all())`,
    },
    {
      language: "python-handle",
      label: "Handling Unknown Categories",
      code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Create training data
train_data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'Size': ['Small', 'Medium', 'Large', 'Extra Large', 'Small', 'Large']
}
train_df = pd.DataFrame(train_data)

# Create test data with unknown categories
test_data = {
    'Education': ['High School', 'Bachelor', 'Master', 'PostDoc', 'PhD'],  # PostDoc is unknown
    'Size': ['Small', 'Large', 'Medium', 'Extra Large', 'Extra']  # Extra is unknown
}
test_df = pd.DataFrame(test_data)

print("=" * 50)
print("1. TRAINING DATA")
print("=" * 50)
print(train_df)

print("\\n" + "=" * 50)
print("2. TEST DATA (WITH UNKNOWN CATEGORIES)")
print("=" * 50)
print(test_df)
print("Unknown categories: PostDoc, Extra")

print("\\n" + "=" * 50)
print("3. ORDINAL ENCODER WITH UNKNOWN CATEGORIES")
print("=" * 50)

# Define known categories
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
size_order = ['Small', 'Medium', 'Large', 'Extra Large']

# Create encoder
encoder = OrdinalEncoder(
    categories=[education_order, size_order]
)

# Fit on training data
encoder.fit(train_df[['Education', 'Size']])

print("Training categories:")
print(f"Education: {encoder.categories_[0]}")
print(f"Size: {encoder.categories_[1]}")

print("\\n" + "=" * 50)
print("4. HANDLING UNKNOWN CATEGORIES")
print("=" * 50)

# Option 1: Replace unknown with a known category
def handle_unknown_categories(df, categories_map):
    df_copy = df.copy()
    for col, valid_categories in categories_map.items():
        df_copy[col] = df_copy[col].apply(
            lambda x: x if x in valid_categories else valid_categories[0]
        )
    return df_copy

categories_map = {
    'Education': education_order,
    'Size': size_order
}

test_df_handled = handle_unknown_categories(test_df, categories_map)

print("Test data after handling unknown categories:")
print(test_df_handled)

# Transform
test_encoded = encoder.transform(test_df_handled[['Education', 'Size']])
test_encoded_df = pd.DataFrame(
    test_encoded,
    columns=['Education_Encoded', 'Size_Encoded']
)

print("\\nEncoded test data:")
print(pd.concat([test_df_handled, test_encoded_df], axis=1))`,
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is Ordinal Encoding?",
      options: [
        "Assigning random integers to categories",
        "Assigning integers to categories while preserving order",
        "Creating binary columns for categories",
        "Using target means for encoding"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding assigns integers to categories while respecting their natural order (e.g., Small=0, Medium=1, Large=2).",
    },
    {
      id: 2,
      question: "What is the main difference between Ordinal Encoding and Label Encoding?",
      options: [
        "They are the same",
        "Ordinal Encoding respects custom order, Label Encoding assigns alphabetical order",
        "Label Encoding respects custom order, Ordinal Encoding assigns alphabetical order",
        "Ordinal Encoding is for regression"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding allows you to specify the order of categories, while Label Encoding assigns integers alphabetically by default.",
    },
    {
      id: 3,
      question: "When should you use Ordinal Encoding?",
      options: [
        "For nominal data",
        "For ordinal data where order matters",
        "For binary data only",
        "For continuous data"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding is specifically designed for ordinal data where there is a meaningful order (e.g., education levels, sizes).",
    },
    {
      id: 4,
      question: "What class in scikit-learn is used for Ordinal Encoding?",
      options: [
        "LabelEncoder",
        "OrdinalEncoder",
        "OneHotEncoder",
        "TargetEncoder"
      ],
      correctAnswer: 1,
      explanation: "OrdinalEncoder is the scikit-learn class specifically designed for ordinal encoding with custom category order.",
    },
    {
      id: 5,
      question: "What is the advantage of Ordinal Encoding over One-Hot Encoding?",
      options: [
        "It creates more columns",
        "It's memory efficient with a single column",
        "It works with all algorithms",
        "It's faster to compute"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding creates only one column for the feature, making it memory efficient compared to One-Hot Encoding.",
    },
    {
      id: 6,
      question: "What is the disadvantage of Ordinal Encoding?",
      options: [
        "It creates false ordinal relationships",
        "It assumes equal spacing between categories",
        "It doesn't work with nominal data",
        "It's computationally expensive"
      ],
      correctAnswer: 1,
      explanation: "Ordinal Encoding assumes equal spacing between categories (e.g., 0,1,2), which may not reflect the actual differences.",
    },
    {
      id: 7,
      question: "How do you specify category order in OrdinalEncoder?",
      options: [
        "Using the order parameter",
        "Using the categories parameter",
        "Using the mapping parameter",
        "Using the levels parameter"
      ],
      correctAnswer: 1,
      explanation: "The categories parameter in OrdinalEncoder allows you to specify the order of categories as a list of lists.",
    },
    {
      id: 8,
      question: "What happens if test data has categories not seen in training?",
      options: [
        "The encoder works automatically",
        "It raises an error unless handled",
        "It creates new categories",
        "It removes the rows"
      ],
      correctAnswer: 1,
      explanation: "OrdinalEncoder will raise an error if it encounters unknown categories in test data, so they must be handled beforehand.",
    },
    {
      id: 9,
      question: "Which models work well with Ordinal Encoding?",
      options: [
        "Linear models (SVM, Logistic Regression)",
        "Tree-based models (Random Forest, XGBoost)",
        "Neural Networks",
        "All of the above"
      ],
      correctAnswer: 3,
      explanation: "Ordinal Encoding works with all models, but tree-based models handle it particularly well as they can split on the ordinal values.",
    },
    {
      id: 10,
      question: "What is the difference between Ordinal Encoding and Mapping using dictionaries?",
      options: [
        "They are the same",
        "Ordinal Encoding is automatic, dictionary mapping requires manual definition",
        "Dictionary mapping is for pandas only",
        "Ordinal Encoding is faster"
      ],
      correctAnswer: 1,
      explanation: "Both achieve the same result, but OrdinalEncoder is automated and integrated with sklearn pipelines, while dictionary mapping requires manual definition.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-12">
     

        {/* 1. What is Ordinal Encoding? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <ListOrdered className="h-5 w-5 text-primary" />
            1. What is Ordinal Encoding?
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="text-muted-foreground mb-3">
                Ordinal Encoding is a technique that <span className="font-semibold text-foreground">converts categorical variables into numerical format</span> while <span className="font-semibold text-foreground">preserving the natural order</span> of categories.
              </p>
              <div className="bg-muted p-3 rounded-lg text-center">
                <p className="font-mono text-lg">Small → 0, Medium → 1, Large → 2</p>
                <p className="text-xs text-muted-foreground mt-2">Preserves the order: Small &lt; Medium &lt; Large</p>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-2">Key Characteristics</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    <li>Preserves natural order</li>
                    <li>Single column output</li>
                    <li>Memory efficient</li>
                    <li>Customizable order</li>
                    <li>Works with ordinal data</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. When to Use Ordinal Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            2. When to Use Ordinal Encoding
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Best Use Cases
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Ordinal Data</span> - Categories with natural order</li>
                <li>• <span className="font-medium text-foreground">Tree-based Models</span> - Decision Trees, Random Forest</li>
                <li>• <span className="font-medium text-foreground">High Cardinality</span> - Many categories</li>
                <li>• <span className="font-medium text-foreground">Memory Constraints</span> - Single column output</li>
                <li>• <span className="font-medium text-foreground">When order is important</span></li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                When to Avoid
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Nominal Data</span> - No natural order</li>
                <li>• <span className="font-medium text-foreground">Linear Models</span> - Assumes equal spacing</li>
                <li>• <span className="font-medium text-foreground">Distance-based Models</span> - Creates false distances</li>
                <li>• <span className="font-medium text-foreground">When order is unknown</span></li>
              </ul>
            </div>
          </div>
        </section>

        {/* 3. How Ordinal Encoding Works */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            3. How Ordinal Encoding Works
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 1</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Define order</p>
                  <p className="text-xs text-muted-foreground mt-1">[Small, Medium, Large]</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 2</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Assign integers</p>
                  <p className="text-xs text-muted-foreground mt-1">Small→0, Medium→1, Large→2</p>
                </div>
              </div>

              <div className="text-center">
                <h4 className="font-semibold text-foreground mb-2">Step 3</h4>
                <div className="bg-muted p-3 rounded-lg">
                  <p className="font-mono text-sm">Transform data</p>
                  <p className="text-xs text-muted-foreground mt-1">[Small, Large, Medium] → [0, 2, 1]</p>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-muted/50 p-3 rounded-lg">
              <p className="text-xs text-muted-foreground text-center">
                <span className="font-medium text-foreground">Note:</span> The order is preserved, but equal spacing is assumed (0,1,2)
              </p>
            </div>
          </div>
        </section>

        {/* 4. Ordinal Encoding in Scikit-learn */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Cog className="h-5 w-5 text-primary" />
            4. Ordinal Encoding in Scikit-learn
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Basic Implementation</h4>
              <div className="bg-muted p-3 rounded-lg">
                <p className="font-mono text-xs text-primary">from sklearn.preprocessing import OrdinalEncoder</p>
                <p className="font-mono text-xs text-primary mt-2">encoder = OrdinalEncoder(</p>
                <p className="font-mono text-xs text-primary ml-4">categories=[['Small', 'Medium', 'Large']]</p>
                <p className="font-mono text-xs text-primary">)</p>
                <p className="font-mono text-xs text-primary mt-1">df['encoded'] = encoder.fit_transform(df[['size']])</p>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div>
                  <p className="font-medium text-foreground">Key Methods</p>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li><span className="font-medium">fit()</span>: Learn mapping</li>
                    <li><span className="font-medium">transform()</span>: Apply mapping</li>
                    <li><span className="font-medium">fit_transform()</span>: Fit and transform</li>
                    <li><span className="font-medium">inverse_transform()</span>: Decode</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2">Important Parameters</h4>
              <div className="space-y-3">
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">categories</p>
                  <p className="text-xs text-muted-foreground">Specify the order of categories</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">dtype</p>
                  <p className="text-xs text-muted-foreground">Output data type (default: float64)</p>
                </div>
                <div className="bg-muted p-2 rounded">
                  <p className="font-mono text-xs text-primary">handle_unknown</p>
                  <p className="text-xs text-muted-foreground">How to handle unknown categories</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Advantages and Disadvantages */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            5. Advantages and Disadvantages
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 dark:bg-green-950 dark:border-green-800">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Advantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Preserves Order</span> - Maintains natural order</li>
                <li>• <span className="font-medium text-foreground">Memory Efficient</span> - Only one column</li>
                <li>• <span className="font-medium text-foreground">Customizable</span> - Specify your own order</li>
                <li>• <span className="font-medium text-foreground">Works with Tree-based Models</span> - Handles ordinal values well</li>
                <li>• <span className="font-medium text-foreground">Simple Implementation</span> - Easy to use</li>
              </ul>
            </div>

            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-destructive" />
                Disadvantages
              </h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li>• <span className="font-medium text-foreground">Assumes Equal Spacing</span> - 0,1,2 implies equal differences</li>
                <li>• <span className="font-medium text-foreground">Not for Nominal Data</span> - Can't handle unordered categories</li>
                <li>• <span className="font-medium text-foreground">Linear Models Issue</span> - Assumes linear relationship</li>
                <li>• <span className="font-medium text-foreground">Distance-based Issue</span> - Creates artificial distances</li>
                <li>• <span className="font-medium text-foreground">Unknown Categories</span> - Must be handled separately</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 6. Ordinal vs Label Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Scale className="h-5 w-5 text-primary" />
            6. Ordinal vs Label Encoding
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  <ListOrdered className="h-4 w-4 text-primary" />
                  Ordinal Encoding
                </h4>
                <div className="space-y-2 text-xs">
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">Education: [High School, Bachelor, Master, PhD]</p>
                    <p className="font-mono text-xs">→ [0, 1, 2, 3]</p>
                  </div>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Respects custom order</li>
                    <li>Perfect for ordinal data</li>
                    <li>You define the order</li>
                  </ul>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  <Hash className="h-4 w-4 text-primary" />
                  Label Encoding
                </h4>
                <div className="space-y-2 text-xs">
                  <div className="bg-muted p-2 rounded">
                    <p className="font-mono text-xs">Education: [High School, Bachelor, Master, PhD]</p>
                    <p className="font-mono text-xs">→ [0, 1, 2, 3] (alphabetical)</p>
                  </div>
                  <ul className="text-muted-foreground list-disc list-inside">
                    <li>Alphabetical order by default</li>
                    <li>May create wrong order</li>
                    <li>Not recommended for ordinal data</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-primary/10 p-3 rounded">
              <p className="text-sm text-muted-foreground">
                <span className="font-medium text-foreground">Key Difference:</span> 
                Ordinal Encoding lets you specify the order, Label Encoding assigns arbitrary alphabetical order.
              </p>
            </div>
          </div>
        </section>

        {/* 7. Equal Spacing Assumption */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary" />
            7. The Equal Spacing Assumption
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-foreground mb-2">The Problem</h4>
                <p className="text-sm text-muted-foreground">
                  Ordinal Encoding assumes that the gaps between categories are equal:
                </p>
                <div className="bg-muted p-3 rounded-lg mt-2">
                  <p className="font-mono text-sm">Small → 0, Medium → 1, Large → 2</p>
                  <p className="text-xs text-muted-foreground mt-1">Assumes: Medium - Small = Large - Medium (both = 1)</p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-foreground mb-2">Real-World Example</h4>
                <div className="grid md:grid-cols-2 gap-3">
                  <div className="bg-destructive/10 p-3 rounded">
                    <p className="font-medium text-foreground text-sm">Education Levels</p>
                    <p className="text-xs text-muted-foreground">
                      High School (0) → Bachelor (1) → Master (2) → PhD (3)
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Assumes equal gaps, but real-world gaps are not equal
                    </p>
                  </div>
                  <div className="bg-destructive/10 p-3 rounded">
                    <p className="font-medium text-foreground text-sm">Experience Levels</p>
                    <p className="text-xs text-muted-foreground">
                      Junior (0) → Mid (1) → Senior (2) → Lead (3)
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Assumes equal gaps, but experience gaps vary
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-primary/10 p-3 rounded">
                <h4 className="font-semibold text-foreground text-sm">Solution</h4>
                <p className="text-sm text-muted-foreground">
                  Tree-based models handle this well. For linear models, consider using 
                  <span className="font-medium text-foreground"> Target Encoding</span> or 
                  <span className="font-medium text-foreground"> creating custom weights</span>.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 8. Handling Unknown Categories */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            8. Handling Unknown Categories
          </h2>

          <div className="bg-card border border-border rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">The Problem</h4>
            <p className="text-sm text-muted-foreground mb-3">
              OrdinalEncoder doesn't have a built-in 'ignore' parameter like OneHotEncoder. Unknown categories will raise an error.
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-destructive/10 p-3 rounded">
                <p className="font-medium text-foreground text-sm"> Wrong Approach</p>
                <div className="bg-muted p-2 rounded mt-1">
                  <p className="font-mono text-xs">encoder.fit(train_data)</p>
                  <p className="font-mono text-xs">encoder.transform(test_data)</p>
                  <p className="text-xs text-destructive mt-1">✗ Fails if test has unseen categories</p>
                </div>
              </div>

              <div className="bg-green-50 p-3 rounded dark:bg-green-950">
                <p className="font-medium text-foreground text-sm"> Best Practice</p>
                <div className="bg-muted p-2 rounded mt-1">
                  <p className="font-mono text-xs"># Handle unknown categories first</p>
                  <p className="font-mono text-xs">def handle_unknown(df, valid_categories):</p>
                  <p className="font-mono text-xs ml-4">df['col'] = df['col'].apply(</p>
                  <p className="font-mono text-xs ml-8">lambda x: x if x in valid_categories else 'Other'</p>
                  <p className="font-mono text-xs ml-4">)</p>
                  <p className="font-mono text-xs"># Then encode</p>
                  <p className="font-mono text-xs">encoder.fit_transform(df[['col']])</p>
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
                  <h4 className="font-semibold text-foreground mb-1">Using on Nominal Data</h4>
                  <p className="text-sm text-muted-foreground">
                    Applying Ordinal Encoding to categories without order creates false relationships.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Incorrect Order</h4>
                  <p className="text-sm text-muted-foreground">
                    Specifying the wrong order for categories can mislead the model.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Equal Spacing Assumption</h4>
                  <p className="text-sm text-muted-foreground">
                    Assuming categories are equally spaced when they're not.
                  </p>
                </div>
              </div>

              <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Unknown Categories</h4>
                  <p className="text-sm text-muted-foreground">
                    Not handling unseen categories in test data causes errors.
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
                <h4 className="font-semibold text-foreground mb-1">Know Your Data</h4>
                <p className="text-sm text-muted-foreground">Only use Ordinal Encoding when there's a clear natural order</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Specify Order Explicitly</h4>
                <p className="text-sm text-muted-foreground">Always define the order of categories explicitly</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Handle Unknown Categories</h4>
                <p className="text-sm text-muted-foreground">Preprocess data to handle categories not seen in training</p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Consider Tree-Based Models</h4>
                <p className="text-sm text-muted-foreground">Tree-based models handle ordinal values well and don't assume equal spacing</p>
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
            See ordinal encoding in action with these practical code examples:
          </p>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Test Your Knowledge
          </h2>
          <Quiz questions={quizQuestions} title="Ordinal Encoding Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}