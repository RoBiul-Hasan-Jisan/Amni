"use client";

import { useRef, useState, memo } from "react";
import { TopicContent } from "@/components/topic-content";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Variant {
  title: string;
  desc: string;
}

interface Section {
  id: string;
  num: string;
  title: string;
  tagLabel: string;
  tagColor: string;
  tagTextColor: string;
  dotBg: string;
  dotBorder: string;
  content: React.ReactNode;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <span className="font-mono text-[11px] text-muted-foreground tracking-wide">{label}</span>
      <div className="flex-1 h-px bg-border" />
    </div>
  );
}

function CardTitle({ children }: { children: React.ReactNode }) {
  return (
    <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground mb-3">
      {children}
    </p>
  );
}

function BulletList({ items }: { items: string[] }) {
  return (
    <ul className="space-y-1.5">
      {items.map((item, i) => (
        <li key={i} className="flex gap-2 text-[13px] text-muted-foreground leading-snug">
          <span className="font-mono text-[11px] text-muted-foreground/60 mt-0.5 shrink-0">—</span>
          {item}
        </li>
      ))}
    </ul>
  );
}

function VariantCard({ title, desc }: Variant) {
  return (
    <div className="bg-muted/40 border border-border rounded-lg p-3.5 transition-all hover:border-muted-foreground/20 hover:bg-muted/60">
      <p className="text-[13px] font-medium mb-1.5">{title}</p>
      <p className="text-[12px] text-muted-foreground leading-relaxed">{desc}</p>
    </div>
  );
}

function Highlight({ children }: { children: React.ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = typeof children === 'string' ? children : '';

  const handleCopy = () => {
    if (text) {
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="relative group inline-flex items-center"
      aria-label={`Copy ${text} to clipboard`}
    >
      <code className="font-mono text-[12px] bg-muted border border-border px-1.5 py-0.5 rounded text-foreground transition-all group-hover:border-muted-foreground/30">
        {children}
      </code>
      {copied && (
        <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] bg-foreground text-background px-1.5 py-0.5 rounded whitespace-nowrap z-10">
          Copied!
        </span>
      )}
    </button>
  );
}

function InfoTooltip({ content, children }: { content: string; children: React.ReactNode }) {
  const [show, setShow] = useState(false);
  
  return (
    <div 
      className="relative inline-block" 
      onMouseEnter={() => setShow(true)} 
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
    >
      {children}
      {show && (
        <div className="absolute z-20 bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-foreground text-background text-[11px] rounded whitespace-nowrap pointer-events-none">
          {content}
          <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-foreground" />
        </div>
      )}
    </div>
  );
}

// ─── Architecture Diagrams (memoized) ─────────────────────────────────────────

const MLPDiagram = memo(function MLPDiagram() {
  const inputNodes = [0, 1, 2, 3];
  const hiddenNodes = [0, 1, 2];
  const outputNodes = [0, 1];

  return (
    <div className="bg-muted/30 border border-border rounded-xl p-5 mb-5">
      <SectionDivider label="architecture" />
      <div className="flex items-center justify-center gap-0 py-2" aria-hidden="true">
        {/* Input Layer */}
        <div className="flex flex-col items-center gap-2">
          <span className="font-mono text-[10px] text-muted-foreground mb-1">Input</span>
          <div className="flex flex-col gap-2">
            {inputNodes.map((n) => (
              <InfoTooltip key={n} content={`Input node ${n + 1}`}>
                <div className="w-5 h-5 rounded-full bg-foreground/10 border-[1.5px] border-foreground/20 cursor-help" />
              </InfoTooltip>
            ))}
          </div>
        </div>
        {/* Arrow SVG */}
        <svg width="48" height="100" className="shrink-0">
          {inputNodes.flatMap((_, i) =>
            hiddenNodes.map((_, j) => (
              <line
                key={`i${i}h${j}`}
                x1={4} y1={16 + i * 22}
                x2={44} y2={18 + j * 26}
                stroke="currentColor"
                strokeWidth="0.7"
                className="text-border"
              />
            ))
          )}
        </svg>
        {/* Hidden Layer */}
        <div className="flex flex-col items-center gap-2">
          <span className="font-mono text-[10px] text-muted-foreground mb-1">Hidden</span>
          <div className="flex flex-col gap-2.5">
            {hiddenNodes.map((n) => (
              <InfoTooltip key={n} content={`Hidden node ${n + 1}`}>
                <div className="w-5 h-5 rounded-full cursor-help" style={{ background: "#EEEDFE", border: "1.5px solid #AFA9EC" }} />
              </InfoTooltip>
            ))}
          </div>
        </div>
        {/* Arrow SVG */}
        <svg width="40" height="80" className="shrink-0">
          {hiddenNodes.flatMap((_, i) =>
            outputNodes.map((_, j) => (
              <line
                key={`h${i}o${j}`}
                x1={4} y1={14 + i * 24}
                x2={36} y2={20 + j * 28}
                stroke="currentColor"
                strokeWidth="0.7"
                className="text-border"
              />
            ))
          )}
        </svg>
        {/* Output Layer */}
        <div className="flex flex-col items-center gap-2">
          <span className="font-mono text-[10px] text-muted-foreground mb-1">Output</span>
          <div className="flex flex-col gap-3">
            {outputNodes.map((n) => (
              <InfoTooltip key={n} content={`Output node ${n + 1}`}>
                <div className="w-5 h-5 rounded-full cursor-help" style={{ background: "#534AB7", border: "1.5px solid #3C3489" }} />
              </InfoTooltip>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

const CNNDiagram = memo(function CNNDiagram() {
  const blocks = [
    { label: "224×224", w: 48, h: 48, bg: "#E6F1FB", color: "#185FA5", sub: "input" },
    { label: "Conv", w: 42, h: 42, bg: "#B5D4F4", color: "#185FA5" },
    { label: "Conv", w: 36, h: 36, bg: "#B5D4F4", color: "#185FA5" },
    { label: "Pool", w: 28, h: 28, bg: "#D3D1C7", color: "#5F5E5A" },
    { label: "Conv", w: 24, h: 24, bg: "#85B7EB", color: "#0C447C" },
    { label: "Pool", w: 18, h: 18, bg: "#D3D1C7", color: "#5F5E5A" },
    { label: "Flatten", w: 52, h: 24, bg: "#EEEDFE", color: "#3C3489" },
    { label: "FC + out", w: 52, h: 28, bg: "#534AB7", color: "#CECBF6" },
  ];

  return (
    <div className="bg-muted/30 border border-border rounded-xl p-5 mb-5">
      <SectionDivider label="pipeline" />
      <div className="flex items-center gap-2 overflow-x-auto py-1" aria-hidden="true">
        {blocks.map((b, i) => (
          <div key={i} className="flex items-center gap-2 shrink-0">
            <InfoTooltip content={`${b.label} layer${b.sub ? ` (${b.sub})` : ''}`}>
              <div
                className="rounded flex items-center justify-center font-mono shrink-0 cursor-help"
                style={{ width: b.w, height: b.h, background: b.bg, color: b.color, fontSize: 10 }}
              >
                {b.sub ? (
                  <div className="flex flex-col items-center">
                    <span style={{ fontSize: 8 }}>{b.sub}</span>
                    <span>{b.label}</span>
                  </div>
                ) : b.label}
              </div>
            </InfoTooltip>
            {i < blocks.length - 1 && (
              <span className="text-muted-foreground text-sm">→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
});

const AutoencoderDiagram = memo(function AutoencoderDiagram() {
  return (
    <div className="bg-muted/30 border border-border rounded-xl p-5 mb-5">
      <SectionDivider label="encoder → bottleneck → decoder" />
      <div className="flex items-center justify-center gap-1 py-1" aria-hidden="true">
        {/* Input nodes */}
        <div className="flex flex-col gap-1.5">
          {[0,1,2,3].map(i => (
            <InfoTooltip key={i} content={`Input dimension ${i + 1}`}>
              <div className="w-3.5 h-3.5 rounded-full cursor-help" style={{ background: "#FAEEDA", border: "1.5px solid #EF9F27" }} />
            </InfoTooltip>
          ))}
        </div>
        <svg width="28" height="72" className="shrink-0">
          {[0,1,2,3].flatMap((_, i) => [0,1,2].map((_, j) => (
            <line key={`e${i}${j}`} x1={2} y1={10+i*18} x2={26} y2={16+j*20} stroke="currentColor" strokeWidth="0.7" className="text-border" />
          )))}
        </svg>
        {/* Encoder nodes */}
        <div className="flex flex-col gap-2">
          {[0,1,2].map(i => (
            <InfoTooltip key={i} content={`Encoder node ${i + 1}`}>
              <div className="w-3.5 h-3.5 rounded-full cursor-help" style={{ background: "#FAC775", border: "1.5px solid #BA7517" }} />
            </InfoTooltip>
          ))}
        </div>
        <svg width="24" height="60" className="shrink-0">
          {[0,1,2].map((_, i) => (
            <line key={i} x1={2} y1={12+i*18} x2={22} y2={30} stroke="currentColor" strokeWidth="0.7" className="text-border" />
          ))}
        </svg>
        {/* Bottleneck */}
        <div className="flex flex-col items-center gap-1">
          <InfoTooltip content="Latent space representation (z)">
            <div className="w-5 h-5 rounded-full cursor-help" style={{ background: "#BA7517", border: "2px solid #854F0B" }} />
          </InfoTooltip>
          <span className="font-mono text-[9px]" style={{ color: "#854F0B" }}>z</span>
        </div>
        <svg width="24" height="60" className="shrink-0">
          {[0,1,2].map((_, i) => (
            <line key={i} x1={2} y1={30} x2={22} y2={12+i*18} stroke="currentColor" strokeWidth="0.7" className="text-border" />
          ))}
        </svg>
        {/* Decoder nodes */}
        <div className="flex flex-col gap-2">
          {[0,1,2].map(i => (
            <InfoTooltip key={i} content={`Decoder node ${i + 1}`}>
              <div className="w-3.5 h-3.5 rounded-full cursor-help" style={{ background: "#FAC775", border: "1.5px solid #BA7517" }} />
            </InfoTooltip>
          ))}
        </div>
        <svg width="28" height="72" className="shrink-0">
          {[0,1,2].flatMap((_, i) => [0,1,2,3].map((_, j) => (
            <line key={`d${i}${j}`} x1={2} y1={16+i*20} x2={26} y2={10+j*18} stroke="currentColor" strokeWidth="0.7" className="text-border" />
          )))}
        </svg>
        {/* Output nodes */}
        <div className="flex flex-col gap-1.5">
          {[0,1,2,3].map(i => (
            <InfoTooltip key={i} content={`Reconstructed output ${i + 1}`}>
              <div className="w-3.5 h-3.5 rounded-full cursor-help" style={{ background: "#FAEEDA", border: "1.5px solid #EF9F27" }} />
            </InfoTooltip>
          ))}
        </div>
      </div>
    </div>
  );
});

const RNNUnrolledDiagram = memo(function RNNUnrolledDiagram() {
  return (
    <div className="bg-muted/30 border border-border rounded-xl p-5 mb-5">
      <SectionDivider label="unrolled through time" />
      <div className="flex items-center justify-center gap-6 py-4 relative">
        {[0, 1, 2, 3].map((t) => (
          <div key={t} className="flex flex-col items-center gap-2">
            <div className="relative">
              <div className="w-16 h-16 rounded-lg bg-[#E1F5EE] border-2 border-[#5DCAA5] flex items-center justify-center">
                <span className="font-mono text-sm font-medium">h<sub>{t}</sub></span>
              </div>
              {t === 0 && (
                <div className="absolute -top-6 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#0F6E56]/10 border border-[#0F6E56]/20 flex items-center justify-center">
                  <span className="font-mono text-[9px] text-[#0F6E56]">x<sub>{t}</sub></span>
                </div>
              )}
            </div>
            <span className="font-mono text-[10px] text-muted-foreground">t={t}</span>
            {t < 3 && (
              <div className="absolute translate-x-[72px]">
                <svg width="24" height="20">
                  <path d="M20,10 L4,10" stroke="#5DCAA5" strokeWidth="2" />
                  <polygon points="20,10 14,6 14,14" fill="#5DCAA5" />
                </svg>
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="text-center text-[11px] text-muted-foreground mt-3">
        Each time step shares the same weights (W<sub>x</sub>, W<sub>h</sub>)
      </p>
    </div>
  );
});

// ─── Main Page ────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { id: "sec-mlp", num: "01", label: "MLP / ANN", dotBg: "#EEEDFE", dotBorder: "#AFA9EC" },
  { id: "sec-cnn", num: "02", label: "CNN",        dotBg: "#E6F1FB", dotBorder: "#85B7EB" },
  { id: "sec-rnn", num: "03", label: "RNN",        dotBg: "#E1F5EE", dotBorder: "#5DCAA5" },
  { id: "sec-ae",  num: "04", label: "Autoencoder",dotBg: "#FAEEDA", dotBorder: "#EF9F27" },
  { id: "sec-gan", num: "05", label: "GAN",        dotBg: "#FAECE7", dotBorder: "#F0997B" },
];

// Topic and subtopic definitions
const topic = {
  id: "deep-learning",
  title: "Deep Learning",
  description: "Advanced neural networks and deep learning architectures",
  icon: "sparkles",
  subtopics: [
    { id: "dl-intro", title: "Introduction to Deep Learning", slug: "dl-intro" },
    { id: "type-NN", title: "Types of Neural Networks", slug: "type-NN" },
    { id: "ann", title: "Artificial Neural Networks (ANN)", slug: "ann" },
  ]
};

const subtopic = {
  id: "type-NN",
  title: "Types of Neural Networks",
  slug: "type-NN",
  topicId: "deep-learning"
};

export default function TypesOfNeuralNetworks() {
  const refs = useRef<Record<string, HTMLDivElement | null>>({});
  const [activeSection, setActiveSection] = useState<string>("sec-mlp");

  function scrollToSection(id: string) {
    refs.current[id]?.scrollIntoView({ behavior: "smooth" });
    setActiveSection(id);
  }

  function setRef(id: string) {
    return (el: HTMLDivElement | null) => {
      refs.current[id] = el;
    };
  }

  // Handle scroll spy
  const handleScroll = () => {
    const scrollPosition = window.scrollY + 100;
    for (const [id, ref] of Object.entries(refs.current)) {
      if (ref) {
        const offsetTop = ref.offsetTop;
        const offsetBottom = offsetTop + ref.offsetHeight;
        if (scrollPosition >= offsetTop && scrollPosition < offsetBottom) {
          if (activeSection !== id) setActiveSection(id);
          break;
        }
      }
    }
  };

  if (typeof window !== 'undefined') {
    window.addEventListener('scroll', handleScroll);
  }

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <main className="max-w-3xl mx-auto px-6 pb-20">

        {/* Hero */}
        <header className="py-12 border-b border-border mb-10">
          <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground mb-3">
            Deep Learning · Architecture Guide
          </p>
          <h1 className="font-serif text-5xl font-normal leading-tight mb-4">
            Types of Neural Networks
          </h1>
          <p className="text-[15px] text-muted-foreground leading-relaxed max-w-lg">
            A structured reference covering the five core families of neural networks —
            their architectures, training dynamics, variants, and applications.
          </p>
        </header>

        {/* Nav Tree */}
        <nav
          aria-label="Network types"
          className="grid grid-cols-5 gap-2 mb-12 p-5 bg-muted/40 border border-border rounded-xl sticky top-4 z-10 backdrop-blur-sm bg-background/80"
        >
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className={`flex flex-col items-center gap-1.5 px-2 py-3 rounded-lg border transition-all ${
                activeSection === item.id
                  ? 'bg-background border-border shadow-sm'
                  : 'border-transparent hover:bg-background hover:border-border'
              }`}
              aria-current={activeSection === item.id ? 'location' : undefined}
            >
              <span className="font-mono text-[10px] text-muted-foreground">{item.num}</span>
              <div
                className="w-7 h-7 rounded-full transition-transform group-hover:scale-110"
                style={{ background: item.dotBg, border: `1.5px solid ${item.dotBorder}` }}
                aria-hidden="true"
              />
              <span className="text-[11px] font-medium text-center text-muted-foreground leading-tight">
                {item.label}
              </span>
            </button>
          ))}
        </nav>

        {/* ── 1. MLP ─────────────────────────────────────────────────────────── */}
        <section
          ref={setRef("sec-mlp")}
          id="sec-mlp"
          className="mb-14 pb-14 border-b border-border scroll-mt-6"
          aria-labelledby="mlp-title"
        >
          <div className="flex items-start gap-4 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground border border-border rounded-md px-2 py-1 mt-1 shrink-0">
              01
            </span>
            <h2 id="mlp-title" className="text-3xl font-serif font-normal leading-tight">
              Multi-Layer Perceptron
              <span
                className="inline-block font-mono text-[10px] px-2 py-0.5 rounded-full ml-3 align-middle"
                style={{ background: "#EEEDFE", color: "#3C3489" }}
              >
                Feed-Forward
              </span>
            </h2>
          </div>

          <p className="text-[14px] text-muted-foreground leading-relaxed mb-5">
            MLPs (also called <Highlight>ANN</Highlight>) are the simplest feed-forward neural networks.
            Data passes linearly from input to output, making them ideal for supervised learning —
            classification and regression — especially where <strong className="font-medium">non-linear relationships</strong> exist in data.
          </p>

          <MLPDiagram />

          <SectionDivider label="training process" />
          <div className="grid grid-cols-3 gap-3 mb-5">
            {[
              { title: "Forward pass",    desc: "Data flows through layers generating predictions" },
              { title: "Backpropagation", desc: "Error propagated backward to adjust weights" },
              { title: "Gradient descent",desc: "Connection strengths updated iteratively" },
            ].map((c) => (
              <div key={c.title} className="bg-background border border-border rounded-xl p-4 transition-all hover:shadow-sm">
                <CardTitle>{c.title}</CardTitle>
                <p className="text-[13px] text-muted-foreground leading-snug">{c.desc}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-background border border-border rounded-xl p-4">
              <CardTitle>Strengths</CardTitle>
              <BulletList items={[
                "Universal function approximators",
                "Good for non-linear tabular data",
                "Simple, versatile architecture",
              ]} />
            </div>
            <div className="bg-background border border-border rounded-xl p-4">
              <CardTitle>Limitations</CardTitle>
              <BulletList items={[
                "Cannot handle spatial data (images)",
                "Struggles with sequential data",
                "Requires flattened input",
              ]} />
            </div>
          </div>
        </section>

        {/* ── 2. CNN ─────────────────────────────────────────────────────────── */}
        <section
          ref={setRef("sec-cnn")}
          id="sec-cnn"
          className="mb-14 pb-14 border-b border-border scroll-mt-6"
          aria-labelledby="cnn-title"
        >
          <div className="flex items-start gap-4 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground border border-border rounded-md px-2 py-1 mt-1 shrink-0">
              02
            </span>
            <h2 id="cnn-title" className="text-3xl font-serif font-normal leading-tight">
              Convolutional Neural Networks
              <span
                className="inline-block font-mono text-[10px] px-2 py-0.5 rounded-full ml-3 align-middle"
                style={{ background: "#E6F1FB", color: "#185FA5" }}
              >
                Spatial
              </span>
            </h2>
          </div>

          <p className="text-[14px] text-muted-foreground leading-relaxed mb-5">
            CNNs are feed-forward networks requiring at least one <Highlight>convolution layer</Highlight>.
            Specialized for grid-like data (images, video), they are the backbone of computer vision —
            from self-driving cars to medical imaging.
          </p>

          <CNNDiagram />

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-background border border-border rounded-xl p-4">
              <CardTitle>Key operations</CardTitle>
              <BulletList items={[
                "Kernels — small filter matrices that extract features",
                "Feature maps — outputs after applying filters",
                "Max pooling — selects maximum in region",
                "Stride — controls filter movement step",
                "Padding — preserves spatial dimensions",
              ]} />
            </div>
            <div className="bg-background border border-border rounded-xl p-4">
              <CardTitle>Famous architectures</CardTitle>
              <BulletList items={[
                "LeNet-5 (1998) — digit recognition pioneer",
                "AlexNet (2012) — ImageNet winner",
                "VGGNet (2014) — depth demonstrated",
                "ResNet (2015) — skip connections",
                "Inception/GoogLeNet — multi-scale",
              ]} />
            </div>
          </div>
        </section>

        {/* ── 3. RNN ─────────────────────────────────────────────────────────── */}
        <section
          ref={setRef("sec-rnn")}
          id="sec-rnn"
          className="mb-14 pb-14 border-b border-border scroll-mt-6"
          aria-labelledby="rnn-title"
        >
          <div className="flex items-start gap-4 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground border border-border rounded-md px-2 py-1 mt-1 shrink-0">
              03
            </span>
            <h2 id="rnn-title" className="text-3xl font-serif font-normal leading-tight">
              Recurrent Neural Networks
              <span
                className="inline-block font-mono text-[10px] px-2 py-0.5 rounded-full ml-3 align-middle"
                style={{ background: "#E1F5EE", color: "#0F6E56" }}
              >
                Sequential
              </span>
            </h2>
          </div>

          <p className="text-[14px] text-muted-foreground leading-relaxed mb-5">
            RNNs process sequential data by maintaining a hidden state that carries memory across time
            steps. The output is <Highlight>backpropagated as feedback</Highlight>, enabling temporal
            pattern recognition. Widely paired with LSTM for NLP applications like Siri.
          </p>

          <RNNUnrolledDiagram />

          {/* Formula */}
          <div
            className="border-l-2 border-border/60 pl-5 py-3 mb-5 font-mono text-[13px]"
            aria-label="RNN recurrence formula"
          >
            h<sub>t</sub> = tanh(W<sub>x</sub> · x<sub>t</sub> + W<sub>h</sub> · h<sub>t-1</sub> + b)
            <div className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
              x_t: input at t &nbsp;·&nbsp; h_t: hidden state at t &nbsp;·&nbsp; W_x, W_h: weight matrices &nbsp;·&nbsp; b: bias
            </div>
          </div>

          <SectionDivider label="variants" />
          <div className="grid grid-cols-2 gap-2 mb-5">
            {([
              { title: "LSTM",              desc: "Input, forget & output gates. Cell state stores long-term memory. Mitigates vanishing/exploding gradients." },
              { title: "GRU",              desc: "Reset & update gates only. Fewer parameters than LSTM. Faster training with similar performance." },
              { title: "Bidirectional RNN", desc: "Two directions: forward & backward. Captures both past and future context for richer representations." },
              { title: "Deep RNN",          desc: "Multiple stacked recurrent layers. Learns more abstract temporal hierarchies from sequences." },
            ] as Variant[]).map((v) => (
              <VariantCard key={v.title} {...v} />
            ))}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-background border border-border rounded-xl p-4">
              <CardTitle>Applications</CardTitle>
              <BulletList items={["Natural language processing", "Speech recognition", "Time-series prediction", "Sequence generation"]} />
            </div>
            <div className="bg-background border border-border rounded-xl p-4">
              <CardTitle>Challenges</CardTitle>
              <BulletList items={["Vanishing / exploding gradients", "Slow sequential processing", "Limited memory on long sequences"]} />
            </div>
          </div>
        </section>

        {/* ── 4. Autoencoders ────────────────────────────────────────────────── */}
        <section
          ref={setRef("sec-ae")}
          id="sec-ae"
          className="mb-14 pb-14 border-b border-border scroll-mt-6"
          aria-labelledby="ae-title"
        >
          <div className="flex items-start gap-4 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground border border-border rounded-md px-2 py-1 mt-1 shrink-0">
              04
            </span>
            <h2 id="ae-title" className="text-3xl font-serif font-normal leading-tight">
              Autoencoders
              <span
                className="inline-block font-mono text-[10px] px-2 py-0.5 rounded-full ml-3 align-middle"
                style={{ background: "#FAEEDA", color: "#854F0B" }}
              >
                Unsupervised
              </span>
            </h2>
          </div>

          <p className="text-[14px] text-muted-foreground leading-relaxed mb-5">
            Autoencoders learn efficient compressed representations through an encoder–bottleneck–decoder
            structure. Primarily used for <Highlight>lossless compression</Highlight> of text, images,
            and video, and for feature learning without labeled data.
          </p>

          <AutoencoderDiagram />

          <SectionDivider label="variants" />
          <div className="grid grid-cols-2 gap-2">
            {([
              { title: "Vanilla Autoencoder", desc: "Basic encoder-decoder. Undercomplete latent space. Used for dimensionality reduction." },
              { title: "VAE",                 desc: "Probabilistic encoder. Adds KL divergence loss. Can generate new samples from latent space." },
              { title: "Denoising AE",        desc: "Input is deliberately corrupted. Must reconstruct clean output. Learns noise-invariant features." },
              { title: "Sparse AE",           desc: "Penalizes active neurons. Only a small subset fires at once. More efficient representations." },
            ] as Variant[]).map((v) => (
              <VariantCard key={v.title} {...v} />
            ))}
          </div>
        </section>

        {/* ── 5. GAN ─────────────────────────────────────────────────────────── */}
        <section
          ref={setRef("sec-gan")}
          id="sec-gan"
          className="mb-14 pb-14 border-b border-border scroll-mt-6"
          aria-labelledby="gan-title"
        >
          <div className="flex items-start gap-4 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground border border-border rounded-md px-2 py-1 mt-1 shrink-0">
              05
            </span>
            <h2 id="gan-title" className="text-3xl font-serif font-normal leading-tight">
              Generative Adversarial Networks
              <span
                className="inline-block font-mono text-[10px] px-2 py-0.5 rounded-full ml-3 align-middle"
                style={{ background: "#FAECE7", color: "#993C1D" }}
              >
                Generative
              </span>
            </h2>
          </div>

          <p className="text-[14px] text-muted-foreground leading-relaxed mb-5">
            GANs employ two competing networks in an adversarial game: a <Highlight>Generator</Highlight>{" "}
            that creates synthetic data and a <Highlight>Discriminator</Highlight> that distinguishes real
            from fake. Used to generate images, music, and stories.
          </p>

          {/* Formula */}
          <div
            className="border-l-2 border-border/60 pl-5 py-3 mb-5 font-mono text-[13px]"
            aria-label="GAN minimax objective"
          >
            min<sub>G</sub> max<sub>D</sub> V(D,G) = 𝔼<sub>x~p_data</sub>[log D(x)] + 𝔼<sub>z~p_z</sub>[log(1 − D(G(z)))]
            <div className="mt-2 text-[11px] text-muted-foreground leading-relaxed">
              G: generator &nbsp;·&nbsp; D: discriminator &nbsp;·&nbsp; z: random noise &nbsp;·&nbsp; Nash equilibrium = convergence
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {([
              { title: "DCGAN",          desc: "Uses CNNs in both networks. Better training stability and higher resolution outputs." },
              { title: "CycleGAN",       desc: "Domain translation without paired examples. Cycle consistency enforces coherent mapping." },
              { title: "StyleGAN",       desc: "Separates content from style. Progressive training. State-of-the-art image generation quality." },
              { title: "Conditional GAN",desc: "Generates samples conditioned on class labels. Enables text-to-image and image translation." },
            ] as Variant[]).map((v) => (
              <VariantCard key={v.title} {...v} />
            ))}
          </div>
        </section>

        {/* ── Comparison Table ────────────────────────────────────────────────── */}
        <section className="mb-14 pb-14 border-b border-border">
          <div className="flex items-center gap-3 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground tracking-wide">comparison</span>
            <div className="flex-1 h-px bg-border" />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]" aria-label="Neural network type comparison">
              <thead>
                <tr>
                  {["Type", "Data", "Memory", "Supervised", "Primary use"].map((h) => (
                    <th
                      key={h}
                      className="text-left font-mono text-[11px] text-muted-foreground border-b border-border pb-2 px-3"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { type: "MLP",         data: "Tabular",      mem: false, sup: true,  use: "Classification, regression" },
                  { type: "CNN",         data: "Grid (images)",mem: false, sup: true,  use: "Computer vision" },
                  { type: "RNN",         data: "Sequential",   mem: true,  sup: true,  use: "NLP, time series" },
                  { type: "Autoencoder", data: "Any",          mem: false, sup: false, use: "Dimensionality reduction" },
                  { type: "GAN",         data: "Any",          mem: false, sup: false, use: "Synthetic data generation" },
                ].map((row) => (
                  <tr key={row.type} className="hover:bg-muted/30 transition-colors">
                    <td className="px-3 py-2 font-medium border-b border-border">{row.type}</td>
                    <td className="px-3 py-2 text-muted-foreground border-b border-border">{row.data}</td>
                    <td className="px-3 py-2 border-b border-border">
                      <span
                        className="font-mono text-[10px] px-2 py-0.5 rounded-full"
                        style={row.mem
                          ? { background: "#EAF3DE", color: "#3B6D11" }
                          : { background: "#F1EFE8", color: "#5F5E5A" }
                        }
                      >
                        {row.mem ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-3 py-2 border-b border-border">
                      <span
                        className="font-mono text-[10px] px-2 py-0.5 rounded-full"
                        style={row.sup
                          ? { background: "#EAF3DE", color: "#3B6D11" }
                          : { background: "#F1EFE8", color: "#5F5E5A" }
                        }
                      >
                        {row.sup ? "Yes" : "No*"}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-muted-foreground border-b border-border">{row.use}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="font-mono text-[11px] text-muted-foreground mt-3">
            * Self-supervised — creates its own supervision signal
          </p>
        </section>

        {/* ── Timeline ────────────────────────────────────────────────────────── */}
        <section className="mb-14 pb-14 border-b border-border">
          <div className="flex items-center gap-3 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground tracking-wide">evolution timeline</span>
            <div className="flex-1 h-px bg-border" />
          </div>
          <div
            className="flex border border-border rounded-xl overflow-hidden mb-5"
            aria-label="Timeline of neural network architectures"
          >
            {[
              { year: "1950s",    label: "Perceptron" },
              { year: "1980–90s", label: "MLP + Backprop" },
              { year: "1990s",    label: "CNN & RNN" },
              { year: "2012–15",  label: "Deep CNN era" },
              { year: "2017+",    label: "Transformers" },
            ].map((item, i, arr) => (
              <div
                key={item.year}
                className={`flex-1 p-3 bg-background transition-all hover:bg-muted/20 ${i < arr.length - 1 ? "border-r border-border" : ""}`}
              >
                <p className="font-mono text-[10px] text-muted-foreground mb-1">{item.year}</p>
                <p className="text-[12px] font-medium leading-snug">{item.label}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── References ────────────────────────────────────────────────────────── */}
        <section className="mt-14 pt-6">
          <div className="flex items-center gap-3 mb-5">
            <span className="font-mono text-[11px] text-muted-foreground tracking-wide">references & further reading</span>
            <div className="flex-1 h-px bg-border" />
          </div>
          <ul className="space-y-2 text-[13px] text-muted-foreground">
            <li>• Goodfellow, I., Bengio, Y., & Courville, A. (2016). <em>Deep Learning</em>. MIT Press.</li>
            <li>• LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. <em>Nature</em>, 521(7553), 436-444.</li>
            <li>• Schmidhuber, J. (2015). Deep learning in neural networks: An overview. <em>Neural Networks</em>, 61, 85-117.</li>
            <li>• <Highlight>https://playground.tensorflow.org</Highlight> — Interactive neural network visualization</li>
          </ul>
        </section>

        {/* Print styles */}
        <style jsx global>{`
          @media print {
            nav, button, .group {
              display: none !important;
            }
            .bg-muted, .bg-background, [class*="bg-"] {
              background-color: transparent !important;
              border: 1px solid #ccc !important;
            }
            body {
              background: white !important;
              color: black !important;
            }
          }
        `}</style>

      </main>
    </TopicContent>
  );
}