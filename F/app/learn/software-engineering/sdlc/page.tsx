import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Users, Calendar, FileText, Code, Rocket, GitBranch, MessageSquare, Target, Settings, Shield, Zap, Layers, BarChart } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";

export default function SoftwareLifeCyclePage() {
  const result = getSubtopicBySlug("software-engineering", "sdlc");
  if (!result) return null;

  const { topic, subtopic } = result;

  // SDLC Phases
  const sdlcPhases = [
    {
      phase: "Communication",
      icon: <MessageSquare className="h-6 w-6" />,
      description: "Initial project initiation and requirement gathering",
      activities: [
        "Project initiation meeting",
        "Stakeholder identification",
        "Feasibility study",
        "Scope definition"
      ],
      deliverables: ["Project charter", "Initial requirement document", "Feasibility report"],
      duration: "1-4 weeks",
      keyRole: "Project Manager / Business Analyst",
    },
    {
      phase: "Planning",
      icon: <Calendar className="h-6 w-6" />,
      description: "Detailed project planning and resource allocation",
      activities: [
        "Project planning",
        "Resource allocation",
        "Risk assessment",
        "Cost estimation",
        "Schedule creation"
      ],
      deliverables: ["Project plan", "Risk management plan", "Budget estimate", "Schedule"],
      duration: "2-6 weeks",
      keyRole: "Project Manager",
    },
    {
      phase: "Modeling/Analysis",
      icon: <GitBranch className="h-6 w-6" />,
      description: "System design and architecture planning",
      activities: [
        "Requirement analysis",
        "System design",
        "Architecture planning",
        "UI/UX design",
        "Database design"
      ],
      deliverables: ["SRS document", "System design document", "Architecture diagram", "Wireframes"],
      duration: "3-8 weeks",
      keyRole: "System Architect / Designer",
    },
    {
      phase: "Construction/Implementation",
      icon: <Code className="h-6 w-6" />,
      description: "Actual coding and development of the software",
      activities: [
        "Coding",
        "Unit testing",
        "Code review",
        "Integration",
        "Version control"
      ],
      deliverables: ["Source code", "Unit test cases", "Technical documentation", "Build artifacts"],
      duration: "8-20 weeks",
      keyRole: "Software Developer",
    },
    {
      phase: "Testing",
      icon: <Shield className="h-6 w-6" />,
      description: "Verification and validation of the software",
      activities: [
        "Test planning",
        "Test case development",
        "Test execution",
        "Bug reporting",
        "Performance testing"
      ],
      deliverables: ["Test plans", "Test cases", "Bug reports", "Test summary report"],
      duration: "4-8 weeks",
      keyRole: "QA Engineer / Tester",
    },
    {
      phase: "Deployment",
      icon: <Rocket className="h-6 w-6" />,
      description: "Release and installation of software in production",
      activities: [
        "Deployment planning",
        "Environment setup",
        "Data migration",
        "User training",
        "Go-live"
      ],
      deliverables: ["Deployed system", "User manuals", "Training materials", "Release notes"],
      duration: "1-2 weeks",
      keyRole: "DevOps Engineer",
    },
    {
      phase: "Maintenance",
      icon: <Settings className="h-6 w-6" />,
      description: "Post-deployment support and enhancements",
      activities: [
        "Bug fixing",
        "Performance monitoring",
        "Updates and patches",
        "Enhancements",
        "User support"
      ],
      deliverables: ["Maintenance logs", "Patch releases", "Performance reports", "Support tickets"],
      duration: "Ongoing",
      keyRole: "Support Engineer",
    },
  ];

  // Requirement Types
  const requirementTypes = [
    {
      type: "Functional Requirements",
      description: "What the system should do - specific behaviors and functions",
      icon: <Target className="h-5 w-5" />,
      examples: [
        "User authentication and authorization",
        "Data processing and calculations",
        "Reporting and analytics",
        "Integration with other systems",
        "Business rules implementation"
      ],
      format: "User stories, use cases, functional specifications",
      verification: "Unit testing, integration testing",
    },
    {
      type: "Non-Functional Requirements",
      description: "How the system should perform - quality attributes",
      icon: <Zap className="h-5 w-5" />,
      examples: [
        "Performance (response time, throughput)",
        "Security (authentication, encryption)",
        "Reliability (uptime, error rates)",
        "Usability (user interface, accessibility)",
        "Scalability (load handling, growth)",
        "Maintainability (code quality, documentation)"
      ],
      format: "Technical specifications, SLA documents",
      verification: "Performance testing, security testing",
    },
  ];

  // Process Models
  const processModels = [
    {
      name: "Waterfall Model",
      icon: <Layers className="h-6 w-6" />,
      description: "Linear sequential approach where each phase must be completed before the next begins",
      phases: ["Requirements", "Design", "Implementation", "Verification", "Maintenance"],
      advantages: [
        "Simple and easy to understand",
        "Clear milestones and deliverables",
        "Good for small, well-defined projects",
        "Easy to manage due to rigidity"
      ],
      disadvantages: [
        "Inflexible to changes",
        "No working software until late",
        "High risk and uncertainty",
        "Not suitable for complex projects"
      ],
      whenToUse: "Projects with clear, stable requirements",
      example: "Government systems, Banking applications",
    },
    {
      name: "Agile Model",
      icon: <Zap className="h-6 w-6" />,
      description: "Iterative approach with focus on customer collaboration and responding to change",
      phases: ["Sprint Planning", "Development", "Review", "Retrospective"],
      advantages: [
        "Flexible to changing requirements",
        "Early and continuous delivery",
        "Customer collaboration",
        "Reduced risk through iterations"
      ],
      disadvantages: [
        "Less predictable",
        "Requires experienced team",
        "Documentation can be minimal",
        "Difficult to scale"
      ],
      whenToUse: "Projects with evolving requirements",
      example: "Startups, Web applications, Mobile apps",
    },
    {
      name: "Incremental Model",
      icon: <BarChart className="h-6 w-6" />,
      description: "Multiple development cycles where system is built in increments",
      phases: ["Requirements", "Design", "Implementation", "Testing (per increment)"],
      advantages: [
        "Early delivery of partial functionality",
        "Easier to test and debug",
        "Flexible to changes",
        "Lower initial delivery cost"
      ],
      disadvantages: [
        "Need good planning",
        "Total cost may be higher",
        "Requires clear interfaces",
        "System architecture must be robust"
      ],
      whenToUse: "Large projects that can be modularized",
      example: "ERP systems, Large enterprise applications",
    },
    {
      name: "Spiral Model",
      icon: <GitBranch className="h-6 w-6" />,
      description: "Risk-driven iterative model combining waterfall and prototyping",
      phases: ["Planning", "Risk Analysis", "Engineering", "Evaluation"],
      advantages: [
        "Good risk management",
        "Flexible to changes",
        "Early prototyping",
        "Customer feedback incorporation"
      ],
      disadvantages: [
        "Complex to manage",
        "Expensive",
        "Not suitable for small projects",
        "Requires risk assessment expertise"
      ],
      whenToUse: "Large, complex, high-risk projects",
      example: "Military systems, Space missions",
    },
    {
      name: "V-Model",
      icon: <FileText className="h-6 w-6" />,
      description: "Extension of waterfall with verification and validation at each stage",
      phases: ["Requirements", "Design", "Implementation", "Testing", "Deployment"],
      advantages: [
        "High quality assurance",
        "Early defect detection",
        "Disciplined approach",
        "Good for critical systems"
      ],
      disadvantages: [
        "Rigid and inflexible",
        "No early prototypes",
        "Expensive",
        "Not good for complex projects"
      ],
      whenToUse: "Mission-critical systems",
      example: "Medical software, Aviation systems",
    },
    {
      name: "RAD Model",
      icon: <Rocket className="h-6 w-6" />,
      description: "Rapid Application Development using prototypes and iterative design",
      phases: ["Requirements Planning", "User Design", "Construction", "Cutover"],
      advantages: [
        "Reduced development time",
        "Increased customer satisfaction",
        "Early integration",
        "Reusability of components"
      ],
      disadvantages: [
        "Requires strong team skills",
        "Only for modularizable systems",
        "High dependency on modeling skills",
        "Not suitable for high-risk projects"
      ],
      whenToUse: "Systems that can be modularized",
      example: "E-commerce sites, Business applications",
    },
  ];

  // Scrum Framework Details
  const scrumFramework = {
    roles: [
      {
        role: "Product Owner",
        description: "Represents stakeholders and business, manages product backlog",
        responsibilities: [
          "Define product vision",
          "Prioritize backlog items",
          "Accept or reject work results",
          "Communicate with stakeholders"
        ],
        skills: ["Business knowledge", "Decision making", "Communication"],
      },
      {
        role: "Scrum Master",
        description: "Facilitates Scrum process, removes impediments, ensures team follows practices",
        responsibilities: [
          "Facilitate Scrum events",
          "Remove obstacles",
          "Coach team on Scrum",
          "Protect team from distractions"
        ],
        skills: ["Facilitation", "Coaching", "Conflict resolution"],
      },
      {
        role: "Development Team",
        description: "Cross-functional team that delivers product increments",
        responsibilities: [
          "Plan sprint work",
          "Design and develop features",
          "Test and integrate",
          "Maintain quality standards"
        ],
        skills: ["Technical expertise", "Collaboration", "Problem solving"],
      },
    ],
    artifacts: [
      {
        name: "Product Backlog",
        description: "Ordered list of everything needed in the product",
        content: "User stories, features, enhancements, bug fixes",
        management: "Owned by Product Owner, regularly refined",
      },
      {
        name: "Sprint Backlog",
        description: "Set of Product Backlog items selected for Sprint plus plan",
        content: "Tasks, estimated effort, assignment, progress",
        management: "Owned by Development Team, updated daily",
      },
      {
        name: "Increment",
        description: "Sum of all Product Backlog items completed during Sprint",
        content: "Working software, documentation, tests",
        management: "Must be in usable condition, potentially shippable",
      },
    ],
    events: [
      {
        name: "Sprint Planning",
        duration: "4*8h-week",
        purpose: "Plan work for upcoming sprint",
        participants: "Scrum Team",
        output: "Sprint backlog, Sprint goal",
      },
      {
        name: "Daily Scrum",
        duration: "15 minutes",
        purpose: "Inspect progress toward Sprint Goal",
        participants: "Development Team, Scrum Master",
        output: "Updated plan for next 24 hours",
      },
      {
        name: "Sprint Review",
        duration: "4*4h-week ",
        purpose: "Inspect increment and adapt Product Backlog",
        participants: "Scrum Team, stakeholders",
        output: "Revised Product Backlog",
      },
      {
        name: "Sprint Retrospective",
        //duration: "4*3h W ",
        purpose: "Plan improvements for next Sprint",
        participants: "Scrum Team",
        output: "Improvement plan",
      },
    ],
  };


  const tools = [
    {
      category: "Project Management",
      tools: [
        {
          name: "Jira",
          description: "Agile project management and issue tracking",
          features: ["Scrum/Kanban boards", "Backlog management", "Sprint planning", "Reports"],
          pricing: "Free for up to 10 users",
          bestFor: "Software development teams",
        },
        {
          name: "ClickUp",
          description: "All-in-one productivity platform",
          features: ["Tasks, Docs, Goals", "Time tracking", "Mind maps", "Custom views"],
          pricing: "Free plan available",
          bestFor: "Teams needing flexibility",
        },
        {
          name: "Trello",
          description: "Visual collaboration tool",
          features: ["Kanban boards", "Cards and lists", "Power-ups", "Automation"],
          pricing: "Free basic plan",
          bestFor: "Simple project tracking",
        },
        {
          name: "Asana",
          description: "Work management platform",
          features: ["Task management", "Timeline view", "Workflow automation", "Portfolios"],
          pricing: "Free for up to 15 users",
          bestFor: "Cross-functional teams",
        },
      ],
    },
    {
      category: "Requirements Management",
      tools: [
        {
          name: "Confluence",
          description: "Team collaboration and documentation",
          features: ["Requirements docs", "Meeting notes", "Knowledge base", "Integration with Jira"],
          pricing: "Free for up to 10 users",
          bestFor: "Documentation and collaboration",
        },
        {
          name: "Jama Connect",
          description: "Requirements management and traceability",
          features: ["Requirements tracking", "Traceability matrix", "Compliance", "Reviews"],
          pricing: "Enterprise pricing",
          bestFor: "Complex, regulated projects",
        },
        {
          name: "IBM DOORS",
          description: "Enterprise requirements management",
          features: ["Requirements capture", "Change management", "Impact analysis", "Reporting"],
          pricing: "Enterprise pricing",
          bestFor: "Large-scale enterprise projects",
        },
      ],
    },
    {
      category: "Modeling & Design",
      tools: [
        {
          name: "Lucidchart",
          description: "Diagramming and visualization",
          features: ["UML diagrams", "Flowcharts", "Wireframes", "Collaboration"],
          pricing: "Free basic plan",
          bestFor: "Visual design and modeling",
        },
        {
          name: "Draw.io",
          description: "Free diagramming tool",
          features: ["Multiple diagram types", "Export options", "Integration", "Open source"],
          pricing: "Completely free",
          bestFor: "Budget-conscious teams",
        },
        {
          name: "Figma",
          description: "Interface design and prototyping",
          features: ["UI/UX design", "Prototyping", "Design systems", "Collaboration"],
          pricing: "Free for individuals",
          bestFor: "UI/UX designers",
        },
      ],
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which SDLC phase involves stakeholder identification and feasibility study?",
      options: ["Planning", "Communication", "Modeling", "Construction"],
      correctAnswer: 1,
      explanation: "The Communication phase includes project initiation, stakeholder identification, and feasibility study.",
    },
    {
      id: 2,
      question: "What is the main difference between functional and non-functional requirements?",
      options: [
        "Functional are about what system does, non-functional are about how",
        "Functional are technical, non-functional are business",
        "Functional are optional, non-functional are mandatory",
        "There is no difference"
      ],
      correctAnswer: 0,
      explanation: "Functional requirements specify what the system should do, while non-functional requirements specify how it should perform.",
    },
    {
      id: 3,
      question: "Which process model is best for projects with clear, stable requirements?",
      options: ["Waterfall", "Agile", "Spiral", "RAD"],
      correctAnswer: 0,
      explanation: "Waterfall model works best for projects with clear, stable requirements due to its sequential nature.",
    },
    {
      id: 4,
      question: "Who is responsible for managing the product backlog in Scrum?",
      options: ["Scrum Master", "Product Owner", "Development Team", "Project Manager"],
      correctAnswer: 1,
      explanation: "The Product Owner is responsible for managing and prioritizing the product backlog.",
    },
    {
      id: 5,
      question: "Which tool is specifically designed for Agile project management?",
      options: ["Jira", "Microsoft Project", "Excel", "Google Sheets"],
      correctAnswer: 0,
      explanation: "Jira is specifically designed for Agile project management with features for Scrum and Kanban.",
    },
    {
      id: 6,
      question: "What is the purpose of the Sprint Retrospective in Scrum?",
      options: [
        "To demonstrate completed work",
        "To plan improvements for next sprint",
        "To assign new tasks",
        "To report to management"
      ],
      correctAnswer: 1,
      explanation: "The Sprint Retrospective is for the team to reflect and plan improvements for the next sprint.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Software Development Life Cycle (SDLC)</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">SDLC</strong> is a structured process for building software that ensures 
            quality, meets requirements, and is delivered on time and within budget. 
            It encompasses all activities from initial concept to maintenance and support.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Why SDLC Matters</h4>
                <p className="text-sm text-muted-foreground">
                  Following a proper SDLC is like having a blueprint for construction. It provides 
                  structure, reduces risks, ensures quality, and helps teams deliver software that 
                  actually meets user needs. Without it, projects often fail due to scope creep, 
                  poor quality, or missed deadlines.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* SDLC Phases */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">SDLC Phases</h2>
          <p className="text-muted-foreground mb-6">
            The software development life cycle typically consists of seven main phases, 
            each with specific activities and deliverables.
          </p>
          
          <div className="space-y-6">
            {sdlcPhases.map((phase, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex flex-col gap-6">
                    {/* Phase Header - Full Width on Mobile */}
                    <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                      <div className="flex items-center gap-3">
                        <div className="p-3 bg-primary/10 rounded-lg shrink-0">
                          {phase.icon}
                        </div>
                        <div className="min-w-0">
                          <h3 className="font-bold text-lg sm:text-xl text-foreground">{phase.phase}</h3>
                          <div className="flex items-center gap-2 mt-1">
                            <Clock className="h-4 w-4 text-muted-foreground shrink-0" />
                            <span className="text-sm text-muted-foreground">{phase.duration}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="p-3 bg-muted rounded-lg sm:ml-auto sm:max-w-xs">
                        <p className="text-sm font-medium text-foreground mb-1">Key Role:</p>
                        <p className="text-sm text-muted-foreground break-words">{phase.keyRole}</p>
                      </div>
                    </div>
                    
                    {/* Phase Details */}
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-semibold text-foreground mb-2">Description</h4>
                        <p className="text-muted-foreground text-sm sm:text-base">{phase.description}</p>
                      </div>
                      
                      <div className="grid sm:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Activities</h4>
                          <ul className="space-y-1.5">
                            {phase.activities.map((activity, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm">
                                <div className="h-2 w-2 rounded-full bg-primary mt-2 shrink-0" />
                                <span className="text-muted-foreground break-words">{activity}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-foreground mb-2">Deliverables</h4>
                          <ul className="space-y-1.5">
                            {phase.deliverables.map((deliverable, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm">
                                <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                                <span className="text-muted-foreground break-words">{deliverable}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Requirements Types */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Types of Requirements</h2>
          <p className="text-muted-foreground mb-6">
            Requirements define what needs to be built. They are categorized into 
            functional and non-functional requirements.
          </p>
          
          <div className="grid lg:grid-cols-2 gap-6">
            {requirementTypes.map((reqType, idx) => (
              <Card key={idx} className={`${idx === 0 ? 'border-blue-200' : 'border-purple-200'}`}>
                <CardContent className="p-6">
                  <div className="flex flex-col sm:flex-row sm:items-start gap-3 mb-4">
                    <div className={`p-3 rounded-lg shrink-0 ${idx === 0 ? 'bg-blue-100' : 'bg-purple-100'}`}>
                      {reqType.icon}
                    </div>
                    <div className="min-w-0">
                      <h3 className="font-bold text-lg sm:text-xl text-foreground break-words">{reqType.type}</h3>
                      <p className="text-sm text-muted-foreground mt-1">{reqType.description}</p>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">Examples</h4>
                      <div className="space-y-2">
                        {reqType.examples.map((example, i) => (
                          <div key={i} className="flex items-start gap-2 text-sm p-2 bg-muted rounded">
                            <div className={`h-2 w-2 rounded-full mt-2 shrink-0 ${idx === 0 ? 'bg-blue-500' : 'bg-purple-500'}`} />
                            <span className="text-muted-foreground break-words">{example}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div className="grid sm:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold text-foreground mb-2">Format</h4>
                        <p className="text-sm text-muted-foreground break-words">{reqType.format}</p>
                      </div>
                      <div>
                        <h4 className="font-semibold text-foreground mb-2">Verification</h4>
                        <p className="text-sm text-muted-foreground break-words">{reqType.verification}</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          
          {/* Requirements Example */}
          <div className="mt-6">
            <Card>
              <CardContent className="p-6">
                <h4 className="font-semibold mb-4 text-foreground">Requirements Specification Example</h4>
                <div className="grid lg:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h5 className="font-medium text-blue-700 dark:text-blue-400 mb-1">Functional Requirement</h5>
                      <p className="text-sm text-blue-600 dark:text-blue-300 break-words">
                        <strong>User Story:</strong> As a registered user, I want to reset my password 
                        so that I can regain access to my account if I forget it.
                      </p>
                      <div className="mt-2 text-xs text-blue-500">
                        <strong>Acceptance Criteria:</strong> 
                        <ul className="mt-1 space-y-1 ml-2">
                          <li className="break-words">• Password reset link sent to registered email</li>
                          <li className="break-words">• Link expires after 24 hours</li>
                          <li className="break-words">• Password must meet security requirements</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <h5 className="font-medium text-purple-700 dark:text-purple-400 mb-1">Non-Functional Requirement</h5>
                      <p className="text-sm text-purple-600 dark:text-purple-300 break-words">
                        <strong>Performance Requirement:</strong> The password reset page must 
                        load within 2 seconds for 95% of users under normal load conditions.
                      </p>
                      <div className="mt-2 text-xs text-purple-500">
                        <strong>Quality Attributes:</strong> 
                        <ul className="mt-1 space-y-1 ml-2">
                          <li className="break-words">• Response time: {"<"} 2 seconds</li>
                          <li className="break-words">• Availability: 99.9% uptime</li>
                          <li className="break-words">• Security: HTTPS encryption required</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Process Models */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Process Models</h2>
          <p className="text-muted-foreground mb-6">
            Different approaches to organizing and executing the software development process.
          </p>
          
          <Tabs defaultValue="waterfall" className="w-full">
            <div className="overflow-x-auto -mx-2 px-2">
              <TabsList className="inline-flex w-auto min-w-full">
                {processModels.map((model, idx) => (
                  <TabsTrigger 
                    key={idx} 
                    value={model.name.toLowerCase().replace(' ', '-')}
                    className="text-xs sm:text-sm whitespace-nowrap"
                  >
                    {model.name}
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>
            
            {processModels.map((model, idx) => (
              <TabsContent key={idx} value={model.name.toLowerCase().replace(' ', '-')}>
                <Card>
                  <CardContent className="p-6">
                    <div className="flex flex-col sm:flex-row sm:items-start gap-3 mb-6">
                      <div className="p-3 bg-primary/10 rounded-lg shrink-0">
                        {model.icon}
                      </div>
                      <div className="min-w-0">
                        <h3 className="font-bold text-xl sm:text-2xl text-foreground break-words">{model.name}</h3>
                        <p className="text-muted-foreground mt-1 text-sm sm:text-base">{model.description}</p>
                      </div>
                    </div>
                    
                    <div className="grid lg:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold text-foreground mb-3">Phases</h4>
                        <div className="space-y-2">
                          {model.phases.map((phase, i) => (
                            <div key={i} className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                              <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                                <span className="font-bold text-primary">{i + 1}</span>
                              </div>
                              <span className="font-medium text-sm sm:text-base break-words">{phase}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-semibold text-foreground mb-3">When to Use</h4>
                          <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                            <p className="text-green-700 dark:text-green-400 text-sm sm:text-base break-words">{model.whenToUse}</p>
                            <p className="text-sm text-green-600 dark:text-green-300 mt-2 break-words">
                              <strong>Example:</strong> {model.example}
                            </p>
                          </div>
                        </div>
                        
                        <div className="grid sm:grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Advantages</h4>
                            <ul className="space-y-1.5">
                              {model.advantages.map((adv, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                                  <span className="text-muted-foreground break-words">{adv}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold text-foreground mb-2">Disadvantages</h4>
                            <ul className="space-y-1.5">
                              {model.disadvantages.map((disadv, i) => (
                                <li key={i} className="flex items-start gap-2 text-sm">
                                  <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
                                  <span className="text-muted-foreground break-words">{disadv}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </section>

        {/* Scrum Framework */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Scrum Framework</h2>
          <p className="text-muted-foreground mb-6">
            Scrum is an Agile framework for managing complex software development projects.
          </p>
          
          <div className="space-y-8">
            {/* Scrum Roles */}
            <div>
              <h3 className="text-xl font-bold mb-4 text-foreground">Scrum Roles</h3>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {scrumFramework.roles.map((role, idx) => (
                  <Card key={idx}>
                    <CardContent className="p-6">
                      <h4 className="font-bold text-lg text-foreground mb-2 break-words">{role.role}</h4>
                      <p className="text-sm text-muted-foreground mb-4">{role.description}</p>
                      
                      <div className="space-y-3">
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-2">Responsibilities</h5>
                          <ul className="space-y-1.5">
                            {role.responsibilities.map((resp, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm">
                                <div className="h-2 w-2 rounded-full bg-primary mt-2 shrink-0" />
                                <span className="text-muted-foreground break-words">{resp}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-2">Key Skills</h5>
                          <div className="flex flex-wrap gap-2">
                            {role.skills.map((skill, i) => (
                              <Badge key={i} variant="outline" className="text-xs">{skill}</Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
            
            {/* Scrum Artifacts */}
            <div>
              <h3 className="text-xl font-bold mb-4 text-foreground">Scrum Artifacts</h3>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {scrumFramework.artifacts.map((artifact, idx) => (
                  <Card key={idx}>
                    <CardContent className="p-6">
                      <h4 className="font-bold text-lg text-foreground mb-2 break-words">{artifact.name}</h4>
                      <p className="text-sm text-muted-foreground mb-4">{artifact.description}</p>
                      
                      <div className="space-y-3">
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-1">Content</h5>
                          <p className="text-sm text-muted-foreground break-words">{artifact.content}</p>
                        </div>
                        
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-1">Management</h5>
                          <p className="text-sm text-muted-foreground break-words">{artifact.management}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
            
            {/* Scrum Events */}
            <div>
              <h3 className="text-xl font-bold mb-4 text-foreground">Scrum Events</h3>
              <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
                {scrumFramework.events.map((event, idx) => (
                  <Card key={idx}>
                    <CardContent className="p-6">
                      <div className="flex flex-col sm:flex-row items-start justify-between gap-2 mb-3">
                        <h4 className="font-bold text-base sm:text-lg text-foreground break-words">{event.name}</h4>
                        <Badge variant="outline" className="text-xs whitespace-nowrap shrink-0">{event.duration}</Badge>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-1">Purpose</h5>
                          <p className="text-sm text-muted-foreground break-words">{event.purpose}</p>
                        </div>
                        
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-1">Participants</h5>
                          <p className="text-sm text-muted-foreground break-words">{event.participants}</p>
                        </div>
                        
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-1">Output</h5>
                          <p className="text-sm text-muted-foreground break-words">{event.output}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Tools */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Tools & Software</h2>
          
          {tools.map((category, catIdx) => (
            <div key={catIdx} className="mb-8">
              <h3 className="text-xl font-bold mb-4 text-foreground">{category.category}</h3>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {category.tools.map((tool, toolIdx) => (
                  <Card key={toolIdx} className="hover:shadow-lg transition-shadow">
                    <CardContent className="p-6">
                      <h4 className="font-bold text-lg text-foreground mb-2 break-words">{tool.name}</h4>
                      <p className="text-sm text-muted-foreground mb-4">{tool.description}</p>
                      
                      <div className="space-y-3">
                        <div>
                          <h5 className="font-semibold text-sm text-foreground mb-2">Key Features</h5>
                          <ul className="space-y-1.5">
                            {tool.features.map((feature, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm">
                                <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                                <span className="text-muted-foreground break-words">{feature}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="grid grid-cols-1 gap-3">
                          <div>
                            <h5 className="font-semibold text-sm text-foreground mb-1">Pricing</h5>
                            <p className="text-sm text-muted-foreground break-words">{tool.pricing}</p>
                          </div>
                          <div>
                            <h5 className="font-semibold text-sm text-foreground mb-1">Best For</h5>
                            <p className="text-sm text-muted-foreground break-words">{tool.bestFor}</p>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ))}
          
          {/* Jira vs ClickUp Comparison */}
          <Card className="mt-8">
            <CardContent className="p-6">
              <h3 className="text-xl font-bold mb-4 text-foreground">Jira vs ClickUp: Comparison</h3>
              <div className="overflow-x-auto -mx-6 px-6">
                <table className="w-full border-collapse min-w-[600px]">
                  <thead>
                    <tr className="bg-muted">
                      <th className="border border-border p-3 text-left text-foreground text-sm">Aspect</th>
                      <th className="border border-border p-3 text-left text-foreground text-sm">Jira</th>
                      <th className="border border-border p-3 text-left text-foreground text-sm">ClickUp</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground text-sm">Primary Focus</td>
                      <td className="border border-border p-3 text-sm">Software development & Agile</td>
                      <td className="border border-border p-3 text-sm">All-in-one productivity</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground text-sm">Agile Support</td>
                      <td className="border border-border p-3">
                        <Badge variant="default" className="text-xs mb-1">Excellent</Badge>
                        <p className="text-xs mt-1 text-muted-foreground">Native Scrum/Kanban, advanced reporting</p>
                      </td>
                      <td className="border border-border p-3">
                        <Badge variant="secondary" className="text-xs mb-1">Good</Badge>
                        <p className="text-xs mt-1 text-muted-foreground">Basic Agile features, flexible views</p>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground text-sm">Learning Curve</td>
                      <td className="border border-border p-3">
                        <Badge variant="destructive" className="text-xs mb-1">Steep</Badge>
                        <p className="text-xs mt-1 text-muted-foreground">Complex setup, many configurations</p>
                      </td>
                      <td className="border border-border p-3">
                        <Badge variant="default" className="text-xs mb-1">Gentle</Badge>
                        <p className="text-xs mt-1 text-muted-foreground">Intuitive interface, easy to start</p>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground text-sm">Customization</td>
                      <td className="border border-border p-3 text-sm">High (via add-ons)</td>
                      <td className="border border-border p-3 text-sm">Very High (built-in)</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground text-sm">Integration</td>
                      <td className="border border-border p-3 text-sm">Excellent (3000+ apps)</td>
                      <td className="border border-border p-3 text-sm">Good (1000+ apps)</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 font-medium text-foreground text-sm">Best For</td>
                      <td className="border border-border p-3 text-sm">Software teams needing deep Agile features</td>
                      <td className="border border-border p-3 text-sm">Teams wanting all-in-one flexibility</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common SDLC Mistakes</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div className="min-w-0">
                <h4 className="font-semibold text-foreground mb-1">Skipping Requirements Phase</h4>
                <p className="text-sm text-muted-foreground break-words">
                  Jumping straight to coding without proper requirements leads to 
                  scope creep, rework, and unsatisfied stakeholders.
                </p>
              </div>
            </div>
            
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div className="min-w-0">
                <h4 className="font-semibold text-foreground mb-1">Ignoring Non-Functional Requirements</h4>
                <p className="text-sm text-muted-foreground break-words">
                  Focusing only on what the system does (functional) while ignoring 
                  how it performs (non-functional) leads to poor user experience and system failures.
                </p>
              </div>
            </div>
            
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div className="min-w-0">
                <h4 className="font-semibold text-foreground mb-1">Wrong Process Model Selection</h4>
                <p className="text-sm text-muted-foreground break-words">
                  Using Waterfall for rapidly changing requirements or Agile for 
                  highly regulated, fixed-scope projects leads to project failure.
                </p>
              </div>
            </div>
            
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div className="min-w-0">
                <h4 className="font-semibold text-foreground mb-1">Poor Tool Selection</h4>
                <p className="text-sm text-muted-foreground break-words">
                  Choosing complex tools for simple projects or simple tools for 
                  complex projects reduces productivity and increases frustration.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Best Practices */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Best Practices</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Users className="h-5 w-5 text-primary shrink-0" />
                <h4 className="font-semibold text-foreground">Stakeholder Involvement</h4>
              </div>
              <p className="text-sm text-muted-foreground break-words">
                Involve stakeholders throughout the process. Regular communication 
                and feedback prevent misunderstandings and ensure the final product meets needs.
              </p>
            </div>
            
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="h-5 w-5 text-primary shrink-0" />
                <h4 className="font-semibold text-foreground">Clear Requirements</h4>
              </div>
              <p className="text-sm text-muted-foreground break-words">
                Document both functional and non-functional requirements clearly. 
                Use user stories, acceptance criteria, and measurable quality attributes.
              </p>
            </div>
            
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <GitBranch className="h-5 w-5 text-primary shrink-0" />
                <h4 className="font-semibold text-foreground">Right Process Model</h4>
              </div>
              <p className="text-sm text-muted-foreground break-words">
                Choose the process model based on project characteristics: 
                Waterfall for stable requirements, Agile for changing requirements, 
                Hybrid for mixed scenarios.
              </p>
            </div>
            
            <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Settings className="h-5 w-5 text-primary shrink-0" />
                <h4 className="font-semibold text-foreground">Appropriate Tools</h4>
              </div>
              <p className="text-sm text-muted-foreground break-words">
                Select tools that match your team size, process, and complexity. 
                Don't over-complicate with enterprise tools for small projects.
              </p>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your SDLC Knowledge</h2>
          <Quiz questions={quizQuestions} title="Software Life Cycle Quiz" />
        </section>

      
      
         
      </div>
    </TopicContent>
  );
}