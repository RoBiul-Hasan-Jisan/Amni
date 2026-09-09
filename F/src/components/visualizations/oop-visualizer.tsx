"use client";

import * as React from "react";
import { Play, Pause, RotateCcw, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Classes & Objects Visualizer
export function ClassObjectVisualizer() {
  const [instances, setInstances] = React.useState<
    { id: number; name: string; age: number; color: string }[]
  >([]);
  const [nextId, setNextId] = React.useState(1);
  const [showingMethod, setShowingMethod] = React.useState<number | null>(null);

  const colors = ["bg-blue-500", "bg-green-500", "bg-orange-500", "bg-pink-500", "bg-cyan-500"];
  const names = ["Alice", "Bob", "Charlie", "Diana", "Eve"];

  const createInstance = () => {
    if (instances.length >= 5) return;
    const newInstance = {
      id: nextId,
      name: names[instances.length % names.length],
      age: 20 + Math.floor(Math.random() * 30),
      color: colors[instances.length % colors.length],
    };
    setInstances([...instances, newInstance]);
    setNextId(nextId + 1);
  };

  const callMethod = (id: number) => {
    setShowingMethod(id);
    setTimeout(() => setShowingMethod(null), 1500);
  };

  const reset = () => {
    setInstances([]);
    setNextId(1);
    setShowingMethod(null);
  };

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-foreground">Class & Object Visualization</h3>
        <div className="flex gap-2">
          <Button size="sm" onClick={createInstance} disabled={instances.length >= 5}>
            <Play className="h-4 w-4 mr-1" /> New Instance
          </Button>
          <Button size="sm" variant="outline" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Class Blueprint */}
        <div className="p-4 bg-muted/50 rounded-lg border-2 border-dashed border-primary/50">
          <div className="text-center mb-3">
            <span className="px-3 py-1 bg-primary text-primary-foreground text-sm rounded-full">
              Class Blueprint
            </span>
          </div>
          <div className="bg-background rounded-lg p-4 font-mono text-sm">
            <div className="text-primary font-bold mb-2">class Person:</div>
            <div className="pl-4 space-y-1">
              <div className="text-muted-foreground"># Attributes</div>
              <div>name: <span className="text-blue-500">string</span></div>
              <div>age: <span className="text-green-500">int</span></div>
              <div className="text-muted-foreground mt-3"># Methods</div>
              <div className="text-orange-500">greet()</div>
              <div className="text-orange-500">celebrate_birthday()</div>
            </div>
          </div>
        </div>

        {/* Instances */}
        <div className="space-y-3">
          <div className="text-center mb-3">
            <span className="px-3 py-1 bg-secondary text-secondary-foreground text-sm rounded-full">
              Object Instances
            </span>
          </div>
          {instances.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              Click "New Instance" to create objects from the class
            </div>
          ) : (
            <div className="space-y-3">
              {instances.map((inst) => (
                <div
                  key={inst.id}
                  className={cn(
                    "p-3 rounded-lg border-2 transition-all",
                    showingMethod === inst.id
                      ? "border-primary bg-primary/10 scale-105"
                      : "border-border bg-background"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={cn("w-10 h-10 rounded-full flex items-center justify-center text-white font-bold", inst.color)}>
                        {inst.name[0]}
                      </div>
                      <div>
                        <div className="font-medium text-foreground">{inst.name}</div>
                        <div className="text-sm text-muted-foreground">Age: {inst.age}</div>
                      </div>
                    </div>
                    <Button size="sm" variant="ghost" onClick={() => callMethod(inst.id)}>
                      greet()
                    </Button>
                  </div>
                  {showingMethod === inst.id && (
                    <div className="mt-2 p-2 bg-primary/20 rounded text-sm text-primary animate-pulse">
                      "Hello, I'm {inst.name}!"
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 p-3 bg-muted rounded-lg text-sm text-muted-foreground">
        <strong className="text-foreground">Key Concept:</strong> A class is a blueprint that defines attributes and methods. 
        Objects are instances created from that blueprint, each with their own data.
      </div>
    </div>
  );
}

// Inheritance Visualizer
export function InheritanceVisualizer() {
  const [activeClass, setActiveClass] = React.useState<string | null>(null);
  const [showInherited, setShowInherited] = React.useState(false);

  const classes = {
    Animal: {
      attributes: ["name", "age"],
      methods: ["eat()", "sleep()"],
      color: "bg-blue-500",
    },
    Dog: {
      attributes: ["breed"],
      methods: ["bark()", "fetch()"],
      inherited: { from: "Animal", attributes: ["name", "age"], methods: ["eat()", "sleep()"] },
      color: "bg-green-500",
    },
    Cat: {
      attributes: ["indoor"],
      methods: ["meow()", "scratch()"],
      inherited: { from: "Animal", attributes: ["name", "age"], methods: ["eat()", "sleep()"] },
      color: "bg-orange-500",
    },
  };

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-foreground">Inheritance Hierarchy</h3>
        <Button
          size="sm"
          variant={showInherited ? "default" : "outline"}
          onClick={() => setShowInherited(!showInherited)}
        >
          {showInherited ? "Hide" : "Show"} Inherited Members
        </Button>
      </div>

      {/* Hierarchy Tree */}
      <div className="flex flex-col items-center gap-4">
        {/* Parent Class */}
        <div
          className={cn(
            "p-4 rounded-lg border-2 cursor-pointer transition-all w-64",
            activeClass === "Animal"
              ? "border-primary bg-primary/10 scale-105"
              : "border-border bg-background hover:border-primary/50"
          )}
          onClick={() => setActiveClass(activeClass === "Animal" ? null : "Animal")}
        >
          <div className="flex items-center gap-2 mb-2">
            <div className="w-4 h-4 rounded bg-blue-500" />
            <span className="font-bold text-foreground">Animal</span>
            <span className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-600 dark:text-blue-400 rounded">Parent</span>
          </div>
          <div className="text-sm space-y-1 pl-6">
            <div className="text-muted-foreground">+ name, age</div>
            <div className="text-orange-500">+ eat(), sleep()</div>
          </div>
        </div>

        {/* Arrow */}
        <div className="flex flex-col items-center">
          <div className="w-0.5 h-8 bg-border" />
          <div className="flex items-center gap-8">
            <div className="w-16 h-0.5 bg-border" />
            <div className="w-16 h-0.5 bg-border" />
          </div>
        </div>

        {/* Child Classes */}
        <div className="flex gap-8">
          {["Dog", "Cat"].map((className) => {
            const cls = classes[className as keyof typeof classes];
            const inherited = "inherited" in cls ? cls.inherited : null;
            
            return (
              <div
                key={className}
                className={cn(
                  "p-4 rounded-lg border-2 cursor-pointer transition-all w-56",
                  activeClass === className
                    ? "border-primary bg-primary/10 scale-105"
                    : "border-border bg-background hover:border-primary/50"
                )}
                onClick={() => setActiveClass(activeClass === className ? null : className)}
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className={cn("w-4 h-4 rounded", cls.color)} />
                  <span className="font-bold text-foreground">{className}</span>
                  <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-600 dark:text-green-400 rounded">Child</span>
                </div>
                <div className="text-sm space-y-1 pl-6">
                  {showInherited && inherited && (
                    <>
                      <div className="text-blue-500 opacity-60">
                        + {inherited.attributes.join(", ")}
                      </div>
                      <div className="text-blue-500 opacity-60">
                        + {inherited.methods.join(", ")}
                      </div>
                    </>
                  )}
                  <div className="text-muted-foreground">+ {cls.attributes.join(", ")}</div>
                  <div className="text-orange-500">+ {cls.methods.join(", ")}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-6 p-3 bg-muted rounded-lg text-sm text-muted-foreground">
        <strong className="text-foreground">Key Concept:</strong> Child classes (Dog, Cat) inherit all attributes and methods 
        from the parent class (Animal) and can add their own unique members.
      </div>
    </div>
  );
}

// Polymorphism Visualizer
export function PolymorphismVisualizer() {
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [currentStep, setCurrentStep] = React.useState(0);
  const [output, setOutput] = React.useState<string[]>([]);

  const animals = [
    { type: "Dog", sound: '"Woof!"', color: "bg-green-500" },
    { type: "Cat", sound: '"Meow!"', color: "bg-orange-500" },
    { type: "Cow", sound: '"Moo!"', color: "bg-pink-500" },
  ];

  React.useEffect(() => {
    if (!isPlaying) return;
    
    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= animals.length) {
          setIsPlaying(false);
          return prev;
        }
        setOutput((o) => [...o, `${animals[prev].type}.speak() → ${animals[prev].sound}`]);
        return prev + 1;
      });
    }, 1200);

    return () => clearInterval(timer);
  }, [isPlaying]);

  const start = () => {
    setIsPlaying(true);
    setCurrentStep(0);
    setOutput([]);
  };

  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
    setOutput([]);
  };

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-foreground">Polymorphism in Action</h3>
        <div className="flex gap-2">
          <Button size="sm" onClick={start} disabled={isPlaying}>
            <Play className="h-4 w-4 mr-1" /> Run
          </Button>
          <Button size="sm" variant="outline" onClick={reset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Code */}
        <div className="bg-muted/50 rounded-lg p-4 font-mono text-sm">
          <div className="text-muted-foreground mb-2"># Same method, different behavior</div>
          <div className="text-primary">for animal in [Dog(), Cat(), Cow()]:</div>
          <div className="pl-4 text-orange-500">animal.speak()</div>
        </div>

        {/* Animation */}
        <div className="space-y-3">
          {animals.map((animal, index) => (
            <div
              key={animal.type}
              className={cn(
                "flex items-center gap-4 p-3 rounded-lg border-2 transition-all",
                currentStep === index && isPlaying
                  ? "border-primary bg-primary/10 scale-105"
                  : currentStep > index
                  ? "border-success/50 bg-success/5"
                  : "border-border bg-background"
              )}
            >
              <div className={cn("w-12 h-12 rounded-full flex items-center justify-center text-white font-bold", animal.color)}>
                {animal.type[0]}
              </div>
              <div className="flex-1">
                <div className="font-medium text-foreground">{animal.type}</div>
                <div className="text-sm text-muted-foreground">speak() method</div>
              </div>
              {currentStep > index && (
                <div className="text-success font-mono text-sm animate-in fade-in">
                  {animal.sound}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Output Console */}
      {output.length > 0 && (
        <div className="mt-4 p-3 bg-foreground/5 rounded-lg font-mono text-sm">
          <div className="text-muted-foreground mb-1"># Output:</div>
          {output.map((line, i) => (
            <div key={i} className="text-success">{line}</div>
          ))}
        </div>
      )}

      <div className="mt-4 p-3 bg-muted rounded-lg text-sm text-muted-foreground">
        <strong className="text-foreground">Key Concept:</strong> Polymorphism allows objects of different classes 
        to be treated uniformly through a common interface, with each class providing its own implementation.
      </div>
    </div>
  );
}

// Encapsulation Visualizer
export function EncapsulationVisualizer() {
  const [balance, setBalance] = React.useState(1000);
  const [attemptedAccess, setAttemptedAccess] = React.useState<string | null>(null);
  const [lastAction, setLastAction] = React.useState<string | null>(null);

  const publicDeposit = (amount: number) => {
    setBalance((b) => b + amount);
    setLastAction(`deposit(${amount}) → Balance: $${balance + amount}`);
    setAttemptedAccess(null);
  };

  const publicWithdraw = (amount: number) => {
    if (amount <= balance) {
      setBalance((b) => b - amount);
      setLastAction(`withdraw(${amount}) → Balance: $${balance - amount}`);
    } else {
      setLastAction(`withdraw(${amount}) → Error: Insufficient funds!`);
    }
    setAttemptedAccess(null);
  };

  const attemptDirectAccess = () => {
    setAttemptedAccess("Access Denied! __balance is private");
    setLastAction(null);
  };

  const reset = () => {
    setBalance(1000);
    setAttemptedAccess(null);
    setLastAction(null);
  };

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-foreground">Encapsulation Demo: Bank Account</h3>
        <Button size="sm" variant="outline" onClick={reset}>
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Class Structure */}
        <div className="p-4 bg-muted/50 rounded-lg">
          <div className="font-mono text-sm space-y-1">
            <div className="text-primary font-bold">class BankAccount:</div>
            <div className="pl-4 flex items-center gap-2">
              <span className="text-destructive">private</span>
              <span>__balance = </span>
              <span className="text-success font-bold">${balance}</span>
            </div>
            <div className="pl-4 mt-2 text-muted-foreground"># Public methods</div>
            <div className="pl-4 text-green-500">+ deposit(amount)</div>
            <div className="pl-4 text-green-500">+ withdraw(amount)</div>
            <div className="pl-4 text-green-500">+ get_balance()</div>
          </div>
        </div>

        {/* Actions */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-foreground mb-2">Try accessing the account:</div>
          
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => publicDeposit(100)}>
              deposit(100)
            </Button>
            <Button size="sm" variant="outline" onClick={() => publicWithdraw(50)}>
              withdraw(50)
            </Button>
            <Button size="sm" variant="outline" onClick={() => publicWithdraw(2000)}>
              withdraw(2000)
            </Button>
          </div>

          <div className="pt-2 border-t border-border">
            <Button 
              size="sm" 
              variant="destructive" 
              onClick={attemptDirectAccess}
              className="w-full"
            >
              account.__balance = 1000000 (Direct Access)
            </Button>
          </div>
        </div>
      </div>

      {/* Feedback */}
      {(attemptedAccess || lastAction) && (
        <div className={cn(
          "mt-4 p-3 rounded-lg font-mono text-sm",
          attemptedAccess ? "bg-destructive/10 text-destructive" : "bg-success/10 text-success"
        )}>
          {attemptedAccess || lastAction}
        </div>
      )}

      <div className="mt-4 p-3 bg-muted rounded-lg text-sm text-muted-foreground">
        <strong className="text-foreground">Key Concept:</strong> Encapsulation hides internal data (like __balance) 
        and provides controlled access through public methods, preventing invalid states.
      </div>
    </div>
  );
}

// Abstraction Visualizer
export function AbstractionVisualizer() {
  const [selectedVehicle, setSelectedVehicle] = React.useState<string | null>(null);
  const [action, setAction] = React.useState<string | null>(null);

  const vehicles = [
    {
      type: "Car",
      color: "bg-blue-500",
      interface: ["start()", "stop()", "accelerate()"],
      hiddenDetails: ["Ignition system", "Fuel injection", "Transmission", "Engine timing"],
    },
    {
      type: "Electric Bike",
      color: "bg-green-500",
      interface: ["start()", "stop()", "accelerate()"],
      hiddenDetails: ["Battery management", "Motor controller", "Regenerative braking"],
    },
    {
      type: "Truck",
      color: "bg-orange-500",
      interface: ["start()", "stop()", "accelerate()"],
      hiddenDetails: ["Diesel engine", "Air brakes", "Hydraulic systems", "Load balancing"],
    },
  ];

  const handleAction = (vehicle: string, method: string) => {
    setAction(`${vehicle}.${method} executed successfully!`);
    setTimeout(() => setAction(null), 2000);
  };

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-foreground">Abstraction: Simple Interface, Complex Implementation</h3>
      </div>

      {/* Abstract Interface */}
      <div className="mb-6 p-4 bg-primary/10 rounded-lg border-2 border-dashed border-primary/50 text-center">
        <div className="text-sm text-muted-foreground mb-2">Abstract Class / Interface</div>
        <div className="font-mono font-bold text-primary">Vehicle</div>
        <div className="flex justify-center gap-4 mt-2 text-sm font-mono">
          <span className="text-orange-500">start()</span>
          <span className="text-orange-500">stop()</span>
          <span className="text-orange-500">accelerate()</span>
        </div>
      </div>

      {/* Concrete Implementations */}
      <div className="grid md:grid-cols-3 gap-4">
        {vehicles.map((vehicle) => (
          <div
            key={vehicle.type}
            className={cn(
              "p-4 rounded-lg border-2 cursor-pointer transition-all",
              selectedVehicle === vehicle.type
                ? "border-primary"
                : "border-border hover:border-primary/50"
            )}
            onClick={() => setSelectedVehicle(selectedVehicle === vehicle.type ? null : vehicle.type)}
          >
            <div className="flex items-center gap-2 mb-3">
              <div className={cn("w-8 h-8 rounded flex items-center justify-center text-white font-bold text-sm", vehicle.color)}>
                {vehicle.type[0]}
              </div>
              <span className="font-medium text-foreground">{vehicle.type}</span>
            </div>

            {/* Public Interface */}
            <div className="mb-3">
              <div className="text-xs text-muted-foreground mb-1">Public Interface:</div>
              <div className="flex flex-wrap gap-1">
                {vehicle.interface.map((method) => (
                  <button
                    key={method}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAction(vehicle.type, method);
                    }}
                    className="px-2 py-1 bg-success/20 text-success text-xs rounded hover:bg-success/30 transition-colors"
                  >
                    {method}
                  </button>
                ))}
              </div>
            </div>

            {/* Hidden Details */}
            {selectedVehicle === vehicle.type && (
              <div className="pt-3 border-t border-border animate-in fade-in slide-in-from-top-2">
                <div className="text-xs text-muted-foreground mb-1">Hidden Implementation:</div>
                <div className="space-y-1">
                  {vehicle.hiddenDetails.map((detail) => (
                    <div key={detail} className="flex items-center gap-1 text-xs text-muted-foreground">
                      <ChevronRight className="h-3 w-3" />
                      {detail}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Action Feedback */}
      {action && (
        <div className="mt-4 p-3 bg-success/10 text-success rounded-lg font-mono text-sm animate-in fade-in">
          {action}
        </div>
      )}

      <div className="mt-4 p-3 bg-muted rounded-lg text-sm text-muted-foreground">
        <strong className="text-foreground">Key Concept:</strong> Abstraction exposes only essential features (start, stop, accelerate) 
        while hiding complex implementation details. Users interact with a simple interface without knowing the internals.
      </div>
    </div>
  );
}
