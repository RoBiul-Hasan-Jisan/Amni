import Link from "next/link";
import { AlertCircle, CheckCircle2, Lightbulb, Code, Clock, HardDrive, Target, Eye, Hash, Zap } from "lucide-react";

export default function TwoSumPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation Bar */}
      <nav className="bg-card border-b border-border shadow-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/learn" className="text-primary hover:text-primary/80 hover:underline transition">
                ← Back to Learning Path
              </Link>
            </div>
            <div className="text-foreground font-semibold">
              Two Sum
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Problem Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <span className="bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 px-3 py-1 rounded-full text-sm font-semibold">
              Easy
            </span>
            <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm font-semibold">
              Array • Hash Table
            </span>
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-3">
            Two Sum
          </h1>
          <p className="text-muted-foreground text-lg">
            Given an integer array <code className="bg-muted px-2 py-0.5 rounded">nums</code> and an integer <code className="bg-muted px-2 py-0.5 rounded">target</code>, return the indices of two different elements such that their sum is <code className="bg-muted px-2 py-0.5 rounded">target</code>.
          </p>
        </div>

        {/* Problem Link */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-8">
          <div className="flex gap-3">
            <Target className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-foreground">Problem Link</p>
              <a 
                href="https://leetcode.com/problems/two-sum/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline break-all"
              >
                https://leetcode.com/problems/two-sum/
              </a>
            </div>
          </div>
        </div>

        {/* Assumptions */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Assumptions
          </h2>
          <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-1">
            <li>Exactly one valid pair exists.</li>
            <li>You cannot use the same element twice.</li>
            <li>Return the answer in any order.</li>
          </ul>
        </section>

        {/* Examples */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Examples
          </h2>
          
          <div className="space-y-4">
            {/* Example 1 */}
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Example 1</p>
              <pre className="bg-muted/50 p-3 rounded text-primary font-mono text-sm overflow-x-auto">
                {`Input: nums = [1,8,10,14], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 1 + 8 = 9, so the answer is [0,1].`}
              </pre>
              <div className="flex justify-around mt-4 text-center">
                {[1, 8, 10, 14].map((num, idx) => (
                  <div key={idx} className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center text-primary font-bold text-lg">
                      {num}
                    </div>
                    <div className="text-muted-foreground text-sm mt-1">{idx}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Example 2 */}
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Example 2</p>
              <pre className="bg-muted/50 p-3 rounded text-primary font-mono text-sm overflow-x-auto">
                {`Input: nums = [5,1,6,9], target = 10
Output: [1,3]
Explanation: nums[1] + nums[3] = 1 + 9 = 10, so the answer is [1,3].`}
              </pre>
              <div className="flex justify-around mt-4 text-center">
                {[5, 1, 6, 9].map((num, idx) => (
                  <div key={idx} className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center text-primary font-bold text-lg">
                      {num}
                    </div>
                    <div className="text-muted-foreground text-sm mt-1">{idx}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Example 3 */}
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Example 3</p>
              <pre className="bg-muted/50 p-3 rounded text-primary font-mono text-sm overflow-x-auto">
                {`Input: nums = [4,13,7,2], target = 9
Output: [2,3]
Explanation: nums[2] + nums[3] = 7 + 2 = 9, so the answer is [2,3].`}
              </pre>
              <div className="flex justify-around mt-4 text-center">
                {[4, 13, 7, 2].map((num, idx) => (
                  <div key={idx} className="text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center text-primary font-bold text-lg">
                      {num}
                    </div>
                    <div className="text-muted-foreground text-sm mt-1">{idx}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Constraints */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Constraints
          </h2>
          <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-1">
            <li><code className="bg-muted px-2 py-0.5 rounded">2 &lt;= nums.length &lt;= 10⁴</code></li>
            <li><code className="bg-muted px-2 py-0.5 rounded">-10⁹ &lt;= nums[i] &lt;= 10⁹</code></li>
            <li><code className="bg-muted px-2 py-0.5 rounded">-10⁹ &lt;= target &lt;= 10⁹</code></li>
            <li>Exactly one valid answer exists.</li>
          </ul>
        </section>

        {/* Solutions Comparison */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Solutions Comparison
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Time</th>
                  <th className="text-left p-3 font-semibold text-foreground">Space</th>
                  <th className="text-left p-3 font-semibold text-foreground">Best for</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Brute Force</td>
                  <td className="p-3 font-mono">O(n²)</td>
                  <td className="p-3 font-mono">O(1)</td>
                  <td className="p-3">Small arrays</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Hash Map (Two-pass)</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3">Most cases</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Hash Map (One-pass)</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3 text-primary font-semibold">Optimal</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Solution Idea */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Solution Idea
          </h2>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="text-foreground font-semibold mb-2">Scan once and keep a hash map from number → index.</p>
                <p className="text-muted-foreground">At each position i:</p>
                <ul className="list-disc list-inside text-muted-foreground ml-4 mt-2 space-y-1">
                  <li><code className="bg-muted px-2 py-0.5 rounded">needed = target - nums[i]</code></li>
                  <li>If <code className="bg-muted px-2 py-0.5 rounded">needed</code> is already in the map, we found the pair.</li>
                  <li>Otherwise, store <code className="bg-muted px-2 py-0.5 rounded">nums[i]</code> with index <code className="bg-muted px-2 py-0.5 rounded">i</code>.</li>
                </ul>
                <p className="text-muted-foreground mt-2">This gives linear time — O(n).</p>
              </div>
            </div>
          </div>
        </section>

        {/* Code Section */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Code (Python 3)
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <div className="bg-muted px-4 py-2 border-b border-border">
              <div className="flex items-center gap-2">
                <Code className="h-4 w-4 text-primary" />
                <span className="text-foreground font-mono text-sm">Solution</span>
              </div>
            </div>
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from typing import List
 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = dict()  # value -> index
 
        for i, num in enumerate(nums):
            needed = target - num
 
            if needed in seen:
                return [seen[needed], i]
 
            seen[num] = i
 
        return []`}
            </pre>
          </div>
        </section>

        {/* Line-by-Line Explanation */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Line-by-Line Explanation
          </h2>
          <div className="space-y-3">
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-mono text-primary text-sm">seen = dict()  # value → index</p>
              <p className="text-muted-foreground text-sm mt-1">Create an empty hash map to store numbers we've seen. Key: number value, Value: index of that number.</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-mono text-primary text-sm">for i, num in enumerate(nums):</p>
              <p className="text-muted-foreground text-sm mt-1">Iterate through array with both index and value.</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-mono text-primary text-sm">needed = target - num</p>
              <p className="text-muted-foreground text-sm mt-1">Calculate what number we need to reach target. If current num is 2 and target is 9, we need 7.</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-mono text-primary text-sm">if needed in seen:</p>
              <p className="text-muted-foreground text-sm mt-1">Check if we've seen the complement before.</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-mono text-primary text-sm">return [seen[needed], i]</p>
              <p className="text-muted-foreground text-sm mt-1">Found! Return indices: [index of complement, current index].</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-mono text-primary text-sm">seen[num] = i</p>
              <p className="text-muted-foreground text-sm mt-1">Haven't found complement yet, store current number. We store AFTER checking so we don't use same element twice.</p>
            </div>
          </div>
        </section>

        {/* Dry Run Examples */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Dry Run Examples
          </h2>
          
          {/* Example 1 Dry Run */}
          <div className="bg-card border border-border rounded-lg p-4 mb-4">
            <p className="font-semibold text-foreground mb-2">Example 1: nums = [2,7,11,15], target = 9</p>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left p-2 text-foreground">Step</th>
                    <th className="text-left p-2 text-foreground">Index</th>
                    <th className="text-left p-2 text-foreground">num</th>
                    <th className="text-left p-2 text-foreground">complement</th>
                    <th className="text-left p-2 text-foreground">seen map</th>
                    <th className="text-left p-2 text-foreground">Action</th>
                  </tr>
                </thead>
               <tbody className="text-muted-foreground text-sm">
  <tr className="border-b border-border">
    <td className="p-2">1</td>
    <td className="p-2">0</td>
    <td className="p-2">3</td>
    <td className="p-2">3</td>
    <td className="p-2 font-mono">{"{}"}</td>
    <td className="p-2">
      3 not in map → store {"{3:0}"}
    </td>
  </tr>

  <tr className="border-b border-border">
    <td className="p-2">2</td>
    <td className="p-2">1</td>
    <td className="p-2">2</td>
    <td className="p-2">4</td>
    <td className="p-2 font-mono">{"{3:0}"}</td>
    <td className="p-2">
      4 not in map → store {"{3:0, 2:1}"}
    </td>
  </tr>

  <tr className="border-b border-border">
    <td className="p-2">3</td>
    <td className="p-2">2</td>
    <td className="p-2">4</td>
    <td className="p-2">2</td>
    <td className="p-2 font-mono">{"{3:0, 2:1}"}</td>
    <td className="p-2 text-green-600">
      2 IN map! → return [1,2] ✅
    </td>
  </tr>
</tbody>
              </table>
            </div>
          </div>

          {/* Example 2 Dry Run */}
          <div className="bg-card border border-border rounded-lg p-4">
            <p className="font-semibold text-foreground mb-2">Example 2: nums = [3,2,4], target = 6</p>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left p-2 text-foreground">Step</th>
                    <th className="text-left p-2 text-foreground">Index</th>
                    <th className="text-left p-2 text-foreground">num</th>
                    <th className="text-left p-2 text-foreground">complement</th>
                    <th className="text-left p-2 text-foreground">seen map</th>
                    <th className="text-left p-2 text-foreground">Action</th>
                  </tr>
                </thead>
<tbody className="text-muted-foreground text-sm">
  <tr className="border-b border-border">
    <td className="p-2">1</td>
    <td className="p-2">0</td>
    <td className="p-2">3</td>
    <td className="p-2">3</td>
    <td className="p-2 font-mono">{"{}"}</td>
    <td className="p-2">
      3 not in map → store {"{3:0}"}
    </td>
  </tr>

  <tr className="border-b border-border">
    <td className="p-2">2</td>
    <td className="p-2">1</td>
    <td className="p-2">2</td>
    <td className="p-2">4</td>
    <td className="p-2 font-mono">{"{3:0}"}</td>
    <td className="p-2">
      4 not in map → store {"{3:0,2:1}"}
    </td>
  </tr>

  <tr className="border-b border-border">
    <td className="p-2">3</td>
    <td className="p-2">2</td>
    <td className="p-2">4</td>
    <td className="p-2">2</td>
    <td className="p-2 font-mono">{"{3:0,2:1}"}</td>
    <td className="p-2 text-green-600">
      2 IN map! → return [1,2] ✅
    </td>
  </tr>
</tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Complexity Analysis */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Complexity Analysis
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="h-5 w-5 text-primary" />
                <h3 className="font-semibold text-foreground">Time Complexity: O(n)</h3>
              </div>
              <ul className="list-disc list-inside text-muted-foreground text-sm ml-4 space-y-1">
                <li>Single pass through array: n iterations</li>
                <li>Hash map lookup (average): O(1)</li>
                <li>Total: O(n) operations</li>
              </ul>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <HardDrive className="h-5 w-5 text-primary" />
                <h3 className="font-semibold text-foreground">Space Complexity: O(n)</h3>
              </div>
              <ul className="list-disc list-inside text-muted-foreground text-sm ml-4 space-y-1">
                <li>Hash map stores at most n-1 elements before finding solution</li>
                <li>In worst case (answer at end), stores n elements</li>
                <li>Each element: integer key + integer value</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Edge Cases */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Edge Cases to Discuss
          </h2>
          <div className="space-y-3">
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-semibold text-foreground text-sm">Case 1: Negative numbers</p>
              <pre className="text-primary font-mono text-sm mt-1">nums = [-1, -2, -3, -4, -5], target = -8  # Works fine</pre>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-semibold text-foreground text-sm">Case 2: Large numbers</p>
              <pre className="text-primary font-mono text-sm mt-1">nums = [1000000000, 2000000000], target = 3000000000  # Works within constraints</pre>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-semibold text-foreground text-sm">Case 3: Duplicate values</p>
              <pre className="text-primary font-mono text-sm mt-1">nums = [1, 1, 2], target = 2  # Returns [0,1] (stores first occurrence)</pre>
            </div>
            <div className="bg-card border border-border rounded-lg p-3">
              <p className="font-semibold text-foreground text-sm">Case 4: Answer uses same value at different positions</p>
              <pre className="text-primary font-mono text-sm mt-1">nums = [5, 5], target = 10  # Returns [0,1]</pre>
            </div>
          </div>
        </section>

        {/* Common Mistakes */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Mistakes to Avoid
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <p className="text-sm text-foreground">❌ <code className="bg-muted px-1.5 py-0.5 rounded">if num in seen:</code> before checking complement (would use same element)</p>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <p className="text-sm text-foreground">❌ Using array/list for lookup (would be O(n) per operation)</p>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <p className="text-sm text-foreground">❌ Returning values instead of indices</p>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <p className="text-sm text-foreground">❌ Forgetting about negative numbers and zero</p>
              </div>
            </div>
          </div>
        </section>

        {/* Follow-up Questions */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Follow-up Questions They Might Ask
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Q: What if there are multiple solutions?</p>
                  <p className="text-muted-foreground text-sm">A: Return first found (order doesn't matter per problem).</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Q: What if no solution exists?</p>
                  <p className="text-muted-foreground text-sm">A: Problem guarantees exactly one solution.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Q: Can we use sorting approach?</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`def twoSumSorting(nums, target):
    # O(n log n) time, O(n) space
    indexed = [(val, idx) for idx, val in enumerate(nums)]
    indexed.sort()
    left, right = 0, len(nums) - 1
    while left < right:
        curr_sum = indexed[left][0] + indexed[right][0]
        if curr_sum == target:
            return [indexed[left][1], indexed[right][1]]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1`}
                  </pre>
                  <p className="text-muted-foreground text-sm mt-2">Q: Why choose hash map over sorting? Hash map is O(n) vs sorting's O(n log n). For 10,000 elements, hash map is ~4x faster.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interview-Ready Explanation */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Interview-Ready Explanation
          </h2>
          <div className="bg-muted/30 rounded-lg p-4 border border-border">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="h-5 w-5 text-primary" />
              <span className="font-semibold text-foreground">Quick Mental Check for Interview</span>
            </div>
            <pre className="text-muted-foreground text-sm whitespace-pre-wrap">
              {`1. Understand: Find two numbers that sum to target
2. Constraint: Single solution, can't reuse element
3. Optimal: Hash map in O(n) time, O(n) space
4. Process: 
   - Map stores seen numbers
   - For each num, check if complement exists
   - Return indices when found`}
            </pre>
            <p className="text-muted-foreground text-sm mt-3">This explanation covers everything you need for a solid interview response!</p>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-border mt-4">
          <Link
            href="/learn/programming-fundamentals/advanced-python"
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Advanced Python
          </Link>
          <Link
            href="/learn/letcode-problem/valid-parentheses"
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Valid Parentheses →
          </Link>
        </div>
      </div>
    </div>
  );
}