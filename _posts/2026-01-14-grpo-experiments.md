---
layout: post
title: "Breaking the Cold Start Barrier: How Guidance Injection Accelerates Reinforcement Learning in Vision-Language Models"
date: 2025-01-14
categories: [Experiments, Side Projects, From-Scratch Implementations]
tags: [GRPO, VLM, Qwen2-VL, RL fine-tuning, guidance injection, PPO]
author: Suyash Shringarpure
image: /assets/img/posts/grpogrpo_cover.png
description: "An empirical study on GRPO training dynamics for vision-language models, exploring the tradeoffs between learning speed and cross-domain generalization when using guidance injection."
math: true
---

Reinforcement Learning from Human Feedback (RLHF) has revolutionized how we align large language models with human preferences. But what happens when we apply similar techniques to *vision-language* models? And more critically, what do we do when our base model cannot produce any reward signal at all?

In this post, we explore Group Relative Policy Optimization (GRPO) for fine-tuning vision-language models on counting tasks. We uncover a fundamental cold start problem in RL training and introduce **Guidance Injection**—a technique that bootstraps learning by providing expert demonstrations during training. Most importantly, we discover a surprising tradeoff: injection accelerates learning but can *hurt* cross-domain generalization.

![Cover image showing GRPO concept](/assets/img/posts/grpogrpo_cover.png)
*GRPO learns by comparing multiple responses to the same prompt, reinforcing better outputs over worse ones.*

---

## Table of Contents

1. [The Problem: Teaching a VLM to Count](#the-problem-teaching-a-vlm-to-count)
2. [From PPO to GRPO: A Simpler Approach to Policy Optimization](#from-ppo-to-grpo-a-simpler-approach-to-policy-optimization)
3. [Why Baseline Assessment is Critical](#why-baseline-assessment-is-critical)
4. [The Two Phases of GRPO Learning](#the-two-phases-of-grpo-learning)
5. [Guidance Injection: Breaking the Cold Start](#guidance-injection-breaking-the-cold-start)
6. [The Speed vs. Generalization Tradeoff](#the-speed-vs-generalization-tradeoff)
7. [The Role of Baseline Performance](#the-role-of-baseline-performance)
8. [Practical Recommendations](#practical-recommendations)
9. [Conclusion](#conclusion)

---

## The Problem: Teaching a VLM to Count

Object counting seems like a simple task for vision-language models—look at an image, count the objects, report a number. Yet when we prompted Qwen2-VL-2B-Instruct with counting questions on the CLEVR dataset, we encountered an unexpected challenge.

The model *understood* counting. It could accurately describe that "there are three blue cubes and two red spheres" in an image. But it almost never output *just* a number. Instead, it produced verbose descriptions, making it impossible for our reward function to extract a clean answer.

**The base model behavior:**
```
Prompt: "How many objects are in this image?"

Model Output: "Looking at this image, I can see several geometric shapes 
arranged on a gray surface. There appear to be three cubes - two blue 
and one yellow - along with two red spheres positioned near the cubes. 
In total, I count approximately five distinct objects in this scene, 
though the exact count may vary depending on how we define 'object'..."
```

Our reward function was intentionally simple:

```python
def compute_reward(response: str, ground_truth: int) -> int:
    """
    Extract the last digit from response, compare to ground truth.
    Returns: +1 (correct), 0 (wrong), -1 (no number found)
    """
    numbers = re.findall(r'\b(\d+)\b', response)
    if not numbers:
        return -1  # No extractable number
    predicted = int(numbers[-1])
    return 1 if predicted == ground_truth else 0
```

The simplicity was deliberate—we wanted the model to learn a clean output format through RL, not through clever reward engineering. But with the base model producing extractable numbers only **14% of the time**, we had a sparse reward signal at best.

This is the **cold start problem** in RL: when the policy cannot produce actions that receive positive feedback, learning stalls. If the digit extraction rate had been close to 0%, naive GRPO would have completely failed to learn.

---

## From PPO to GRPO: A Simpler Approach to Policy Optimization

Before diving into our experiments, it's worth understanding why we chose GRPO over the more established PPO (Proximal Policy Optimization) algorithm. The differences have significant practical implications for training vision-language models.

### PPO: The Standard Approach

PPO, introduced by Schulman et al. (2017), has become the workhorse of RLHF. Its core idea is to update the policy conservatively—clipping probability ratios to prevent destructive updates. The PPO objective looks like this:

$$L^{PPO}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the advantage estimate.

**The complexity comes from advantage estimation.** PPO typically uses Generalized Advantage Estimation (GAE), which requires:

1. **A value network** (critic) trained alongside the policy
2. **Temporal-difference bootstrapping** across multiple timesteps
3. **Careful tuning** of GAE parameters ($\lambda$, discount $\gamma$)

For autoregressive language models where "trajectories" are token sequences and rewards come only at the end, this machinery adds significant complexity. You need to either:
- Train a value head on top of your LLM (expensive in memory and compute)
- Use a separate value model (doubles your model count)
- Estimate returns through Monte Carlo rollouts (high variance)

### GRPO: Eliminating the Critic

Group Relative Policy Optimization, introduced in the DeepSeekMath paper, takes a radically simpler approach. Instead of estimating advantages through a learned value function, GRPO computes advantages **empirically within each prompt's response group**.

Here's the key insight: for each prompt, sample multiple responses and let the rewards themselves determine relative quality.

![GRPO Algorithm Diagram](/assets/img/posts/grpogrpo_algorithm.png)
*GRPO samples multiple responses per prompt, normalizes rewards within each group, then updates the policy to increase probability of relatively better responses.*

**The GRPO advantage calculation:**

For a prompt $x$ with $G$ sampled responses, each response $y_i$ receives reward $r_i$. The advantage is simply:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$$

where $\mu_G$ and $\sigma_G$ are the mean and standard deviation of rewards within the group.

**What this eliminates:**
- ❌ No value network needed (saves ~50% memory)
- ❌ No GAE computation
- ❌ No $\lambda$ or $\gamma$ hyperparameters to tune
- ❌ No temporal credit assignment

**What this enables:**
- ✅ Simpler implementation
- ✅ Lower memory footprint (critical for large VLMs)
- ✅ Naturally handles sparse, outcome-based rewards
- ✅ Works well with binary/ternary reward schemes

### The GRPO Training Loop

Our implementation follows this structure:

```python
for step in range(total_steps):
    # 1. Sample batch of prompts (images + questions)
    prompts = dataset.get_batch(batch_size=4)
    
    # 2. For each prompt, generate G responses
    for prompt in prompts:
        responses = model.generate(prompt, num_samples=8, temperature=1.0)
        rewards = [compute_reward(r, ground_truth) for r in responses]
        
        # 3. Normalize rewards within group
        advantages = (rewards - mean(rewards)) / std(rewards)
        
    # 4. PPO-style clipped update using computed advantages
    loss = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
    loss.backward()
    optimizer.step()
```

The clipping mechanism from PPO is retained to ensure stable updates, but the advantage computation is dramatically simplified.

### Why GRPO Works for VLMs

GRPO is particularly well-suited for vision-language tasks because:

1. **Outcome-based rewards**: VLM tasks often have clear right/wrong answers (counting, VQA, etc.). We don't need dense, per-token rewards.

2. **High variance in responses**: With temperature sampling, the same prompt generates diverse responses—some correct, some wrong. This diversity is exactly what GRPO needs.

3. **Memory constraints**: VLMs are already memory-intensive due to vision encoders. Eliminating the critic frees substantial GPU memory.

4. **Reward sparsity tolerance**: As long as *some* responses in each group get different rewards, GRPO can learn. This tolerance is crucial for the cold start problem.

---

## Why Baseline Assessment is Critical

Before running any RL training, we strongly recommend **evaluating your base model's format compliance rate**. This single metric determines whether naive GRPO will work or whether you need intervention techniques like guidance injection.

### Our Baseline Evaluation

We evaluated Qwen2-VL-2B-Instruct on 1000 CLEVR counting questions with a simple prompt:

```
"How many [objects] are in this image?"
```

**Results:**

| Metric | Value |
|--------|-------|
| **Digit Extraction Rate** | 14.5% |
| **Accuracy (when digit found)** | ~95% |
| **Overall Accuracy** | 14.0% |

This revealed a crucial insight: **the model already knows how to count**. When it produced a number, it was almost always correct! The problem was purely one of output format—the model preferred verbose descriptions over concise numerical answers.

### The Critical Threshold

Based on our experiments, here's how baseline digit rate affects GRPO training:

| Digit Rate | GRPO Viability | Recommendation |
|------------|---------------|----------------|
| **0-5%** | ❌ Will fail | Injection required |
| **5-15%** | ⚠️ Slow start | Consider injection |
| **15-30%** | ✅ Works | Injection optional |
| **>30%** | ✅ Works well | No injection needed |

At 14.5%, we were just above the threshold where naive GRPO could work—but it would be slow. This motivated our exploration of guidance injection.

### Why Does the Threshold Exist?

GRPO requires **reward variance within groups** to generate learning signal. If all 8 responses to a prompt get reward -1 (no number found), the standard deviation is zero, and the normalized advantage is undefined (or zero for all).

Mathematically, if $p$ is the probability of producing a parseable number:

$$P(\text{at least one number in group of } G) = 1 - (1-p)^G$$

For $G=8$ samples:
- $p = 5\%$: 34% chance of any learning signal
- $p = 14.5\%$: 71% chance of any learning signal  
- $p = 30\%$: 94% chance of any learning signal

At 14.5%, about 71% of our training batches had useful gradient signal. This was enough to bootstrap learning, but the early phases were inefficient.

---

## The Two Phases of GRPO Learning

When GRPO training does work, we observed a fascinating two-phase learning pattern in our experiments. **This pattern is specific to our task structure**—where the model needs to learn both output format and task accuracy—and may not generalize to all RL fine-tuning scenarios.

![Format vs Accuracy Learning](/assets/img/posts/grpoformat_vs_accuracy.png)
*GRPO training on our counting task shows distinct phases: first the model learns output format, then it refines accuracy.*

### Phase 1: Format Learning (Steps 0-600)

During the early training steps, the model learns *what kind* of output to produce. The digit extraction rate—the fraction of responses containing a parseable number—climbs from 14% toward 100%.

**What's happening internally:**
- The model discovers that outputting digits leads to higher relative rewards
- Verbose responses consistently get reward -1 (no number)
- Even wrong numbers (reward 0) are better than no numbers (reward -1)
- The policy shifts toward digit-producing behaviors

**Interestingly, accuracy remains relatively flat during this phase.** The model outputs numbers, but often wrong ones. The format learning dominates the gradient signal.

### Phase 2: Accuracy Learning (Steps 600+)

Once format is established (digit rate >90%), the reward signal changes character:
- Most responses now contain numbers
- Reward variance comes from correct (1) vs. wrong (0) answers
- The model learns *which* digit to output

Accuracy improves dramatically in this phase, climbing from ~50% to >90%.

### Training Dynamics Visualization

The training curves tell the story clearly:

**Early training (Steps 0-200: The Cold Start)**
- Mean reward ≈ -0.8 to -1.0 (very few digits in output)
- Number present rate hovers near the base model's 14.5%
- The sparse digit outputs provide just enough signal for GRPO to begin learning

**Middle training (Steps 200-600: The Virtuous Cycle)**
- Model starts outputting more digits → more differentiated rewards
- More differentiated rewards → stronger learning signal
- Number present rate rapidly climbs from ~15% to ~80%
- This is the "breakthrough" phase where format learning accelerates

**Late training (Steps 600-1600: Refinement)**
- Model consistently outputs digits (~90-100% rate)
- Learning signal now comes from correct vs. incorrect answers
- Accuracy improves steadily from ~80% to ~95%+
- Mean reward approaches +1.0

### Implications for Other Tasks

This two-phase pattern likely appears whenever:
1. The base model has a format mismatch with the reward function
2. Getting the format right is a prerequisite for meaningful task feedback
3. Format compliance is learnable through RL

Examples where this might apply:
- JSON output formatting
- Code generation (syntax before correctness)
- Structured reasoning (CoT format before accuracy)

---

## Guidance Injection: Breaking the Cold Start

What if the base model *never* outputs the desired format? With 0% digit rate, we'd have no reward variance, no learning signal, and training would be completely stuck.

**Guidance Injection** solves this by artificially inserting correct answers into a subset of model responses during training. Think of it as providing "expert demonstrations" that show the model what good outputs look like—but integrated directly into the RL training loop.

### The Mechanism

Here's how guidance injection works in our implementation:

1. **Sample responses normally**: Generate 8 responses per prompt using the current policy
2. **Select injection targets**: Randomly choose 1-7 responses (never all) to inject
3. **Replace with ground truth**: Swap the model's output with the correct answer
4. **Compute rewards**: Injected responses get reward +1, others get their earned rewards
5. **Update normally**: GRPO update uses the mixed batch

```python
# Simplified injection logic
def collect_experience_with_injection(prompt, ground_truth, num_samples=8):
    # Generate responses from policy
    responses = policy.generate(prompt, num_samples=num_samples)
    
    # Randomly select how many to inject (1 to num_samples-1)
    num_inject = random.randint(1, num_samples - 1)
    inject_indices = random.sample(range(num_samples), num_inject)
    
    # Replace selected responses with ground truth
    for i in inject_indices:
        responses[i] = format_correct_answer(ground_truth)
    
    # Compute rewards (injected ones will get +1)
    rewards = [compute_reward(r, ground_truth) for r in responses]
    
    return responses, rewards, inject_indices
```

**The critical constraint**: We never inject *all* responses. At least one must remain model-generated to ensure reward variance within each group. If all responses were injected with correct answers, they'd all get reward +1, standard deviation would be zero, and we'd have no learning signal.

### Why Does Injection Work?

Injection creates guaranteed reward variance:
- Injected responses: Always reward +1 (correct format, correct answer)
- Model responses: Usually reward -1 or 0 (wrong format or wrong answer)

This variance provides a consistent learning signal:
- "Increase probability of outputs that look like the injected examples"
- "Decrease probability of the model's natural verbose outputs"

The model learns by imitation embedded within the RL framework. It's similar to expert demonstration in imitation learning, but:
- Demonstrations are mixed with policy samples (not pre-training)
- The credit assignment still comes through GRPO's advantage normalization
- The model can generalize beyond the exact injected format

### Injection Rate Dynamics

In our experiments, we used a variable injection rate within each group:
- Minimum: 1 sample injected (ensures some positive reward)
- Maximum: 7 samples injected (ensures variance)
- Typical: 4-6 samples injected (~50-75% injection rate)

This stochastic injection serves multiple purposes:
1. **Prevents mode collapse**: The model sees varied correct examples
2. **Maintains exploration**: Some model outputs are always preserved
3. **Provides consistent signal**: Every batch has positive examples

---

## The Speed vs. Generalization Tradeoff

We trained two models on the CLEVR dataset with the default (weak baseline) prompt:
- **With Injection**: Correct answers injected for 50-87% of samples
- **Without Injection**: Pure GRPO, model generates all responses

### Training Dynamics

![Learning Curves Default](/assets/img/posts/grpolearning_curves_default.png)
*Training dynamics with weak baseline (14% digit rate). Injection dramatically accelerates early learning but converges to similar final performance.*

The training curves reveal stark differences:

**With Injection:**
- Mean reward climbs immediately from step 0
- Format learning is nearly instantaneous (model sees correct format)
- Reaches 90% accuracy by step ~600
- Training plateaus early (converges around step 700)

**Without Injection:**
- Mean reward stays negative for ~400 steps (cold start phase)
- Format learning happens gradually through exploration
- Reaches 90% accuracy by step ~1,400
- Takes 2.5x longer to converge

### In-Domain Results (CLEVR)

On the training distribution, both approaches achieved similar final performance:

| Approach | Steps to 90% | Final CLEVR Accuracy | Training Time |
|----------|-------------|---------------------|---------------|
| With Injection | ~600 | 93% | ~3 hours |
| Without Injection | ~1,600 | 93% | ~8 hours |

For researchers with limited compute budgets, the **2.5x speedup** from injection is significant. If you only care about performance on a single domain, injection appears to be a clear win.

### Cross-Domain Results (SuperCLEVR)

But the story changes dramatically when we evaluate on **SuperCLEVR**—a different visual domain with:
- Different 3D rendering style (more realistic)
- Different object shapes and textures  
- Varied question phrasings (not just our training template)

![Injection Effect Summary](/assets/img/posts/grpoinjection_effect_summary.png)
*Cross-domain transfer reveals injection's hidden cost: faster learning but significantly worse generalization.*

| Approach | SuperCLEVR Accuracy | Number Present Rate |
|----------|-------------------|-------------------|
| Base Model | 28% | 40% |
| With Injection | 30% | 67% |
| Without Injection | **48%** | 67% |

**Injection hurt generalization by 18 percentage points.**

Both fine-tuned models improved the number present rate similarly (40% → 67%), showing that format learning transferred. But accuracy improvement was vastly different:
- Without injection: +20% accuracy improvement
- With injection: +2% accuracy improvement

---

## Why Does Injection Hurt Generalization?

The answer lies in *how* the model learns output format under each condition.

### With Injection: Coupled Learning

When we inject correct answers, the model learns format from these injected examples. But the injected examples are always:
- Paired with CLEVR visual features (the training domain)
- Using the exact training question phrasing
- Formatted in one specific way

The model learns an implicit association: "When I see CLEVR-style images with this question style, output a digit." Format learning becomes **coupled to the training domain**.

During CLEVR evaluation, this coupling helps—the test distribution matches training. But during SuperCLEVR evaluation, the visual features don't match the learned association, and the model becomes less confident in the digit-output behavior.

### Without Injection: Decoupled Learning

Without injection, the model must *discover* the correct format through pure exploration. It tries many outputs:
- Some verbose descriptions
- Some partial numbers
- Some clean digit outputs

The ones that happen to contain digits get higher relative rewards. Over many training steps, the model learns: "Outputting digits gets rewards, regardless of what the image looks like."

This learning is **domain-agnostic**—it's not tied to specific visual features because the model discovered the format from its own varied outputs across varied images.

![Key Takeaway](/assets/img/posts/grpokey_takeaway.png)
*The fundamental insight: injection couples format learning to training domain features, while exploration-based learning keeps format and domain separable.*

### A Mechanistic Hypothesis

We can think of the model as learning two things:
1. **Format policy**: "How should I structure my output?"
2. **Content policy**: "What number should I output for this image?"

**With injection:**
- Format policy: Learned from injected examples → implicitly conditioned on CLEVR features
- Content policy: Learned through GRPO → generalizes based on reward signal

**Without injection:**
- Format policy: Learned through exploration → not conditioned on specific features
- Content policy: Learned through GRPO → generalizes based on reward signal

The coupling in injection isn't because of any flaw in the technique—it's a natural consequence of learning from examples that share domain features. The model doesn't know that format should be domain-independent; it just learns what correlates with reward.

---

## The Role of Baseline Performance

Our analysis so far assumed a **weak baseline**—a model that rarely produces the desired format naturally. But what if we could engineer a **strong baseline** through prompt engineering?

### Creating a Strong Baseline with R1V Prompting

We tested the R1-V prompting style (from the R1-V paper), which includes explicit format instructions:

**System Prompt:**
```
A conversation between User and Assistant. The user asks a question, 
and the Assistant solves it. The assistant first thinks about the 
reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> 
and <answer> </answer> tags, respectively.
```

**User Prompt Suffix:**
```
Output the thinking process in <think> </think> and final answer 
(number) in <answer> </answer> tags.
```

This simple prompt change transformed baseline performance:

![Baseline Comparison](/assets/img/posts/grpobaseline_comparison.png)
*Prompt engineering creates a strong baseline with 99% digit extraction rate, vs. 14% for the default prompt—a 6.8x improvement.*

| Prompt Style | Base Digit Rate | Base Accuracy |
|--------------|-----------------|---------------|
| Default (Weak) | 14% | 14% |
| R1V (Strong) | 99% | 68% |

With the R1V prompt, the model almost always produces responses in the `<think>...</think><answer>N</answer>` format. The digit extraction rate jumps from 14% to 99%, and base accuracy improves from 14% to 68% (since we're now extracting from most responses).

**The cold start problem essentially disappears.** Even without any RL training, 99% of responses are in the correct format.

### Does Injection Still Help with Strong Baseline?

We repeated our experiments with the R1V prompt:

![Learning Curves R1V](/assets/img/posts/grpolearning_curves_r1v.png)
*With strong baseline (R1V prompt), both injection and no-injection approaches show similar learning dynamics.*

**Training Dynamics:**

| Condition | Steps to 90% Accuracy | Final CLEVR Accuracy |
|-----------|----------------------|---------------------|
| R1V + Injection | ~800 | 97% |
| R1V + No Injection | ~700 | 95% |

With the strong baseline:
- Both methods reach high accuracy quickly
- The speedup from injection is minimal (if any)
- Final performance is similar

**Cross-Domain Transfer:**

![Transfer Results](/assets/img/posts/grpotransfer_results.png)
*With strong baseline, both training methods achieve similar cross-domain transfer to SuperCLEVR.*

| Condition | SuperCLEVR Accuracy |
|-----------|-------------------|
| R1V Base Model | 68% |
| R1V + Injection | 74-75% |
| R1V + No Injection | 72% |

The dramatic generalization gap we saw with weak baseline (30% vs. 48%) nearly disappears with strong baseline (74% vs. 72%).

### The Complete Picture

![Results Summary](/assets/img/posts/grporesults_summary_bar.png)
*Summary of all experimental conditions showing how baseline strength affects the injection decision.*

![Results Table](/assets/img/posts/grporesults_table_v2.png)
*Complete results table across all conditions. The number of training steps also varies between conditions.*

| Baseline | Injection | CLEVR | SuperCLEVR | Steps |
|----------|-----------|-------|------------|-------|
| Weak (14%) | Yes | 93% | 30% | 634 |
| Weak (14%) | No | 93% | 48% | 1,612 |
| Strong (99%) | Yes | 97% | 74-75% | 1,574 |
| Strong (99%) | No | 95% | 72% | 776* |

*Note: R1V + No Injection was trained for fewer steps due to early convergence.

### Why Does Strong Baseline Change Everything?

With a strong baseline, format learning is already solved by the prompt. The model doesn't need to learn "output digits"—it already does that 99% of the time.

Both training approaches (injection and no-injection) only need to teach **accuracy**: "output the *correct* digit." This learning:
- Doesn't benefit much from seeing injected examples
- Generalizes well because it's about visual understanding, not format
- Transfers across domains since the accuracy skill isn't domain-coupled

**Key insight**: The injection-generalization tradeoff exists because injection couples format learning to domain features. When format is pre-solved by prompting, there's nothing to couple, and the tradeoff disappears.

---

## Practical Recommendations

Based on our comprehensive experiments, here are actionable guidelines for practitioners applying RL fine-tuning to vision-language models.

### Decision Framework

```
                    ┌─────────────────────────────────────┐
                    │   Measure Base Model Format Rate    │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              Rate < 10%         10% < Rate < 50%    Rate > 50%
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ Try prompt    │  │ Consider your │  │ Use pure GRPO │
           │ engineering   │  │ deployment    │  │ No injection  │
           │ first         │  │ scenario      │  │ needed        │
           └───────┬───────┘  └───────┬───────┘  └───────────────┘
                   │                  │
                   ▼                  │
           Still < 10%?              │
           ┌────┴────┐               │
           Yes       No              │
           │         │               │
           ▼         ▼               ▼
    ┌─────────────┐ ┌─────────────────────────────────────────┐
    │ Injection   │ │ Single-domain: Injection OK             │
    │ Required    │ │ Multi-domain: Avoid/minimize injection  │
    └─────────────┘ └─────────────────────────────────────────┘
```

### Step 1: Assess Your Baseline

Before any training, measure your base model's format compliance rate on a sample of your task:

```python
def assess_baseline(model, eval_samples, reward_fn):
    """Measure format compliance and accuracy."""
    format_compliant = 0
    correct = 0
    
    for sample in eval_samples:
        response = model.generate(sample['prompt'])
        reward = reward_fn(response, sample['ground_truth'])
        
        if reward != -1:  # Number was extractable
            format_compliant += 1
            if reward == 1:  # Correct answer
                correct += 1
    
    return {
        'format_rate': format_compliant / len(eval_samples),
        'accuracy': correct / len(eval_samples),
        'accuracy_given_format': correct / max(format_compliant, 1)
    }
```

Key metrics to collect:
- **Format rate**: What fraction of responses are in the desired format?
- **Accuracy given format**: When format is correct, how often is the answer correct?
- **Overall accuracy**: Format rate × accuracy given format

If accuracy given format is high, the model already has the capability—you just need to shape the output format.

### Step 2: Try Prompt Engineering First

Before resorting to injection, try improving the baseline through prompting:

**Techniques that often help:**
1. **Explicit format instructions**: "Answer with just a number."
2. **Format templates**: "Output your answer in the format: <answer>N</answer>"
3. **Few-shot examples**: Show 1-2 examples of desired format (if context allows)
4. **System prompts**: Set expectations for output style
5. **Chain-of-thought with structure**: `<think>...</think><answer>...</answer>`

The R1V prompting style improved our format rate from 14% to 99%. Similar gains may be possible for your task.

### Step 3: Choose Your Training Strategy

Based on baseline rate and deployment scenario:

**Scenario A: Strong baseline (>50% format rate)**
- Use pure GRPO
- Injection provides minimal benefit
- Both methods generalize well

**Scenario B: Weak baseline, single-domain deployment**
- Injection is efficient and safe
- Speed benefit outweighs transfer concerns
- Example: Production system always uses same visual style

**Scenario C: Weak baseline, multi-domain deployment**
- Avoid injection or use minimal rates
- Pay the compute cost for better generalization
- Example: General-purpose assistant handling varied images

**Scenario D: Very weak baseline (<5%), can't improve via prompting**
- Injection is necessary to bootstrap learning
- Consider injection decay schedule
- Accept some generalization cost as unavoidable

### Step 4: Consider Hybrid Approaches

If you need both speed and generalization, consider **injection decay**:

```python
def get_injection_rate(step, total_steps):
    """Decay injection from high to zero over training."""
    # High injection early (bootstrap), pure GRPO late (generalize)
    initial_rate = 0.8
    final_rate = 0.0
    decay_fraction = 0.7  # Decay for first 70% of training
    
    if step > decay_fraction * total_steps:
        return final_rate
    
    progress = step / (decay_fraction * total_steps)
    return initial_rate * (1 - progress) + final_rate * progress
```

This gives:
- Early training: High injection → fast format learning
- Late training: Low/no injection → exploration-based generalization

### Step 5: Always Evaluate Cross-Domain

Regardless of training strategy, always evaluate on out-of-distribution data:

```python
# Don't just check in-domain accuracy
clevr_accuracy = evaluate(model, clevr_test_set)

# Also check transfer
superclevr_accuracy = evaluate(model, superclevr_test_set)
other_vqa_accuracy = evaluate(model, other_vqa_set)

# Log the ratio
generalization_ratio = superclevr_accuracy / clevr_accuracy
print(f"Generalization ratio: {generalization_ratio:.2f}")
```

A generalization ratio close to 1.0 indicates robust learning. A ratio << 1.0 suggests overfitting to training domain—consider reducing injection.

---

## Limitations and Future Work

Our study provides insights but has important limitations that warrant further investigation:

### Limitations

**Training Duration Mismatch**: Our R1V experiments had different training durations between injection (1,574 steps) and no-injection (776 steps) conditions. The no-injection run converged earlier, but a controlled comparison with matched compute would strengthen conclusions about the strong baseline regime.

**Single Task**: We focused exclusively on object counting. Other vision-language tasks (VQA, captioning, reasoning, grounding) may exhibit different patterns. Tasks with more complex output formats or less clear-cut correctness criteria might show different tradeoffs.

**Single Model Architecture**: Results are specific to Qwen2-VL-2B-Instruct. Larger models (7B, 72B) or different architectures (LLaVA, InternVL) may behave differently. Larger models might have better base format compliance, reducing the need for injection.

**Fixed Injection Schedule**: We used stochastic but non-decaying injection rates. Adaptive schedules based on model capability (e.g., decay when format rate exceeds threshold) might achieve better speed-generalization balance.

**Binary Reward Structure**: Our reward function was intentionally simple (+1, 0, -1). More nuanced rewards (e.g., partial credit for close answers) might interact differently with injection.

### Future Directions

1. **Injection decay schedules**: Systematically study how different decay curves affect the speed-generalization tradeoff.

2. **Larger scale experiments**: Validate findings on larger models and more diverse tasks.

3. **Understanding the coupling mechanism**: Use interpretability techniques to understand exactly what features the model associates with format when trained with vs. without injection.

4. **Other bootstrap techniques**: Compare injection against alternatives like reward shaping, curriculum learning, or supervised fine-tuning warmup.

5. **Theoretical analysis**: Develop formal understanding of when format-domain coupling occurs and how to prevent it.

---

## Conclusion

Reinforcement learning for vision-language models faces a fundamental challenge: the **cold start problem**. When base models cannot produce outputs in the format expected by reward functions, learning cannot begin. We've shown that even a modest format compliance rate (14%) can be sufficient for GRPO to bootstrap learning, but the process is slow and inefficient.

**Guidance Injection** offers a solution by providing expert demonstrations during training, allowing the model to see correct outputs and learn from them directly. This technique dramatically accelerates training—achieving 2.5x speedup in our experiments.

But this speedup comes with a cost: **format learning becomes coupled to training domain features**, hurting cross-domain generalization. Models trained with injection learned "output digits when you see CLEVR images" rather than the more general "output digits for counting tasks." This coupling caused an 18 percentage point drop in cross-domain accuracy.

The key insight is that **baseline performance determines tradeoffs**:

| Baseline | Injection Effect |
|----------|-----------------|
| Weak (14%) | Large speedup, significant generalization cost |
| Strong (99%) | Minimal speedup, minimal generalization cost |

With a strong baseline achieved through prompt engineering, the injection decision becomes nearly irrelevant—both approaches work well.

**For practitioners, this suggests a clear priority order:**

1. **Engineer a strong baseline through prompting**—this eliminates the cold start problem without injection's costs.

2. **If prompting fails, consider your deployment scenario**—single-domain uses can safely employ injection; multi-domain uses should minimize or avoid it.

3. **Consider hybrid approaches**—injection decay gives early-stage speed while allowing late-stage exploration.

4. **Always evaluate on out-of-distribution data**—the speedup you gain may come at the cost of the generalization you need.

The lesson extends beyond GRPO to a general principle in RL: **when accelerating learning through expert guidance, be aware of implicit correlations the model might learn**. Expert demonstrations are never truly "domain-free"—they carry features of their context. Understanding this coupling is crucial for building models that generalize beyond their training distribution.

---

## Appendix: Experimental Details

### Model Configuration

```python
# Base model
model_name = "Qwen/Qwen2-VL-2B-Instruct"
torch_dtype = torch.bfloat16

# GRPO Configuration  
grpo_config = GRPOConfig(
    clip_epsilon = 0.2,
    num_samples_per_experience = 8,  # G = 8 responses per prompt
    num_experiences_per_update = 4,   # 4 prompts per batch
    num_grpo_epochs = 4,              # PPO-style multiple epochs
    kl_loss_coef = 0.0,               # No KL penalty
    use_reference_model = True,        # For logging, not loss
)

# Training Configuration
training_config = TrainingConfig(
    learning_rate = 1e-5,
    weight_decay = 0.01,
    total_steps = 2000,
    warmup_steps_frac = 0.1,
    max_grad_norm = 1.0,
)

# Generation Configuration
generation_config = GenerationConfig(
    max_new_tokens = 50,
    temperature = 1.0,
    top_p = 1.0,
    do_sample = True,
)
```

### Datasets

**CLEVR (Training)**:
- Source: `leonardPKU/clevr_cogen_a_train`
- Task: Object counting in synthetic 3D scenes
- 15,000 samples reserved for evaluation
- Remaining samples for training (streaming)

**SuperCLEVR (Transfer Evaluation)**:
- Different rendering style
- Different object types
- Varied question phrasings
- 200 samples evaluated

### Reward Function

```python
def compute_reward(response: str, ground_truth: int) -> int:
    # For R1V style: extract from <answer> tags
    match = re.search(r"<answer>\s*(\d+)\s*</answer>", response)
    if match:
        predicted = int(match.group(1))
        return 1 if predicted == ground_truth else 0
    
    # Fallback: extract last number
    numbers = re.findall(r"\b(\d+)\b", response)
    if numbers:
        predicted = int(numbers[-1])
        return 1 if predicted == ground_truth else 0
    
    return -1  # No number found
```

---

## References

1. Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv preprint arXiv:2402.03300.

2. DeepSeek-AI. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint arXiv:2501.12948.

3. Liu, Y., et al. (2024). "R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3." GitHub repository.

4. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347.

5. Johnson, J., et al. (2017). "CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning." CVPR.

6. Wang, P., et al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv preprint arXiv:2409.12191.

---

<div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
<h4 style="color: #333;">📦 VLM GRPO Repository</h4>
<p style="color: #333;">GRPO implementation for fine-tuning Vision-Language Models. Includes configurable experiments, guidance injection, and cross-domain evaluation.</p>
<a href="https://github.com/suyashh94/vlm-grpo" target="_blank" style="display: inline-block; background-color: #333; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">View on GitHub →</a>
</div>

---

*Questions about GRPO, VLM fine-tuning, or RL in general? Reach out on [LinkedIn](https://linkedin.com/in/suyash94) or open an issue on the [GitHub repository](https://github.com/suyashh94/vlm-grpo).*
