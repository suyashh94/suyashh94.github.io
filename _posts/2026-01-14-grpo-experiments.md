---
layout: post
title: "Building GRPO from Scratch: Format Learning, Guidance Injection, and Why Baselines Matter"
date: 2026-01-13 10:00:00 +0530
categories: [Experiments, From-Scratch Implementations]
tags: [grpo, ppo, qwen, vision-language-models, rlhf]
description: "A complete implementation of Proximal Policy Optimization for fine-tuning language models, with detailed explanations of GAE, clipped objectives, and the training dynamics that make RLHF work."
image:
  path: /assets/img/posts/grpo/grpo_cover.png
  alt: GRPO from Scratch - Reinforcement Learning for Language Model Alignment
math: true
mermaid: true
pin: true
---

Reinforcement learning from human feedback (RLHF) has become a cornerstone of modern language model training. But what happens when we apply these techniques to vision-language models (VLMs)? In this post, I share my experience implementing **Group Relative Policy Optimization (GRPO)** from scratch to fine-tune Qwen2-VL on counting tasks, and the surprising lessons learned about baseline performance, guidance injection, and cross-domain generalization.

## The Challenge: Sparse Reward Signals in VLM Fine-Tuning

Vision-language models like Qwen2-VL are impressive out of the box, but fine-tuning them for specific tasks presents unique challenges. Consider a simple counting task: given an image, the model should output how many objects are present.

The challenge? When you ask a base VLM "How many items are there in this image?", you might get responses like:

> "The image shows a colorful scene with various geometric shapes arranged in an interesting pattern..."

This response is coherent and reasonable, but it doesn't contain an extractable number. In my experiments with Qwen2-VL-2B-Instruct, only **~14% of base model outputs contained a digit** that could be evaluated for correctness.

This creates a fundamental problem for reinforcement learning: if most outputs can't be scored, how does the model learn?

![Baseline performance comparison showing 14% vs 99% digit extraction rates](/assets/img/posts/grpo/baseline_comparison.png)
*Figure 1: The stark difference in baseline performance between prompt styles. The Default prompt achieves only 14.5% digit extraction rate, while the R1V prompt with explicit format instructions achieves 99%.*

## GRPO: Learning from Relative Comparisons

**Group Relative Policy Optimization (GRPO)** offers an elegant solution to sparse rewards. Unlike standard PPO which uses absolute reward values, GRPO compares multiple responses to the same prompt and learns from their *relative* quality.

### How GRPO Works

The algorithm follows these steps:

1. **Experience Collection**: For each training prompt, generate multiple responses (e.g., 8 samples)
2. **Reward Computation**: Score each response (+1 for correct, 0 for wrong, -1 for no number)
3. **Advantage Normalization**: Normalize rewards *within each group* to create relative advantages
4. **Policy Update**: Use PPO-style clipped objective with the normalized advantages

The key insight is in step 3. Even if 7 out of 8 responses receive a -1 reward (no extractable number), and only 1 response gets a +1, GRPO can learn from this contrast. The single correct response becomes a strong positive signal relative to the others.

![GRPO algorithm illustration](/assets/img/posts/grpo/grpo_algorithm.png)
*Figure 2: GRPO generates multiple responses per prompt and normalizes rewards within groups, enabling learning even from sparse signals.*

### The Reward Function

For counting tasks, I implemented a simple three-valued reward:

| Model Output | Ground Truth | Reward |
|:-------------|:-------------|:-------|
| "The answer is **5**" | 5 | +1 (correct) |
| "The answer is **3**" | 5 | 0 (incorrect) |
| "There are several objects" | 5 | -1 (no number) |

The reward function extracts the *last* number from the model's response (handling cases like "I see 3 red objects and 2 blue objects, so **5** total") and compares it to ground truth.

## Learning Dynamics: Two Phases of Training

Training a model with GRPO on the weak baseline (14% digit rate) revealed an interesting two-phase learning pattern:

![Format learning vs accuracy learning](/assets/img/posts/grpo/format_vs_accuracy.png)
*Figure 3: Training proceeds in two distinct phases. First, the model learns to output digits (format learning). Then, it learns to output the correct digit (accuracy learning).*

### Phase 1: Format Learning

In the early stages of training, the primary change is in *output format*. The model learns that responses containing numbers receive higher rewards than verbose descriptions. The digit extraction rate climbs from 14% to nearly 100%.

### Phase 2: Accuracy Learning

Once the model reliably produces numbers, training shifts to accuracy. The model has learned *what* to output (a number), and now learns *which* number is correct. This phase is where actual task performance improves.

This observation has important implications: **the model must first learn the reward-earning format before it can optimize for correctness.**

## Guidance Injection: Accelerating Learning

Given that format learning takes time, can we accelerate it? This is where **Guidance Injection** comes in.

### The Cold Start Problem

With only 14% of outputs containing numbers, GRPO faces a cold start problem. In any group of 8 samples, we might get 0-2 samples with extractable answers. The learning signal is sparse, and early training is slow.

### How Guidance Injection Works

Guidance Injection is simple: during training, we replace some model outputs with the correct answer. If the model generates "There are several objects" but the ground truth is 5, we substitute "5" as one of the training samples.

This guarantees reward variance in every group—at least one sample will be correct (+1 reward), providing a consistent learning signal.

```python
# Simplified injection logic
if injection_enabled:
    # Ensure at least one correct sample per group
    for group in training_groups:
        if not any_correct(group):
            group[random_idx].response = str(ground_truth)
            group[random_idx].reward = 1
```

### The Impact on Learning Speed

The effect is dramatic:

![Learning curves with and without injection](/assets/img/posts/grpo/learning_curves_default.png)
*Figure 4: With Guidance Injection (red), the model reaches high accuracy much faster. Without injection (blue), learning is slower but eventually reaches the same level.*

With the Default prompt (weak baseline):
- **With Injection**: Reaches 90% accuracy at ~470 steps
- **Without Injection**: Reaches 90% accuracy at ~1,490 steps
- **Speedup**: ~3.2×

Both approaches eventually achieve ~93% accuracy on the training domain (CLEVR). The difference is purely in learning speed.

## The Generalization Tradeoff

Here's where things get interesting. Speed isn't free.

### Cross-Domain Evaluation

To test generalization, I evaluated trained models on **SuperCLEVR**—a more complex counting benchmark with different visual characteristics than the CLEVR training set.

![Cross-domain transfer results](/assets/img/posts/grpo/transfer_results.png)
*Figure 5: Cross-domain transfer to SuperCLEVR reveals a stark tradeoff with weak baseline. Injection hurts transfer by 18 percentage points.*

The results were surprising:

| Configuration | CLEVR (In-Domain) | SuperCLEVR (Cross-Domain) |
|:--------------|:-----------------:|:-------------------------:|
| Default + Injection | 93% | 30% |
| Default + No Injection | 93% | **48%** |

Despite identical in-domain performance, the model trained *without* injection generalized **18 percentage points better** to the new domain.

### Why Does Injection Hurt Generalization?

The likely explanation relates to what the model learns:

**With injection on weak baseline:**
- Model sees: CLEVR images → many injected "just the number" responses
- Model learns: "CLEVR-style images → output format change to single digit"
- This couples the format change to the visual domain characteristics

**Without injection on weak baseline:**
- Model discovers the digit format through its own exploration
- The format learning is gradual and not tied to specific visual features
- The learned behavior is more domain-agnostic

In essence, injection teaches the model a **shortcut**: "these images → this format." Without injection, the model learns a more general principle: "counting questions → numeric answer."

## The Baseline Effect: Does Prompt Engineering Eliminate the Tradeoff?

The observations above used the Default prompt with its 14% baseline digit rate. But what if we could start from a stronger baseline?

### The R1V Prompt Style

The R1V prompt style uses explicit format instructions:

```
System: A conversation between User and Assistant. The assistant first
        thinks about the reasoning process and then provides the answer.
        The reasoning process and answer are enclosed within <think></think>
        and <answer></answer> tags.

User: How many items are there in the image? Output the thinking process
      in <think></think> and final answer (number) in <answer></answer> tags.
```

This achieves **~99% digit extraction rate** on the base model—a dramatically stronger starting point.

### Does Injection Still Help?

With a strong baseline, the picture changes substantially:

![Learning curves with R1V prompt](/assets/img/posts/grpo/learning_curves_r1v.png)
*Figure 6: With R1V prompt (strong baseline), both injection and no-injection methods converge quickly. The gap between them is much smaller.*

Key observations:
1. **Both methods start strong**: ~68-70% accuracy from step 1
2. **Both converge quickly**: High performance achieved rapidly
3. **The gap is much smaller**: No dramatic speedup from injection

### Cross-Domain Transfer with Strong Baseline

Perhaps most importantly, the generalization tradeoff appears to diminish:

| Configuration | SuperCLEVR Accuracy | Training Steps |
|:--------------|:-------------------:|:--------------:|
| R1V + Injection | 74.5% | 1,574 |
| R1V + No Injection | 71.5% | 776* |

*Note: The no-injection model trained for fewer steps, making direct comparison difficult.*

Both methods achieve >70% on SuperCLEVR—dramatically better than either method with the weak baseline (30% and 48%).

![Injection effect by baseline strength](/assets/img/posts/grpo/injection_effect_summary.png)
*Figure 7: The injection effect depends on baseline strength. With weak baseline, injection hurts transfer by 18%. With strong baseline, both methods achieve similar transfer performance.*

### Interpreting the Results

The strong baseline appears to eliminate the problematic format-domain coupling:

**With weak baseline + injection:**
- Injection teaches: format change + correct answer
- Format change gets coupled to training domain
- Hurts transfer

**With strong baseline + injection:**
- Format is already handled by the prompt
- Injection only teaches: correct answer
- This knowledge is domain-agnostic

The key insight: **when the model already outputs the right format, injection can only teach accuracy—which transfers well.**

## Summary of Findings

The table below summarizes all configurations, separating weak and strong baseline results for clarity:

![Results comparison tables](/assets/img/posts/grpo/results_table_v2.png)
*Figure 8: Complete results organized by baseline strength. Green highlighting shows better transfer; pink shows worse transfer. Note the dramatic difference in SuperCLEVR accuracy between baselines.*

The bar chart below makes the cross-domain transfer comparison clearer:

![Results summary bar chart](/assets/img/posts/grpo/results_summary_bar.png)
*Figure 9: Cross-domain transfer (SuperCLEVR) by configuration. With weak baseline, injection hurts transfer by 18%. With strong baseline, both methods perform similarly well.*

### What We Observed

1. **GRPO works with sparse signals**: Even with only 14% of outputs scoreable, GRPO eventually learns through relative comparisons.

2. **Learning has two phases**: Format learning precedes accuracy learning when baseline digit rate is low.

3. **Guidance Injection accelerates learning**: With weak baseline, injection provides ~3× speedup to reach 90% accuracy.

4. **Injection can hurt generalization**: With weak baseline, injection couples format learning to the training domain, reducing cross-domain transfer by 18 percentage points.

5. **Strong baseline changes everything**: With R1V prompt (99% digit rate), both methods achieve strong transfer performance (>70% on SuperCLEVR).

6. **Prompt engineering may be more important than training tricks**: The choice of prompt style had a larger impact on final performance than the choice of injection strategy.

### Caveats and Limitations

These findings come with important limitations:

- **Single task**: All experiments used counting tasks. Other tasks may show different patterns.
- **Single model**: Results are specific to Qwen2-VL-2B-Instruct. Larger models may behave differently.
- **Training duration differences**: The R1V experiments had different training durations, making some comparisons uncertain.
- **Limited evaluation**: SuperCLEVR is still a synthetic benchmark. Real-world generalization may differ.
- **Small scale**: These experiments used hundreds to thousands of training steps, not the millions typical of production systems.

### The Key Mechanism

The figure below summarizes why baseline strength matters so much:

![Key takeaway mechanism](/assets/img/posts/grpo/key_takeaway.png)
*Figure 10: The mechanism behind the results. With weak baseline, injection teaches format + accuracy together, coupling them to the training domain. With strong baseline, format is handled by the prompt, so both methods teach only accuracy—which transfers well.*

## Practical Recommendations

Based on these observations, here are some tentative recommendations:

### Decision Framework

```
Can you use explicit format instructions in your prompt?
│
├── YES → Use strong baseline (R1V-style)
│         └── Injection optional (minor effect either way)
│
└── NO → Weak baseline path
         ├── Single domain only? → Use injection (faster training)
         └── Need generalization? → Skip injection (better transfer)
```

### When to Use Guidance Injection

**Consider injection when:**
- Training time is limited
- Single-domain deployment
- Base model rarely produces scoreable outputs
- Exploration is not yielding reward signal

**Consider skipping injection when:**
- Cross-domain transfer is important
- You can use explicit format instructions
- Training time is not a constraint

## Future Directions

Several questions remain unexplored:

1. **Injection scheduling**: Could we inject heavily early (for format learning) then taper off?
2. **Partial injection**: Does the injection rate affect the format-domain coupling?
3. **Other prompt styles**: Are there prompts that achieve high digit rate without the verbosity of R1V?
4. **Other tasks**: Do these patterns hold for non-counting tasks?

## Conclusion

Building GRPO from scratch for VLM fine-tuning revealed that the details matter enormously. The same algorithm (GRPO) with the same data (CLEVR) can produce models with vastly different generalization capabilities depending on prompt style and injection strategy.

The most surprising finding was that prompt engineering—a simple, cost-free intervention—had a larger impact on final performance than sophisticated training modifications. With the R1V prompt, even the base model achieved 68% on SuperCLEVR, and fine-tuning pushed this above 70% regardless of injection strategy.

This suggests a practical hierarchy: **optimize your prompts first, then worry about training tricks.**

<div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
<h4 style="color: #333;">📦 VLM GRPO Repository</h4>
<p style="color: #333;">GRPO implementation for fine-tuning Vision-Language Models. Includes configurable experiments, guidance injection, and cross-domain evaluation.</p>
<a href="https://github.com/suyashh94/vlm-grpo" target="_blank" style="display: inline-block; background-color: #333; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">View on GitHub →</a>
</div>

---

*Thanks for reading! If you have questions or thoughts, feel free to reach out or open an issue on the repository.*
