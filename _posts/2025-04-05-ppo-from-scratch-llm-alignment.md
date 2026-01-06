---
title: "PPO from Scratch: Implementing Proximal Policy Optimization for LLM Alignment"
date: 2025-04-05 10:00:00 +0530
categories: [Side Projects]
tags: [reinforcement-learning, ppo, transformers, llm, pytorch, alignment, deep-learning]
description: "A complete implementation of Proximal Policy Optimization for fine-tuning language models, with detailed explanations of GAE, clipped objectives, and the training dynamics that make RLHF work."
image:
  path: /assets/img/posts/ppo/ppo_cover.png
  alt: PPO from Scratch - Reinforcement Learning for Language Model Alignment
math: true
mermaid: true
pin: true
---

## Why Build PPO from Scratch?

Reinforcement Learning from Human Feedback (RLHF) has become the dominant paradigm for aligning large language models with human preferences. At the heart of most RLHF implementations sits Proximal Policy Optimization (PPO)—an algorithm that balances exploration with stability, enabling models to learn from reward signals without catastrophic policy collapse.

Libraries like TRL and trlX abstract away PPO's complexity. You call `PPOTrainer.step()` and trust the magic happens. But abstraction is the enemy of understanding. When training diverges, when KL penalties explode, when rewards plateau—you need to understand *why*.

This post walks through a complete PPO implementation for fine-tuning Qwen 1.5B using torchtune. Not a toy example. Not pseudocode. A working system with all the engineering details that tutorials omit: proper advantage normalization, value function clipping, KL penalty scheduling, and the mask arithmetic that makes token-level credit assignment work.

The results speak for themselves:

| Metric | Base Model | Aligned Model | Improvement |
|--------|------------|---------------|-------------|
| Average Reward | -0.935 | +9.847 | **+10.782** |
| EOS Rate | 0.0% | 100.0% | **+100%** |

The aligned model learns to generate positive sentiment completions and properly terminate responses—all through the PPO training signal.

---

## The Alignment Problem PPO Solves

Consider a language model completing the prompt: *"I think the book was..."*

The base model might continue with anything—positive sentiment, negative sentiment, factual description, tangential rambling. It has no preference. It simply models $P(\text{next token} | \text{context})$ based on its pre-training corpus.

We want to *steer* this distribution. Make the model prefer completions that a reward model scores highly. In our implementation, we use sentiment as the reward signal—positive sentiment yields positive rewards, negative sentiment yields penalties. We also reward for response length. If the generation is longer than 50 tokens, we penalize the reward with a penalty of -10. Overall, we want to reward short positive sentiment completions. 

The challenge: we can't just maximize reward greedily. A model that always outputs "Great! Amazing! Wonderful!" would score high on sentiment but become useless. We need to:

1. **Improve reward** while staying close to the original model's behavior
2. **Maintain diversity** in outputs rather than collapsing to a single pattern
3. **Learn stably** without the policy oscillating wildly between updates

PPO provides all three through its clipped objective and KL penalty mechanism.

---

## Architecture: Four Models Working Together

The PPO training setup requires **four model instances**, each serving a distinct role:

```
SetupQwenModel (base: loads weights, creates tokenizer/optimizer)
├── QwenModel (adds generation logic with/without KV cache)
│   ├── PolicyQwenModel (trainable, requires_grad=True)
│   └── ReferenceQwenModel (frozen, provides KL baseline)
└── QwenModelValueHead (adds linear value head for advantage estimation)
```

![PPO Architecture Diagram](/assets/img/posts/ppo/ppo_architecture.png)
*The four-model PPO architecture: Policy generates text, Reference provides KL baseline, Value estimates advantages, Reward scores completions.*

```python
trainer = PPOTrainer(
    policy_model=PolicyQwenModel(training_enabled=True, model_path=base_model_path),
    value_model=QwenModelValueHead(
        training_enabled=True, model_path=base_model_path, freeze_backbone=False
    ),
    reference_model=ReferenceQwenModel(model_path=base_model_path),
    reward_model=SentimentRewarder(),
)
```

### Policy Model
The model we're training. It generates text and gets updated via PPO. Gradients flow through this model during the policy loss computation.

### Reference Model
A frozen copy of the initial policy. It never updates. We use it to compute KL divergence—how far the current policy has drifted from the starting point. This prevents reward hacking where the model finds degenerate high-reward outputs that destroy language quality.

### Value Model  
A separate model (same architecture + linear head) that predicts expected future rewards from any state. The value head projects hidden states to a scalar:

```python
class QwenModelValueHead(SetupQwenModel):
    def __init__(self, model_path: str, ...):
        super().__init__(model_path=model_path, ...)
        self.hidden_size = self._get_hidden_size()
        self.value_head = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.get_last_hidden_state(input_ids)
        values = self.value_head(hidden_states)
        return values
```

### Reward Model
Scores completions. Our implementation uses `cardiffnlp/twitter-roberta-base-sentiment-latest`:

```python
class SentimentRewarder(nn.Module):
    LABEL_MAP = {
        "negative": -1.0,
        "neutral": 0.1,
        "positive": 10.0,
    }
    
    @torch.no_grad()
    def get_reward(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", ...).to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        dominant_idx = probs.argmax().item()
        dominant_label = self.id2label[dominant_idx].lower()
        confidence = probs[dominant_idx].item()
        
        base_reward = self.LABEL_MAP.get(dominant_label, 0.0)
        reward = base_reward * confidence  # Scale by confidence
        return {"reward": reward, "label": dominant_label, ...}
```

The asymmetric reward structure (positive: +10, negative: -1) encourages positive sentiment while not overly penalizing neutral or negative outputs. The confidence scaling adds nuance—a tentatively positive completion scores lower than an enthusiastically positive one. When calculating the reward, before passing through the reward model, we check for EOS token in the generations. If generation is missing the EOS token, we assign the reward of -10 to that generation. Only in the cases where EOS token is found, we pass the generations over to the reward model, for rewards to be caluculated. 

---

## Understanding the Mask Semantics

One of the trickiest aspects of PPO for language models is understanding *which tokens to update*. This requires careful mask arithmetic that respects the autoregressive nature of generation.

### Action-Based Indexing

The key insight: **masks indicate what type of token was generated, not token membership**.

- `response_mask[i]=1` means the action at position `i` (inputting token[i]) **generated a response token**
- The last prompt token (e.g., "was") has `response_mask=1` because it triggered the first response token
- EOS has `response_mask=0` because it terminates generation (no token generated after it)

![Mask Visualization](/assets/img/posts/ppo/mask_visualization.png)
*Visualization of prompt_mask, response_mask, and padding_mask alignment. The response starts when the last prompt token is input.*

Consider this example:

```
Prompt: "I think the book was"
Response: " really interesting and I enjoyed reading it.<EOS>"
```

The sequence after tokenization:
```
Position:  0    1    2     3     4     5      6     7      8    9    10   ...
Token:    [I] [think] [the] [book] [was] [really] [int...] [and] [I] [enjoyed] ...
```

The `response_mask` is 1 starting at position 4 ("was") because that's where the action of inputting "was" generates the first response token "really". The EOS token position has `response_mask=0` because no token is generated after it.

```python
def get_batched_response_properties(self, generated_outputs_list: list[dict], **kwargs):
    # ... padding and forward pass ...
    
    batch_size, seq_len = padded_generated_ids_tensor.size()
    prompt_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
    response_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
    
    for i in range(batch_size):
        prompt_length = input_lengths[i]
        if prompt_length > 0:
            prompt_mask[i, : prompt_length - 1] = 1
        response_mask[i, prompt_length - 1 : response_id_lengths[i] - 1] = 1
    
    return {
        "logprobs": log_probs,
        "prompt_mask": prompt_mask,
        "response_mask": response_mask,
        # ...
    }
```

### Reward Alignment

Rewards are sparse—they come only at the end of generation. But PPO needs to assign credit to individual tokens. We place the reward at the last response action position:

```python
def collate_rewards_tensor(self, rewards, response_id_lengths, max_len):
    batch_size = rewards.size(0)
    rewards_tensor = torch.zeros((batch_size, max_len), device=rewards.device)
    
    for i in range(batch_size):
        response_length = response_id_lengths[i]
        # Place reward at last response action position
        reward_position = max(0, response_length - 2)
        rewards_tensor[i, reward_position] = rewards[i]
    
    return rewards_tensor
```

![Reward Alignment](/assets/img/posts/ppo/reward_alignment.png)
*The reward is placed at position response_length - 2, aligned with the last response action in the shifted indexing.*

The `-2` offset accounts for:
1. The shift from token positions to action positions
2. The fact that EOS doesn't have a response_mask (no token generated after it)

---

## KL Penalty: Staying Close to the Reference

The KL divergence between the current policy and reference policy acts as a regularizer, preventing the model from drifting too far in pursuit of reward:

```python
kl_diff = policy_response_properties["logprobs"] - policy_response_properties["ref_logprobs"]
kl_penalty = self.ppo_config.kl_coef * kl_diff

total_rewards_aligned = total_rewards_aligned - kl_penalty
```

This is the *per-token* KL penalty. Positive values mean the current policy assigns higher probability to this token than the reference did—the policy is becoming more confident. The penalty gets subtracted from rewards, discouraging deviation from the reference distribution.

```python
@dataclass
class PPOConfig:
    kl_coef: float = 0.02
    target_kl: float | None = 0.015
    max_kl: float = 0.05
```

The `max_kl` provides early stopping—if KL divergence exceeds this threshold during an epoch, we stop to prevent instability.

---

## Generalized Advantage Estimation (GAE)

The advantage function $A(s, a)$ measures how much better action $a$ is compared to the average action in state $s$. GAE provides a bias-variance tradeoff in estimating this quantity.

### The Intuition

Consider a completion that receives reward +5 at the end. Which tokens deserve credit? The final token that triggered the positive sentiment? The opening phrase that set the tone? All of them equally?

GAE answers this by propagating rewards backward through time, discounted by $\gamma$ (how much we care about future rewards) and $\lambda$ (how much we trust the value function's estimates):

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the temporal difference error.

### The Implementation

```python
def compute_gae(self, rewards, values, masks):
    batch_size, seq_len = rewards.size()
    advantages = torch.zeros((batch_size, seq_len), device=rewards.device)
    last_gae_lam = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_non_terminal = 0.0
            next_values = 0.0
        else:
            next_non_terminal = masks[:, t + 1]
            next_values = values[:, t + 1]

        delta = (
            rewards[:, t]
            + self.ppo_config.gamma * next_values * next_non_terminal
            - values[:, t]
        )
        advantages[:, t] = last_gae_lam = (
            delta
            + self.ppo_config.gamma * self.ppo_config.gae_lambda 
            * next_non_terminal * last_gae_lam
        )

    return advantages
```

The backward iteration is essential—we need to know future advantages to compute current ones.

### Why γ = 1.0?

```python
@dataclass
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
```

For text generation, we typically use $\gamma = 1.0$ (no discounting). There's no reason to value early tokens over late tokens. The reward is sparse—it comes at the end—and we want full credit propagation.

---

## The PPO Objective: Clipped Surrogate Loss

The core of PPO is the clipped surrogate objective.

### Probability Ratios

First, we compute how much more (or less) likely the current policy makes each action compared to the policy that generated the data:

```python
log_ratio = logprobs - mb_old_logprobs
log_ratio = torch.clamp(log_ratio, -20.0, 20.0)  # Prevent exp overflow
ratios = torch.exp(log_ratio)
```

### The Clipping Trick

Unconstrained, the policy gradient objective can make arbitrarily large updates. PPO clips the ratio to prevent updates that are "too large":

```python
surr1 = ratios * mb_advantages
surr2 = (
    torch.clamp(
        ratios,
        1.0 - self.ppo_config.clip_epsilon,
        1.0 + self.ppo_config.clip_epsilon,
    )
    * mb_advantages
)
policy_loss = -torch.min(surr1, surr2)
```

With `clip_epsilon = 0.2`, the ratio is clamped to `[0.8, 1.2]`. The `min` operation means:
- For positive advantages: we want to increase the ratio, but stop at 1.2
- For negative advantages: we want to decrease the ratio, but stop at 0.8

This creates a "trust region" around the old policy.

### Advantage Normalization

Before computing the policy loss, we normalize advantages:

```python
if self.ppo_config.normalize_advantages:
    advantages = whiten(advantages, response_mask)
```

```python
@torch.no_grad()
def whiten(xs, mask, shift_mean=True):
    masked_xs = xs * mask.float()
    n_valid = mask.sum().clamp(min=1)
    mean = masked_xs.sum() / n_valid
    var = torch.sum(((xs - mean) ** 2) * mask.float()) / n_valid
    
    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    whitened = torch.clamp(whitened, -10.0, 10.0)
    
    return whitened
```

Normalization stabilizes training by ensuring advantages have consistent scale across batches.

---

## Value Function Learning

The value model learns to predict expected returns:

```python
values = self.value_model(input_ids=mb_padded_generated_ids).squeeze(-1)

# Use RAW advantages (not whitened) for computing returns
returns = mb_raw_advantages.detach() + mb_value_preds.detach()
value_losses1 = (values - returns) ** 2
value_losses2 = (value_pred_clipped - returns) ** 2
value_loss = torch.max(value_losses1, value_losses2)
```

Note: we use *raw* (unnormalized) advantages when computing returns. Whitened advantages have mean zero by construction, which would bias the value function.

---

## Training Results

The training produces clear learning curves showing the model's progression from random to aligned behavior.

![Learning Curves](/assets/img/posts/ppo/learning_curves.png)
*PPO training metrics over 10,000 steps. Top-left: Mean reward improves from -10 to +9.8. Top-right: Policy and value losses remain stable. Bottom-left: KL divergence stays bounded. Bottom-right: EOS rate increases from 0% to 100%.*

### Key Observations

1. **Mean Reward**: Starts at -10 (no EOS penalty) and climbs to +9.8 (positive sentiment with proper termination)

2. **EOS Rate**: The base model never produces EOS tokens (0% rate). After training, 100% of completions properly terminate.

3. **KL Divergence**: Stays bounded throughout training, indicating the policy doesn't drift too far from the reference.

4. **Clip Fraction**: Remains low, suggesting updates stay within the trust region.

---

## Base vs Aligned Model Comparison

The proof is in the generations. Here's a direct comparison:

| Prompt | Base Model | Aligned Model |
|--------|------------|---------------|
| "I think the book was" | "_______ interesting _______ I couldn't put it down. such,that so..." (no EOS, reward: -0.74) | "really interesting and I enjoyed reading it." (EOS, reward: +9.86) |
| "The movie was" | "released in 1999 and starred Tom Hanks..." (no EOS, reward: -2.26) | "very entertaining and I enjoyed watching it." (EOS, reward: +9.87) |
| "Overall, the product is" | "solid for its price point. There are a few areas..." (no EOS, reward: -1.12) | "excellent." (EOS, reward: +9.83) |

The base model generates verbose, meandering completions without termination. The aligned model produces concise, positive responses that properly end with EOS.

![Model Comparison](/assets/img/posts/ppo/model_comparison.png)
*Side-by-side comparison of base and aligned model outputs. The aligned model consistently produces positive sentiment and proper EOS termination.*

---

## Interactive Walkthrough

For a detailed, step-by-step exploration of the PPO training pipeline, I've created an interactive Jupyter notebook that visualizes:

- Mask semantics and reward alignment
- Token-level credit assignment
- Learning curve analysis
- Base vs aligned model comparisons

<iframe 
  src="/assets/html/ppo_walkthrough.html" 
  width="100%" 
  height="800px" 
  style="border: 1px solid #ddd; border-radius: 8px;"
  loading="lazy">
</iframe>

*Interactive walkthrough of the PPO training pipeline. Scroll through to see mask visualizations, reward alignment, and training metrics.*

---

## Key Implementation Details

### Mixed Precision Training

```python
self.use_amp = self.device.type == "cuda"
self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

with torch.amp.autocast("cuda", enabled=self.use_amp):
    logits = self.policy_model(input_ids=mb_padded_generated_ids)
    # ... compute losses

self.scaler.scale(total_loss).backward()
self.scaler.unscale_(self.policy_model.optimizer)
torch.nn.utils.clip_grad_norm_(...)
self.scaler.step(self.policy_model.optimizer)
self.scaler.update()
```

### Numerical Stability

```python
# Clamp log ratios to prevent exp overflow
log_ratio = torch.clamp(log_ratio, -20.0, 20.0)

# Clamp logits for numerical stability in softmax
logits = torch.clamp(logits, -100.0, 100.0)

# Clamp whitened advantages
whitened = torch.clamp(whitened, -10.0, 10.0)
```

### Early Stopping on KL

```python
for _ in range(self.ppo_config.num_ppo_epochs):
    epoch_kl_divs = []
    for minibatch in minibatches:
        # ... training step
        epoch_kl_divs.append(approx_kl.item())
    
    epoch_kl = sum(epoch_kl_divs) / len(epoch_kl_divs)
    if epoch_kl > self.ppo_config.max_kl:
        break  # Early stopping
```

---

## Configuration Reference

```python
@dataclass
class PPOConfig:
    clip_epsilon: float = 0.2
    vf_clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    num_ppo_epochs: int = 4
    kl_coef: float = 0.02
    target_kl: float | None = 0.015
    max_kl: float = 0.05

@dataclass
class TrainingConfig:
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps_frac: float = 0.1
    lr_scheduler_type: str = "cosine"
    total_steps: int = 10000
    num_rollouts_per_update: int = 8
    minibatch_size: int = 4
```

---

## What This Implementation Teaches

Building PPO from scratch reveals patterns invisible in library abstractions:

1. **Mask arithmetic is everywhere.** Every tensor operation must respect prompt/response/padding boundaries. One wrong index offset corrupts gradients.

2. **The value function matters as much as the policy.** GAE quality depends on value predictions. A poorly trained value function leads to high-variance advantages.

3. **KL divergence is your stability anchor.** Without it, the policy can drift arbitrarily far from coherent language.

4. **Multiple epochs are safe because of clipping.** The trust region allows sample reuse without divergence.

5. **Numerical stability requires constant attention.** Log-probability differences can overflow. Advantage normalization can divide by zero. Gradient norms can explode.

---

## Repository

The complete implementation is available on GitHub:

<div style="border: 2px solid #333; border-radius: 10px; padding: 20px; margin: 20px 0; background-color: #f5f5f5;">
<h4 style="color: #333;">📦 Qwen PPO Tuning Repository</h4>
<p style="color: #333;">PPO implementation for fine-tuning Qwen 1.5B with torchtune. Includes configurable hyperparameters, mixed precision training, and comprehensive metrics logging.</p>
<a href="https://github.com/suyashh94/qwen-ppo-tuning" target="_blank" style="display: inline-block; background-color: #333; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">View on GitHub →</a>
</div>

```bash
# Clone and setup
git clone https://github.com/suyashh94/qwen-ppo-tuning
cd qwen-ppo-tuning
pip install -e .

# Download base model
python -m scripts.download_base_model --output-dir ./base_models

# Run training
python -m qwen_ppo_tuning.ppo_trainer
```

---

## Further Reading

- **PPO Paper**: [Schulman et al., 2017 - Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE Paper**: [Schulman et al., 2015 - High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **InstructGPT**: [Ouyang et al., 2022 - Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- **TRL Library**: [HuggingFace TRL Documentation](https://huggingface.co/docs/trl)
- **The 37 Implementation Details of PPO**: [Huang et al., 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

---

*Questions about PPO implementation details or RLHF in general? Reach out on [LinkedIn](https://linkedin.com/in/suyash94) or open an issue on the repository.*