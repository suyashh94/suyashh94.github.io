---
title: "Fine-Tuning Small Language Models for Structured Function Calling"
date: 2024-12-13 10:00:00 +0530
categories: [Projects]
tags: [transformers, fine-tuning, function-calling, slm, lora, pytorch, nlp]
description: "Fine-tuning GPT-2 (124M parameters) to achieve 94% accuracy on structured function calling for automotive controls using LoRA and synthetic data."
image:
  path: /assets/img/posts/slm-fc/slm_fc_cover.png
  alt: SLM Function Calling - Fine-tuning Small Models for Structured Output
math: true
pin: true
---

In late 2024, I worked on a project for an automotive manufacturer involving natural language control of vehicle systems. The core requirement: convert spoken commands into structured function calls that could execute on edge hardware without network connectivity.

The production system and data remain under NDA, but the techniques are general-purpose. This post walks through a complete implementation using synthetic data—the same architecture and training approach, applied to publicly reproducible examples.

The result: a 124 million parameter GPT-2 model achieving **94.27% accuracy** on 18 car control functions, with inference latency under 50ms on consumer hardware.

---

## Function Calling with Large Language Models

The standard approach to function calling with LLMs works as follows:

1. **Embed function definitions in the prompt** — Describe available functions, their parameters, and expected formats
2. **Send user query + function definitions** — Every request includes the full schema
3. **Parse structured output** — Extract JSON from the model's response

This works for general-purpose applications. The limitations become apparent in constrained environments:

- **Latency**: 200-500ms per API call minimum
- **Token costs**: Function definitions consume thousands of tokens per request
- **Connectivity**: Requires network access
- **Consistency**: LLMs occasionally hallucinate functions or return malformed JSON

For automotive systems processing voice commands at 70mph, these constraints are disqualifying.

---

## Small Language Models for Constrained Tasks

The alternative: train a small model to do one thing well.

GPT-4 can write poetry, debug code, and explain quantum physics—capabilities irrelevant when the task is mapping *"turn up the heat"* to `set_temperature(action='increase')`.

| Model | Parameters | Use Case |
|-------|------------|----------|
| GPT-4 | ~1.76T | General purpose |
| LLaMA 2 | 7-70B | General purpose (open) |
| Phi-2 | 2.7B | Efficient reasoning |
| **GPT-2** | **124M** | **This implementation** |

GPT-2's 124 million parameters represent roughly **14,000x fewer parameters** than GPT-4. For structured output on a fixed function schema, this is sufficient.

---

## How Fine-Tuning Works

Fine-tuning adjusts a pre-trained model's weights toward a specific behavior. The base model (GPT-2) already understands language structure. Fine-tuning teaches it to produce structured outputs for specific inputs.

### Training Signal

We show the model input-output pairs:

```
Input: "Set the temperature to 22 degrees"
Output: {"name": "set_temperature", "arguments": {"temperature": 22}}
```

The model generates predictions, we compute loss (how wrong the prediction was), and backpropagation adjusts weights to reduce loss.

### Label Masking

The critical detail is *where* loss is computed:

```
┌─────────────────────────┬──────────────────────────────────┐
│    SYSTEM + USER        │          ASSISTANT               │
│    (Context Only)       │     (Loss Computed Here)         │
│    Labels: -100         │     Labels: Actual Token IDs     │
└─────────────────────────┴──────────────────────────────────┘
```

System prompt and user message are **masked** (label `-100`, which PyTorch ignores). Loss is computed only on the **assistant response**—the function call.

This teaches: *"Given this input pattern, produce this output pattern."*

---

## LoRA: Parameter-Efficient Fine-Tuning

Full fine-tuning of GPT-2's 124 million parameters requires storing gradient history for each parameter and risks catastrophic forgetting of pre-trained knowledge.

**LoRA (Low-Rank Adaptation)** provides an alternative: freeze original weights and train small adapter matrices.

### The Mathematics

Instead of modifying the full weight matrix `W` (e.g., 768×768 = 589,824 parameters), LoRA adds two low-rank matrices:

$$W' = W + BA$$

Where:
- `B` is a 768×32 matrix (24,576 parameters)
- `A` is a 32×768 matrix (24,576 parameters)
- Rank `r = 32` is significantly smaller than the original dimension

This reduces trainable parameters by ~12x for a single layer. Across the entire model, LoRA trains less than 1% of total parameters.

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=32,                    # Rank of adaptation matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Layers to adapt
        "attn.c_attn",       # Attention projections
        "mlp.c_fc",          # Feed-forward layers
        "mlp.c_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, peft_config)
```

Benefits:
- **Memory efficient**: Store only small adapter weights
- **Fast training**: Fewer parameters = faster backpropagation
- **Modular**: Swap adapters for different tasks without reloading base model
- **Preservation**: Original weights remain frozen

---

## Dataset: 18 Functions, Linguistic Diversity

The function schema covers 18 car control operations:

| Function | Description | Example Parameters |
|----------|-------------|-------------------|
| `set_temperature` | Set zone temperature | temperature, area, unit |
| `adjust_fan_speed` | Increase/decrease fan | speed, area |
| `control_window` | Open/close windows | window_position, location |
| `set_navigation_destination` | Set GPS destination | destination |
| `toggle_headlights` | Turn lights on/off | light_state |
| `lock_doors` | Lock/unlock doors | lock_state |
| `set_cruise_control` | Set cruise speed | speed |
| `play_music` | Control music | track, volume |
| ... | *10 more functions* | ... |

The challenge is linguistic variation. Users don't say *"execute set_temperature with parameter temperature equals 22"*. They say:

- *"Make it warmer back here"*
- *"Crank up the heat"*
- *"I'm freezing, can you do something about that?"*
- *"The kids in the back are cold"*

The training set (~130,000 examples) was generated using template expansion and LLM-assisted paraphrasing, covering:

- Colloquial language ("crank up", "blast", "dial down")
- Implicit references ("back here" → rear seats)
- Multi-target commands ("both front seats")
- Politeness variations ("please", "could you", "would you mind")

---

## Training Pipeline

Training uses HuggingFace's TRL library:

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="outputs",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    max_length=1024,
    lr_scheduler_type="constant",
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,                    # LoRA-wrapped GPT-2
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    args=sft_config,
    data_collator=dataset.collate,  # Dynamic padding
)

trainer.train()
```

Training completed in approximately 2 hours on a single A100 GPU. The adapter weights total **19MB**.

---

## Evaluation Results

Test set: 5,200 held-out samples.

### Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **94.27%** |
| Function Match Rate | 99.00% |
| Argument Match Rate | 95.22% |
| Total Samples | 5,200 |
| Correct Predictions | 4,902 |

The model correctly identifies the right function 99% of the time. When it gets the function right, it extracts the correct arguments 95% of the time.

### Per-Function Breakdown

| Function | Accuracy | Samples |
|----------|----------|---------|
| set_cruise_control | 100.00% | 88 |
| toggle_headlights | 100.00% | 70 |
| toggle_sport_mode | 100.00% | 67 |
| adjust_wiper_speed | 100.00% | 59 |
| set_wiper_speed | 98.18% | 55 |
| adjust_temperature | 96.73% | 1,008 |
| set_temperature | 95.79% | 975 |
| adjust_fan_speed | 95.21% | 1,003 |
| set_fan_speed | 94.41% | 1,001 |
| set_navigation_destination | 94.74% | 76 |
| activate_defroster | 92.05% | 88 |
| lock_doors | 89.83% | 59 |
| start_engine | 89.66% | 58 |
| control_window | 87.78% | 90 |
| control_ambient_lighting | 87.36% | 87 |
| adjust_seat | 83.64% | 269 |
| play_music | 80.22% | 91 |
| check_battery_health | 78.57% | 56 |

---

## Example Predictions

### Colloquial Language

**Input:** *"Crank up the fan speed a bit"*

**Prediction:** `{"name": "adjust_fan_speed", "arguments": {"speed": "increase"}}`

Maps "crank up" → increase without explicit training on that exact phrase.

---

**Input:** *"Make the air blow softer"*

**Prediction:** `{"name": "adjust_fan_speed", "arguments": {"speed": "decrease"}}`

"Softer" requires understanding the semantic relationship to reduced intensity.

---

### Multi-Zone Handling

**Input:** *"Crank up the airflow for the driver's side and the back left"*

**Prediction:** `{"name": "adjust_fan_speed", "arguments": {"speed": "increase", "area": ["driver", "rear-left"]}}`

Correctly identifies two separate zones from natural language descriptions.

---

### Politeness Markers

**Input:** *"Can you adjust the cruising speed to 66 mph, please?"*

**Prediction:** `{"name": "set_cruise_control", "arguments": {"speed": 66}}`

Handles question format and politeness markers without confusion.

---

## Failure Modes

### Complex Multi-Zone References

**Input:** *"Turn up the fan speed for the driver, back left, front passenger, and back right seats"*

**Ground Truth:** `{"area": ["driver", "front-passenger", "rear-left", "rear-right"]}`

**Prediction:** `{"area": ["front-passenger", "rear-left", "rear-right"]}` — Missing "driver"

When users enumerate many zones explicitly, the model occasionally drops one element. This is a known limitation of autoregressive generation.

### Negation

**Input:** *"Please decrease the air flow for everyone except the front driver"*

**Ground Truth:** `{"area": ["front-passenger", "rear-left", "rear-right"]}`

**Prediction:** `{"area": ["driver", "front-passenger"]}` — Incorrect interpretation

Negation ("except", "but not", "excluding") requires reasoning that small models handle inconsistently.

### Ambiguous Function Selection

**Input:** *"Adjust the fan to blow more air for the driver and the rear-right seat"*

**Ground Truth:** `adjust_fan_speed`

**Prediction:** `possibly_incomplete_set_fan_speed` — Wrong function

In ~1% of cases, the model selects a related but incorrect function when linguistic patterns overlap.

---

## Addressing Limitations

These failures are data problems with engineering solutions:

**More diverse training data**: Multi-zone failures indicate insufficient coverage of enumeration patterns.

**Negation-specific examples**: Targeted examples with "except", "but not", "excluding" patterns.

**Larger model**: Phi-2 (2.7B parameters) would likely handle edge cases better while remaining deployable on edge hardware.

**Confidence thresholds**: Low-confidence predictions can trigger clarification requests in production.

---

## Deployment Comparison

| Approach | Latency | Cost/1000 Requests | Internet Required | Privacy |
|----------|---------|-------------------|-------------------|---------|
| GPT-4 API | 500ms+ | $30-60 | Yes | Data leaves device |
| GPT-3.5 API | 200ms+ | $2-4 | Yes | Data leaves device |
| **Fine-tuned GPT-2** | **<50ms** | **$0** | **No** | **Data stays local** |

For systems processing thousands of voice commands daily, the fine-tuned SLM is the only viable option for edge deployment.

---

## Try It Out

<div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">
<h4 style="color: #333;">🚗 SLM Function Calling Demo</h4>
<p style="color: #333;">Enter natural language car commands and see the model's structured function call predictions in real-time.</p>
<a href="https://huggingface.co/spaces/suyash94/slm-function-calling" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Launch Demo →</a>
</div>

---

## Repository

The complete implementation is open-sourced:

- Training pipeline with LoRA configuration
- Evaluation framework with detailed error classification
- 18 function schemas for car control domain
- Inference utilities for deployment

<div style="border: 2px solid #333; border-radius: 10px; padding: 20px; margin: 20px 0; background-color: #f5f5f5;">
<h4 style="color: #333;">📦 SLM Function Calling Repository</h4>
<p style="color: #333;">Full source code, training scripts, and deployment utilities.</p>
<a href="https://github.com/suyashh94/slm-function-calling" target="_blank" style="display: inline-block; background-color: #333; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">View on GitHub →</a>
</div>

```bash
# Clone and setup
git clone https://github.com/suyashh94/slm-function-calling
cd slm-function-calling

# Train a model
fc-train --config configs/default.yaml configs/gpt2.yaml

# Run inference
fc-infer --adapter-path outputs/gpt2-fc-adapter

# Evaluate
fc-eval --adapter-path outputs/gpt2-fc-adapter --config configs/default.yaml
```

---

## Summary

A 124M parameter GPT-2 model, fine-tuned with LoRA on ~130K synthetic examples, achieves 94% accuracy on structured function calling for 18 automotive control functions.

Key takeaways:

- **Task-specific fine-tuning outperforms prompting** for constrained output schemas
- **LoRA reduces training cost** to ~1% of full fine-tuning while preserving base model knowledge
- **Small models are sufficient** when the output space is well-defined
- **Failure modes are addressable** with additional training data and confidence thresholds

For latency-critical, privacy-sensitive, or offline applications, fine-tuned small language models remain a practical alternative to large model APIs.

---

## References

- **Repository**: [github.com/suyashh94/slm-function-calling](https://github.com/suyashh94/slm-function-calling)
- **Demo**: [huggingface.co/spaces/suyash94/slm-function-calling](https://huggingface.co/spaces/suyash94/slm-function-calling)
- **LoRA Paper**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **HuggingFace PEFT**: [Documentation](https://huggingface.co/docs/peft)
- **From First Principles: Building Function Calling by Fine-tuning NanoGPT**: [TowardsAI](https://towardsai.net/p/machine-learning/from-first-principles-building-function-calling-by-fine-tuning-nanogpt)