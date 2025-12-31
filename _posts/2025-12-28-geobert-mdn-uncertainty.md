---
title: "Beyond Point Predictions: Adding Uncertainty Quantification to GeoBERT with Mixture Density Networks"
date: 2025-12-28 10:00:00 +0530
categories: [Side Projects]
tags: [bert, transformers, geocoding, mdn, uncertainty, pytorch, probabilistic-ml,deep learning, statistics]
description: "Extending GeoBERT with Mixture Density Networks to provide calibrated uncertainty estimates alongside coordinate predictions—enabling production systems to know when they don't know."
image:
  path: /assets/img/posts/geobert/geobert_mdn_cover.png
  alt: GeoBERT MDN - Probabilistic Geocoding with Uncertainty Quantification
math: true
mermaid: true
pin: true
---

## The Prediction Isn't Enough

In my [previous post on GeoBERT](/posts/geobert-geocoding-transformers/), I described how we trained a BERT-based model to geocode addresses with higher accuracy than Google Maps for our hyperlocal delivery use case. The model worked remarkably well, predicting coordinates from address text and saving thousands of dollars monthly in API costs.

But there was a problem we hadn't solved.

Every prediction came with the same level of confidence—or rather, no confidence at all. When the model predicted coordinates for "350 5th Avenue, Manhattan, NY 10118" (the Empire State Building), it returned a point. When it predicted coordinates for "near the old temple, behind Sharma ji's shop, Sector 15" (an ambiguous address that could refer to multiple locations), it also returned a point. Both predictions looked equally certain, even though the model's internal state was fundamentally different in each case.

In production, this mattered. Our dispatch system needed to know: *Can I trust this prediction enough to route a delivery partner, or should this order go to a human for manual geocoding?*

The regression model couldn't answer that question. It could only say *where*. It couldn't say *how sure*.

---

## What Regression Models Miss

The standard approach to geocoding with deep learning treats it as a regression problem: given an address, output two numbers (latitude, longitude). The loss function—mean squared error—optimizes for the average case, learning to predict the conditional mean of the target distribution.

$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

This works well when the mapping from input to output is deterministic and well-defined. But addresses aren't always well-defined.

Consider these scenarios:

**Ambiguous references**: "Near Central Park" could refer to dozens of valid locations along the park's perimeter. The regression model picks one point—probably somewhere near the center of mass of training examples—but that single point doesn't capture the inherent uncertainty.

**Sparse training data**: Some neighborhoods have fewer delivery records than others. The model has less evidence for these areas, but its predictions look just as confident as predictions for well-covered zones.

**Novel address patterns**: When an address uses phrasing the model hasn't seen before, it extrapolates. The prediction might be reasonable or wildly off. The regression model gives no indication which.

**Duplicate or similar addresses**: Many cities have multiple streets with similar names. "Main Street" exists in nearly every jurisdiction. Without disambiguating context, the model is guessing—but its output doesn't reflect that uncertainty.

In all these cases, the model has information about its own uncertainty. That information is encoded in its internal representations. But the regression head discards it, collapsing everything to a single point estimate.

---

## Mixture Density Networks: Predicting Distributions, Not Points

The solution is to change what the model outputs. Instead of predicting coordinates directly, we predict the *parameters of a probability distribution* over coordinates. This is the core idea behind Mixture Density Networks (MDNs), introduced by Christopher Bishop in 1994.

An MDN outputs a mixture of Gaussian distributions:

$$p(y|x) = \sum_{k=1}^{K} \pi_k(x) \cdot \mathcal{N}(y | \mu_k(x), \sigma_k(x))$$

Where:
- $K$ is the number of mixture components
- $\pi_k(x)$ are the mixture weights (summing to 1)
- $\mu_k(x)$ are the component means
- $\sigma_k(x)$ are the component standard deviations

All parameters are functions of the input $x$—in our case, the address text. The model learns to output different distributions for different addresses.

For geocoding, this means:
- **Clear addresses** → narrow distribution (low $\sigma$), single dominant component
- **Ambiguous addresses** → wide distribution (high $\sigma$), or multiple components representing alternative interpretations
- **Novel patterns** → larger $\sigma$ reflecting extrapolation uncertainty

The key insight is that uncertainty becomes a first-class output of the model, not something we have to infer post-hoc.

---

## Architecture: From Regression Head to MDN Head

The change from regression to MDN is surprisingly minimal. We keep the entire BERT encoder unchanged—it still produces a 128-dimensional [CLS] embedding capturing the semantic meaning of the address. We only replace the final regression head.

### Regression Architecture (Original)

```
Address Text
    ↓
BERT Encoder → [CLS] Embedding (128-dim)
    ↓
Linear(256) → ReLU → Linear(2)
    ↓
[latitude, longitude]
```

### MDN Architecture (Extended)

```
Address Text
    ↓
BERT Encoder → [CLS] Embedding (128-dim) → ReLU
    ↓
Three parallel heads:
    ├── π head: Linear(K) → Softmax     [mixture weights]
    ├── μ head: Linear(K×2)             [means for lat/lon]
    └── σ head: Linear(K×2) → Softplus  [std devs for lat/lon]
    ↓
K Gaussian mixture components
```

The implementation adds three linear layers branching from the shared BERT representation:

```python
class GeoBERTMDNModel(GeoBERTModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.num_mixtures = config.mdn_num_mixtures  # K = 5 by default
        
        # Three parallel heads
        self.pi_head = nn.Linear(self.bert_config.hidden_size, self.num_mixtures)
        self.mu_head = nn.Linear(self.bert_config.hidden_size, self.num_mixtures * 2)
        self.sigma_head = nn.Linear(self.bert_config.hidden_size, self.num_mixtures * 2)
        
        self.softplus = nn.Softplus()  # Ensures σ > 0
        self.sigma_eps = 1e-6  # Numerical stability
    
    def forward(self, input_ids, attention_mask):
        # Get BERT [CLS] embedding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        hidden = F.relu(cls_embedding)
        
        # Mixture weights (sum to 1 via softmax)
        pi_logits = self.pi_head(hidden)  # (batch, K)
        
        # Means for each component
        mu = self.mu_head(hidden).view(-1, self.num_mixtures, 2)  # (batch, K, 2)
        mu_lat, mu_lon = mu[:, :, 0], mu[:, :, 1]
        
        # Standard deviations (always positive via softplus)
        sigma = self.sigma_head(hidden).view(-1, self.num_mixtures, 2)
        sigma_lat = self.softplus(sigma[:, :, 0]) + self.sigma_eps
        sigma_lon = self.softplus(sigma[:, :, 1]) + self.sigma_eps
        
        return pi_logits, mu_lat, mu_lon, sigma_lat, sigma_lon
```

The model now outputs five tensors instead of one. For each address, we get $K$ mixture components, each with its own mean (predicted location) and standard deviation (uncertainty).

---

## Training: Maximum Likelihood with Gaussian Mixtures

With regression, we minimized mean squared error. With MDN, we maximize the likelihood of the true coordinates under the predicted distribution.

The negative log-likelihood loss for a mixture of Gaussians:

$$\mathcal{L}_{MDN} = -\log \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(y_{true} | \mu_k, \sigma_k)$$

Expanded for our 2D case (latitude and longitude treated as independent):

$$\mathcal{L}_{MDN} = -\log \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(lat_{true} | \mu_k^{lat}, \sigma_k^{lat}) \cdot \mathcal{N}(lon_{true} | \mu_k^{lon}, \sigma_k^{lon})$$

In code, we use the log-sum-exp trick for numerical stability:

```python
class MDNLoss(nn.Module):
    def forward(self, pi_logits, mu_lat, mu_lon, sigma_lat, sigma_lon, 
                target_lat, target_lon):
        # Log mixture weights
        log_pi = F.log_softmax(pi_logits, dim=1)  # (batch, K)
        
        # Log probability under each Gaussian component
        log_prob_lat = -0.5 * (
            torch.log(2 * torch.pi * sigma_lat**2) + 
            ((target_lat.unsqueeze(1) - mu_lat) / sigma_lat)**2
        )
        log_prob_lon = -0.5 * (
            torch.log(2 * torch.pi * sigma_lon**2) + 
            ((target_lon.unsqueeze(1) - mu_lon) / sigma_lon)**2
        )
        
        # Joint log probability (assuming independence)
        log_prob_joint = log_prob_lat + log_prob_lon  # (batch, K)
        
        # Weighted by mixture coefficients
        log_weighted = log_pi + log_prob_joint  # (batch, K)
        
        # Log-sum-exp for numerical stability
        loss = -torch.logsumexp(log_weighted, dim=1).mean()
        return loss
```

This loss function encourages the model to:
1. Place high-weight mixture components ($\pi_k$) near true locations
2. Set appropriate $\sigma$ values—small when confident, large when uncertain
3. Use multiple components when addresses are genuinely ambiguous

---

## From Distribution to Prediction

At inference time, we have a full probability distribution. For applications that need a single point estimate, we take the mean of the highest-weight mixture component:

```python
def sample_mdn(pi_logits, mu_lat, mu_lon, sigma_lat, sigma_lon, 
               deterministic=True):
    # Get mixture weights
    pi = F.softmax(pi_logits, dim=-1)
    
    if deterministic:
        # Select highest-weight component
        component_idx = torch.argmax(pi, dim=-1)
    else:
        # Sample component according to weights
        component_idx = torch.multinomial(pi, num_samples=1).squeeze(-1)
    
    # Extract parameters for selected component
    batch_idx = torch.arange(len(component_idx))
    pred_lat = mu_lat[batch_idx, component_idx]
    pred_lon = mu_lon[batch_idx, component_idx]
    pred_sigma_lat = sigma_lat[batch_idx, component_idx]
    pred_sigma_lon = sigma_lon[batch_idx, component_idx]
    
    return pred_lat, pred_lon, pred_sigma_lat, pred_sigma_lon
```

The critical addition: we also return $\sigma_{lat}$ and $\sigma_{lon}$—the model's uncertainty estimate for each prediction.

---

## The Correlation That Changes Everything

Here's where the MDN approach pays off. When we plot prediction error against predicted uncertainty across the test set, a striking pattern emerges:

**High uncertainty correlates strongly with high error.**

This isn't a trivial result. The model was never trained to minimize error. It was trained to maximize likelihood. Yet it learns that when it's uncertain, it should *say* it's uncertain—and those are exactly the cases where it tends to be wrong.

<iframe 
  src="/assets/html/mdn_error_uncertainty_heatmap.html" 
  width="100%" 
  height="500px" 
  style="border: 1px solid #ddd; border-radius: 8px;"
  loading="lazy">
</iframe>

*Toggle between Error Heatmap and Uncertainty Heatmap in the layer control. Notice how the patterns overlap—regions of high error correspond to regions of high predicted uncertainty.*

This correlation enables something the regression model couldn't do: **automatic triage**.

In production, we can set a threshold on $\sigma$. Predictions below the threshold go directly to dispatch. Predictions above the threshold get flagged for human review or secondary verification. The model tells us which predictions need a second look.

---

## Confidence Intervals in Practice

The $\sigma$ values output by the MDN have a direct geometric interpretation. For a 2D Gaussian, we can construct confidence ellipses:

- **50% CI**: $1.18\sigma$ radius captures 50% of probability mass
- **80% CI**: $1.79\sigma$ radius captures 80% of probability mass  
- **95% CI**: $2.45\sigma$ radius captures 95% of probability mass

The visualization below shows predictions with their confidence circles. Each prediction has three nested circles representing the 50%, 80%, and 95% confidence intervals. Larger circles indicate higher uncertainty.

<iframe 
  src="/assets/html/mdn_confidence_circles.html" 
  width="100%" 
  height="500px" 
  style="border: 1px solid #ddd; border-radius: 8px;"
  loading="lazy">
</iframe>

*Blue dots are predictions, green dots are ground truth. Gray lines connect predicted to actual locations. The concentric blue circles show confidence intervals—50% (darkest), 80%, and 95% (lightest). Hover over points for details including σ values in degrees and meters.*

Several patterns are visible:

1. **Tight circles, short lines**: High confidence, low error. The model is certain and correct.

2. **Wide circles, short lines**: High uncertainty, but the prediction is still reasonably accurate. The model is appropriately cautious.

3. **Wide circles, long lines**: High uncertainty, high error. The model correctly flags its own likely mistakes.

4. **Tight circles, long lines**: Overconfident errors. These are the dangerous cases—fortunately rare in well-calibrated models.

The goal of uncertainty quantification is to minimize case (4) while maximizing the usefulness of cases (1) through (3).

---

## Evaluation: Comparing Regression and MDN

Both models were trained on the same NYC address dataset (~1M address points) with identical BERT encoders. The only difference is the output head.

### Localization Performance

| Metric | Regression | MDN |
|--------|------------|-----|
| Mean Distance Error | 142m | 156m |
| Median Distance Error | 87m | 94m |
| 90th Percentile Error | 318m | 341m |

The MDN shows slightly higher error on aggregate metrics. This is expected—the MDN optimizes likelihood, not point accuracy. It's learning to represent the full distribution, which sometimes means the mode isn't exactly at the conditional mean.

The small accuracy trade-off is acceptable given what we gain.

### Calibration Quality

A well-calibrated uncertainty estimate means that when the model says "95% confidence interval," the true location should fall within that interval roughly 95% of the time.

| Confidence Level | Expected Coverage | MDN Actual Coverage |
|-----------------|-------------------|---------------------|
| 50% | 50% | 48.2% |
| 80% | 80% | 76.8% |
| 95% | 95% | 93.1% |

The MDN's uncertainty estimates are reasonably well-calibrated. The model isn't systematically overconfident or underconfident—its stated uncertainty matches empirical error rates.

### Error-Uncertainty Correlation

The Pearson correlation between predicted $\sigma_{max}$ (the larger of $\sigma_{lat}$ and $\sigma_{lon}$ in meters) and actual Haversine error:

$$\rho = 0.67$$

A correlation of 0.67 is strong for a learned uncertainty estimate. It means the model's confidence is meaningfully predictive of accuracy—exactly what we need for production triage.

---

## Production Applications

The ability to quantify uncertainty unlocks several operational improvements:

### 1. Automatic Quality Triage

```python
def should_flag_for_review(prediction, sigma_threshold_m=500):
    """Flag predictions with high uncertainty for manual review."""
    sigma_max_m = max(prediction['sigma_lat_m'], prediction['sigma_lon_m'])
    return sigma_max_m > sigma_threshold_m
```

With a threshold of 500m, roughly 8% of predictions get flagged. These flagged predictions have 3x higher average error than non-flagged predictions. The model is effectively pre-sorting its own mistakes.

### 2. Dispatch Zone Sizing

Instead of assuming fixed delivery zones, the dispatch system can use confidence intervals to define variable-radius zones. A confident prediction gets a tight dispatch radius. An uncertain prediction gets a wider search area or triggers additional confirmation.

### 3. Data Collection Prioritization

Areas where the model shows high uncertainty are areas where we need more training data. The $\sigma$ heatmap becomes a map of data collection priorities, guiding where to incentivize delivery partner coordinate logging.

### 4. Confidence-Weighted Clustering

When clustering orders for route optimization, uncertain predictions can be weighted differently. The optimizer knows which addresses might shift after human review and plans accordingly.

---

## Why This Works with Small Models

A remarkable aspect of this implementation: we achieve meaningful uncertainty quantification with a **4.4 million parameter** model (BERT-mini). The MDN head adds only a few thousand additional parameters—negligible overhead.

This efficiency matters for edge deployment. Uncertainty-aware geocoding doesn't require cloud inference or massive models. It runs on device, with latency under 50ms.

The key insight is that BERT's pre-trained representations already encode uncertainty-relevant information. When BERT sees an ambiguous address, its internal representations differ from when it sees a clear address. The MDN head just learns to surface that existing information as explicit $\sigma$ values.

Larger models would likely improve both accuracy and calibration. But even at this scale, the approach is practical and valuable.

---

## Limitations and Future Directions

### Current Limitations

**Independence assumption**: We model latitude and longitude with independent Gaussians. In reality, errors are often correlated (e.g., both off to the northeast). A full covariance matrix per component would be more accurate but adds complexity.

**Fixed number of mixtures**: We use $K=5$ components. Some addresses might need more (complex intersections), others fewer (unique addresses). Adaptive mixture counts remain an open problem.

**Out-of-distribution detection**: The model's uncertainty for addresses completely outside its training distribution (e.g., addresses in a different city) may not be calibrated. It might be confidently wrong about novel inputs.

### Future Directions

**Ensemble uncertainty**: Combining MDN with model ensembles could provide orthogonal uncertainty signals—epistemic (model uncertainty) and aleatoric (data uncertainty).

**Calibration training**: Post-hoc calibration methods (temperature scaling, isotonic regression) could further improve the alignment between stated and actual confidence.

**Active learning**: Using uncertainty estimates to guide which new addresses to label could accelerate model improvement in data-sparse regions.

---

## Try It Yourself

The MDN extension is fully integrated into the GeoBERT codebase. Training an MDN model requires only a flag change:

```bash
# Train MDN model (instead of regression)
geobert-train --training-mode mdn --mdn-num-mixtures 5
```

Inference with uncertainty:

```python
from geobert import Inferencer

inferencer = Inferencer("outputs/checkpoints", model_type="mdn")

# Get prediction with uncertainty
results = inferencer.predict_mdn_raw("123 Main Street, Manhattan, NY")
print(f"Prediction: ({results['lat'][0]:.6f}, {results['lon'][0]:.6f})")
print(f"Uncertainty: σ_lat={results['sigma_lat'][0]:.6f}°, σ_lon={results['sigma_lon'][0]:.6f}°")
```

<div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">
<h4 style="color: #333;">🗺️ GeoBERT NYC Geocoder Demo</h4>
<p style="color: #333;">The live demo now supports both Regression and MDN modes—toggle between them to compare point predictions against probabilistic outputs with confidence intervals.</p>
<a href="https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Launch Demo →</a>
</div>

---

## Conclusion

The regression-to-MDN extension represents a philosophical shift in how we think about geocoding. We're no longer asking "where is this address?" We're asking "where might this address be, and how sure are we?"

The practical benefits are substantial:
- **Automatic triage**: Flag uncertain predictions for review
- **Calibrated confidence**: Know when to trust the model
- **Error prediction**: Anticipate where mistakes will occur
- **Data prioritization**: Focus collection efforts where uncertainty is highest

All of this with minimal architectural changes and negligible computational overhead. The same tiny BERT model, the same fast inference, but with dramatically more useful outputs.

For production systems that route real deliveries to real addresses, knowing *how much* to trust a prediction is as valuable as the prediction itself. The MDN provides exactly that—turning a point estimate into a probability distribution, and uncertainty from a hidden state into a first-class output.

---

## References

- **Repository**: [github.com/suyashh94/geobert](https://github.com/suyashh94/geobert)
- **Demo**: [huggingface.co/spaces/suyash94/geobert-nyc-geocoder](https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder)
- **Previous Post**: [Teaching BERT to Read Maps: GeoBERT](/posts/geobert-geocoding-transformers/)
- **MDN Original Paper**: [Bishop, 1994 - Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/)

