---
title: "Teaching BERT to Read Maps: How We Built a Geocoder That Outperformed Google"
date: 2021-03-15 10:00:00 +0530
categories: [Projects]
tags: [bert, transformers, geocoding, nlp, pytorch, logistics]
description: "A deep dive into building a custom geocoding model using BERT that achieved higher accuracy than Google Maps for hyperlocal delivery in India, saving thousands of dollars annually."
image:
  path: /assets/img/posts/geobert/geobert_cover.png
  alt: GeoBERT - Teaching Transformers to Understand Geography
math: true
mermaid: true
pin: true
---

## The Problem That Kept Me Up at Night

It was August 2020. The world was grappling with a pandemic, and I was grappling with addresses.

I was working at a logistics company in India that delivered over 100,000 orders every single day. Each order needed to reach its destination, and each destination started as nothing more than a string of text—an address scribbled by a customer, often incomplete, sometimes cryptic, always uniquely Indian.

Here's the thing about Indian addresses that most geocoding services don't understand: they don't follow rules. An address might reference a landmark that existed twenty years ago. It might mention a "near Sharma ji ki dukaan" (near Mr. Sharma's shop) as if the entire city knows who Sharma ji is. Street numbers are optional. Postal codes are aspirational. And yet, our delivery partners somehow found these places, day after day.

Google Maps API was costing us a fortune. At 100K orders daily, the API calls added up to thousands of dollars every month. But that wasn't even the real problem—the real problem was that Google Maps was *wrong* for a significant portion of our hyperlocal deliveries. Not just slightly off, but confidently pointing delivery partners to the wrong neighborhoods entirely.

I knew there had to be a better way.

---

## The Transformer Revolution Was Just Beginning

In 2020, transformers were still relatively new territory. BERT had been released by Google in 2018, and the NLP community was buzzing with possibilities. People were using it for sentiment analysis, question answering, named entity recognition—the usual suspects. But geocoding? Predicting latitude and longitude from raw address text? That felt like uncharted territory.

The more I thought about it, the more it made sense. What is an address, really? It's a sequence of tokens that encode spatial information. "Sector 15, Gurgaon" means something different from "Sector 51, Gurgaon"—not just semantically, but geographically. The relationship between words in an address is fundamentally *bidirectional*. The "15" in "Sector 15" gets its meaning from "Sector," and "Sector" gets context from "Gurgaon." This was exactly what BERT was designed to capture.

BERT's architecture—Bidirectional Encoder Representations from Transformers—processes text in both directions simultaneously. Unlike older models that read left-to-right or right-to-left, BERT sees "221B Baker Street, London" as a complete picture, understanding that "221B" is a building number because of "Baker Street," and "Baker Street" is in London, not some other city. This bidirectional understanding felt perfect for addresses, where context flows in all directions.

The question was: could we make BERT understand geography?

---

## The Gold Mine Under Our Feet

Before we could train anything, we needed data. Good data. And here's where working at a logistics company turned into an unexpected advantage.

Every time a delivery partner marked an order as delivered, their phone would quietly record their GPS coordinates. Millions of deliveries meant millions of data points: an address string paired with a latitude-longitude coordinate. It was a geocoding dataset being built organically, one delivery at a time.

But raw data is rarely clean data.

When I started analyzing the coordinates, something peculiar emerged. There were clusters—dense concentrations of "delivery" locations that didn't match any residential areas. When I plotted them on a map, the pattern became clear: they were our hub locations.

Here's what was happening: delivery partners, after completing their routes, would return to the hub and batch-mark multiple orders as "delivered." The GPS coordinates would show the hub location, not the actual delivery addresses. It was efficient for the partners, catastrophic for our data.

```python
# Pseudo-code for hub detection
hub_locations = identify_clusters(all_delivery_coords, radius=50m, min_points=1000)
clean_data = remove_points_near(raw_data, hub_locations, threshold=100m)
```

We identified these "hub punches" by looking for statistical anomalies—locations where an unusually high number of orders were marked delivered within a small radius. Once identified, we filtered them out before training. This single preprocessing step dramatically improved our model's eventual accuracy.

---

## Architecture: Teaching BERT a New Trick

The core idea was deceptively simple: take BERT's powerful text representations and add a regression head that predicts two numbers—latitude and longitude.

```
Address Text → BERT Encoder → [CLS] Token Embedding → Regression Head → (lat, lon)
```

Here's how it worked:

1. **Input Processing**: We tokenized addresses using BERT's WordPiece tokenizer, converting each address into a sequence of tokens with attention masks.

2. **BERT Encoding**: The tokenized address passed through BERT's transformer layers. Each layer applied self-attention, allowing every token to "see" every other token—this is where the bidirectional magic happened.

3. **CLS Token Extraction**: BERT prepends a special [CLS] token to every input. After passing through the transformer, this token's embedding captures the semantic meaning of the entire sequence. We used this 128-dimensional vector as our address representation.

4. **Regression Head**: A simple neural network (Linear → ReLU → Linear) mapped the 128-dimensional CLS embedding to 2 outputs: normalized latitude and normalized longitude.

```python
class GeoBERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),  # BERT hidden size → hidden dim
            nn.ReLU(),
            nn.Linear(256, 2)    # hidden dim → lat/lon
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(cls_embedding)
```

We used Z-score normalization for the coordinates to ensure stable training:

$$
z_{lat} = \frac{lat - \mu_{lat}}{\sigma_{lat}}, \quad z_{lon} = \frac{lon - \mu_{lon}}{\sigma_{lon}}
$$

This normalized the targets to have zero mean and unit variance, which helped the gradient descent converge faster and more reliably.

---

## The Training Campaign

Training a model to geocode addresses isn't fundamentally different from training any regression model—you minimize the difference between predicted and actual coordinates. We used Mean Squared Error (MSE) on the normalized coordinates:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\left[(z_{lat}^{pred} - z_{lat}^{true})^2 + (z_{lon}^{pred} - z_{lon}^{true})^2\right]
$$

But the devil is in the details. We used:

- **AdamW optimizer** with weight decay (0.01) to prevent overfitting
- **Linear warmup** followed by linear decay for the learning rate
- **Gradient clipping** (max norm = 1.0) for training stability
- **Early stopping** based on validation MSE

Training on our production data took several GPU-hours, but the investment paid off. The model learned to understand the spatial structure encoded in Indian addresses—the hierarchy of sectors, the importance of landmarks, the subtle differences between neighborhoods.

---

## The Moment of Truth: Beating Google

When we finally deployed the model and compared it against Google Maps API, the results were beyond what we had hoped for.

For hyperlocal delivery in our service areas, our model achieved **higher accuracy than Google Maps**. This wasn't a marginal improvement—it was a fundamental difference in understanding. Our model had learned the peculiarities of Indian addressing from millions of actual deliveries. It understood that "near railway crossing" meant different things in different cities. It knew which landmarks mattered and which were obsolete.

But even more impactful was the cost savings. By replacing Google Maps API calls with our internal model, we saved **thousands of dollars every month**. For a company processing 100K orders daily, this translated to substantial annual savings—all while improving accuracy.

---

## A Working Demonstration: GeoBERT NYC

I can't share the production model or data due to company policies. But I wanted to demonstrate that this approach *works*—that you can teach BERT to geocode addresses with remarkable accuracy.

So I built a miniature version using publicly available NYC address data. The NYC Open Data portal provides over a million address points with coordinates—perfect for demonstrating the concept.

The demo uses **BERT-mini** (`google/bert_uncased_L-2_H-128_A-2`)—a tiny model with just **4.4 million parameters**. For comparison, the full BERT-base has 110 million parameters. Despite being 25x smaller, this mini-BERT still captures the essential bidirectional representations that make the approach work.

### Try It Yourself

<div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">
<h4 style="color: #333;">🗺️ GeoBERT NYC Geocoder Demo</h4>
<p style="color: #333;">Enter any NYC address and watch the model predict its coordinates in real-time.</p>
<a href="https://huggingface.co/spaces/suyash94/geobert-nyc-geocoder" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Launch Demo →</a>
</div>

---

## Visualizing the Model's Understanding

Numbers are one thing, but seeing the model's predictions on a map tells a richer story. Below are interactive visualizations from the evaluation run on the NYC test set.

### Predicted vs. Actual Locations

This visualization shows the model's predictions (red) versus the actual coordinates (green) for sample addresses. The gray lines connecting them represent the error. Shorter lines = better predictions.

<iframe 
  src="/assets/html/evaluation_pred_vs_actual.html" 
  width="100%" 
  height="500px" 
  style="border: 1px solid #ddd; border-radius: 8px;"
  loading="lazy">
</iframe>

*Zoom in to see individual predictions. Each marker shows the address and error distance on hover.*

### Error Categorization Across NYC

Not all predictions are equal. This map categorizes predictions by error magnitude:
- 🟢 **Good** (< 100m): Excellent accuracy, suitable for dispatch optimization
- 🟠 **Medium** (100-500m): Acceptable for clustering, needs refinement
- 🔴 **Bad** (> 500m): Outliers requiring investigation

<iframe 
  src="/assets/html/evaluation_error_markers.html" 
  width="100%" 
  height="500px" 
  style="border: 1px solid #ddd; border-radius: 8px;"
  loading="lazy">
</iframe>

*Notice how certain areas have more consistent accuracy than others—this reveals where the training data was denser.*

### Error Heatmap

Where does the model struggle? This heatmap shows the geographic distribution of errors. Brighter areas indicate higher prediction errors.

<iframe 
  src="/assets/html/evaluation_error_heatmap.html" 
  width="100%" 
  height="500px" 
  style="border: 1px solid #ddd; border-radius: 8px;"
  loading="lazy">
</iframe>

*The heatmap reveals patterns: residential areas with standardized addresses tend to have lower errors than commercial zones or areas with complex addressing.*

---

## Why BERT Works for Geocoding

Let me take a moment to explain *why* this approach works so well, because understanding the "why" helps you apply similar thinking to other problems.

### Bidirectional Context is Everything

Consider the address: `350 5th Avenue, Manhattan, NY 10118`

A left-to-right model would process "350" before knowing it's on "5th Avenue." But is "350" a street number? An apartment? A floor? The meaning is ambiguous until you see what follows.

BERT's bidirectional attention solves this. Every token can attend to every other token simultaneously:

```
350 ←→ 5th ←→ Avenue ←→ Manhattan ←→ NY ←→ 10118
```

The number "350" attends to "Avenue" and learns it's a street number. "5th" attends to "Manhattan" and learns which "5th Avenue" we're talking about (there are many in the US). The ZIP code "10118" reinforces the location with precision.

### Transfer Learning from Language

BERT was pre-trained on billions of words of text. It already "knows" that:
- Numbers often appear before street names
- "Avenue" and "Street" are similar concepts
- City names often follow street addresses
- State abbreviations like "NY" relate to geographic locations

This pre-trained knowledge transfers remarkably well to geocoding. We're not teaching BERT language from scratch—we're fine-tuning its existing knowledge for a specific spatial task.

### The CLS Token as a Spatial Signature

The [CLS] token's final embedding becomes a compact representation of the entire address. You can think of it as a "spatial signature"—a 128-dimensional vector that encodes where this address points on a map. Similar addresses (geographically close) have similar CLS embeddings. Distant addresses have different embeddings.

This property makes the regression head's job relatively simple: it just needs to learn a mapping from this semantic space to the coordinate space.

---

## Implementation Deep Dive

For those who want to build something similar, here's a deeper look at the implementation choices.

### Data Preparation

```python
# Address normalization
address = address.lower().strip()
# Tokenization with BERT tokenizer
encoding = tokenizer(
    address,
    max_length=32,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
```

We kept the maximum sequence length at 32 tokens. Most addresses fit comfortably within this limit, and longer sequences are rare enough that truncation doesn't hurt accuracy significantly.

### Coordinate Normalization

Raw latitude/longitude values would cause training instability. Z-score normalization centers the data:

```python
@dataclass
class NormalizationStats:
    lat_mean: float
    lat_std: float
    lon_mean: float
    lon_std: float
    
    def normalize(self, lat, lon):
        return (lat - self.lat_mean) / self.lat_std, \
               (lon - self.lon_mean) / self.lon_std
```

**Critical**: Compute statistics only from training data. Using test data would leak information.

### Evaluation Metrics

We tracked multiple metrics to understand model behavior:

- **MSE**: Training loss on normalized coordinates
- **MAE (degrees)**: Interpretable error in coordinate units
- **Haversine Distance (meters)**: Real-world distance error

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points."""
    R = 6_371_000  # Earth radius in meters
    # ... haversine formula implementation
    return R * c
```

The Haversine distance is what matters operationally—it tells you how far off your delivery partner would be.

---

## Lessons Learned

After months of development and deployment, here are the key lessons:

### 1. Your Data is Your Moat

The model was only as good as our delivery data. The organic collection of address-coordinate pairs, cleaned of hub punches, gave us something no public API could match: ground truth for hyperlocal Indian addresses.

### 2. Start Small, Then Scale

We began with a tiny BERT model (4M parameters) for rapid iteration. Once the approach proved viable, we scaled up. The mini-demo you see here follows the same philosophy.

### 3. Preprocessing is Half the Battle

Identifying and removing hub punches was crucial. Without this step, the model would have learned that many addresses lead to our hub locations—completely useless for actual delivery.

### 4. Interpretability Matters

The error heatmaps and categorical maps weren't just for show. They helped us identify where the model struggled and where we needed more training data.

### 5. Think Beyond Accuracy

Getting exact coordinates isn't always necessary. For order clustering and dispatch optimization, being within 100-200 meters is often good enough. This insight let us deploy the model confidently even when it wasn't perfect.

---

## The Repository

I've open-sourced the NYC demonstration version of GeoBERT. You'll find:

- Complete training pipeline with PyTorch
- Multi-GPU support via DistributedDataParallel
- MLflow integration for experiment tracking
- Inference utilities for deployment
- Gradio app for interactive demos

<div style="border: 2px solid #333; border-radius: 10px; padding: 20px; margin: 20px 0; background-color: #f5f5f5;">
<h4 style="color: #333;">📦 GeoBERT Repository</h4>
<p style="color: #333;">Full source code, training scripts, and deployment utilities.</p>
<a href="https://github.com/suyashh94/geobert" target="_blank" style="display: inline-block; background-color: #333; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">View on GitHub →</a>
</div>

---

## What Comes Next?

The geocoding problem is far from solved. Here are directions I'm excited about:

### Multi-Resolution Predictions

Instead of predicting exact coordinates, predict a probability distribution over possible locations. This would capture uncertainty in ambiguous addresses.

---

## Conclusion

In August 2020, I started with a problem: addresses that Google Maps couldn't understand. By March 2021, we had a production system that outperformed Google for our use case, saving thousands of dollars monthly while improving delivery accuracy.

The key insight was recognizing that addresses are just text—rich, contextual text that encodes spatial information. BERT's bidirectional representations were perfectly suited to capture this context. With clean training data from millions of actual deliveries and careful preprocessing to remove hub biases, we built something that worked better than off-the-shelf solutions.

This project taught me that sometimes the best solutions come from looking at problems through a different lens. Geocoding wasn't an NLP problem until we made it one. And once we did, the tools of the transformer revolution gave us everything we needed.

If you're facing a similar problem—messy addresses, expensive API costs, or accuracy issues with standard geocoders—I hope this post gives you a roadmap. The code is open, the approach is proven, and the potential is vast.

Now, go teach your own transformers to read maps.

---

*Have questions about GeoBERT or want to discuss similar applications? Reach out to me on [LinkedIn](https://linkedin.com/in/suyash94) or open an issue on the [GitHub repository](https://github.com/suyash94/geobert-nyc).*
