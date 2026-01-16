---
layout: page
title: Paper Summaries
icon: fas fa-book-reader
order: 5
---

<style>
.papers-grid {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  margin-top: 1.5rem;
}

.paper-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border-color);
  border-radius: 0.5rem;
  padding: 1.25rem 1.5rem;
  transition: border-color 0.2s;
}

.paper-card:hover {
  border-color: var(--link-color);
}

.paper-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--heading-color);
  margin-bottom: 0.5rem;
  line-height: 1.4;
  text-decoration: none;
  display: block;
}

.paper-title:hover {
  color: var(--link-color);
}

.paper-summary {
  font-size: 0.9rem;
  color: var(--text-muted-color);
  line-height: 1.6;
  margin-bottom: 1rem;
}

.read-guide-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.9rem;
  background: var(--btn-bg, #ffffff);
  color: var(--btn-color, #1a1a1a);
  border: 1px solid var(--card-border-color);
  border-radius: 0.375rem;
  font-size: 0.85rem;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.15s;
}

.read-guide-btn:hover {
  background: var(--link-color);
  color: #ffffff;
  border-color: var(--link-color);
}

/* Light mode */
[data-mode="light"] .read-guide-btn,
:root:not([data-mode="dark"]) .read-guide-btn {
  --btn-bg: #ffffff;
  --btn-color: #1a1a1a;
}

/* Dark mode */
[data-mode="dark"] .read-guide-btn {
  --btn-bg: #ffffff;
  --btn-color: #1a1a1a;
}

.no-papers {
  text-align: center;
  padding: 3rem;
  color: var(--text-muted-color);
}

.no-papers i {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  opacity: 0.4;
}
</style>

<p style="color: var(--text-muted-color); margin-bottom: 0.5rem;">
  Interactive guides generated using <a href="https://github.com/suyashh94/paper-dapper" target="_blank">Paper Dapper</a>. Each guide features a depth slider—from executive summary to implementation details.
</p>

<div class="papers-grid">
{% if site.data.papers.size > 0 %}
  {% for paper in site.data.papers %}
  <div class="paper-card">
    <a href="{{ '/papers/' | append: paper.id | append: '/' | relative_url }}" class="paper-title">
      {{ paper.title }}
    </a>
    
    {% if paper.one_sentence_summary and paper.one_sentence_summary != "" %}
    <p class="paper-summary">{{ paper.one_sentence_summary }}</p>
    {% endif %}
    
    <a href="{{ '/papers/' | append: paper.id | append: '/' | relative_url }}" class="read-guide-btn">
      <i class="fas fa-book-open"></i> Read Guide
    </a>
  </div>
  {% endfor %}
{% else %}
  <div class="no-papers">
    <i class="fas fa-inbox"></i>
    <p>No paper summaries yet. Check back soon!</p>
  </div>
{% endif %}
</div>
