---
layout: page
title: Paper Summaries
icon: fas fa-book-reader
order: 5
---

<style>
.papers-grid {
  display: grid;
  gap: 1.5rem;
  margin-top: 1.5rem;
}

.paper-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border-color);
  border-radius: 0.75rem;
  padding: 1.5rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

.paper-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.paper-card a {
  text-decoration: none;
  color: inherit;
}

.paper-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--heading-color);
  margin-bottom: 0.5rem;
  line-height: 1.4;
}

.paper-title:hover {
  color: var(--link-color);
}

.paper-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--text-muted-color);
  margin-bottom: 0.75rem;
}

.paper-meta span {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

.paper-summary {
  font-size: 0.95rem;
  color: var(--text-color);
  line-height: 1.6;
}

.paper-type-badge {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.paper-type-methodology { background: #dbeafe; color: #1e40af; }
.paper-type-empirical { background: #dcfce7; color: #166534; }
.paper-type-survey { background: #fef3c7; color: #92400e; }
.paper-type-theoretical { background: #f3e8ff; color: #7c3aed; }
.paper-type-unknown { background: #f3f4f6; color: #4b5563; }

.no-papers {
  text-align: center;
  padding: 3rem;
  color: var(--text-muted-color);
}

.view-button {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 1rem;
  padding: 0.5rem 1rem;
  background: var(--link-color);
  color: white;
  border-radius: 0.5rem;
  font-size: 0.9rem;
  font-weight: 500;
  transition: background 0.2s;
}

.view-button:hover {
  background: var(--link-hover-color);
  color: white;
}
</style>

<p>
  Interactive guides for academic papers, generated using 
  <a href="https://github.com/suyashh94/paper-dapper" target="_blank">Paper Dapper</a>. 
  Each guide features a depth slider letting you choose how deep you want to go—from 
  executive summary to implementation details.
</p>

<div class="papers-grid">
{% if site.data.papers.size > 0 %}
  {% for paper in site.data.papers %}
  <div class="paper-card">
    <a href="{{ '/papers/' | append: paper.id | append: '/' | relative_url }}">
      <h3 class="paper-title">{{ paper.title }}</h3>
    </a>
    
    <div class="paper-meta">
      <span><i class="far fa-calendar-alt"></i> {{ paper.date }}</span>
      {% if paper.authors and paper.authors != "" %}
      <span><i class="far fa-user"></i> {{ paper.authors | truncate: 50 }}</span>
      {% endif %}
      <span class="paper-type-badge paper-type-{{ paper.paper_type | default: 'unknown' }}">
        {{ paper.paper_type | default: 'Paper' }}
      </span>
    </div>
    
    {% if paper.one_sentence_summary and paper.one_sentence_summary != "" %}
    <p class="paper-summary">{{ paper.one_sentence_summary }}</p>
    {% endif %}
    
    <a href="{{ '/papers/' | append: paper.id | append: '/' | relative_url }}" class="view-button">
      <i class="fas fa-book-open"></i> Read Guide
    </a>
  </div>
  {% endfor %}
{% else %}
  <div class="no-papers">
    <i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
    <p>No paper summaries yet. Check back soon!</p>
  </div>
{% endif %}
</div>
