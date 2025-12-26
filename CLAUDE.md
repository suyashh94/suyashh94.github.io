# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal portfolio/blog site built with Jekyll using the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy). It's deployed to GitHub Pages via the `pages-deploy.yml` workflow.

## Common Commands

### Local Development
```bash
# Install dependencies (first time setup)
bundle install

# Run local dev server with live reload
bash tools/run.sh

# Run in production mode
bash tools/run.sh -p

# Build and test the site (runs htmlproofer)
bash tools/test.sh
```

### Build Only
```bash
# Production build
JEKYLL_ENV=production bundle exec jekyll b -d _site
```

## Architecture

### Content Structure
- `_posts/` - Blog posts in markdown with YAML front matter (format: `YYYY-MM-DD-title.md`)
- `_tabs/` - Navigation tabs (about, archives, categories, tags)
- `_data/contact.yml` - Sidebar contact links configuration
- `_data/share.yml` - Post sharing buttons configuration

### Key Configuration
- `_config.yml` - Main Jekyll config; theme settings, site metadata, analytics, comments
- `_plugins/posts-lastmod-hook.rb` - Auto-populates `last_modified_at` from git history

### Post Front Matter
Posts require:
```yaml
---
title: "Post Title"
date: YYYY-MM-DD
categories: [Category1, Category2]
tags: [tag1, tag2]
---
```

### Theme Customization
The Chirpy theme is installed as a gem (`jekyll-theme-chirpy`). Theme files are in the gem, not this repo. To override theme files, copy them from the gem (`bundle info --path jekyll-theme-chirpy`) to corresponding local paths.

## Deployment

Push to `main` or `master` triggers the GitHub Actions workflow which:
1. Builds with `jekyll b`
2. Tests with `htmlproofer`
3. Deploys to GitHub Pages
