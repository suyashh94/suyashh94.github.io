# Paper Dapper Integration

This directory contains auto-synced Paper Dapper guides.

## Structure

```
papers/
├── README.md (this file)
├── 2509.04903/
│   └── index.html  (Paper Dapper HTML guide)
├── attention_is_all_you_need/
│   └── index.html
└── ...
```

## How it works

1. When you push a new paper to your Paper Dapper repo, a GitHub Action triggers
2. The action extracts metadata and copies the HTML guide
3. Files are synced to this directory
4. Jekyll serves them at `/papers/{paper_id}/`

## Manual sync

If automatic sync fails, you can manually copy:
1. Copy `paper_guide.html` from your Paper Dapper output
2. Rename to `index.html`
3. Place in `papers/{paper_id}/index.html`
4. Update `_data/papers.yml` with metadata

## Note

These HTML files are self-contained (base64 images embedded) and don't require
additional assets. They include their own CSS/JS and will display correctly
when served directly.
