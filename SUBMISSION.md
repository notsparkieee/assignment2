# Assignment 2 — Submission Checklist

## Run (Docker)

```bash
docker-compose up --build
# API: http://localhost:8000/docs
```

## Required endpoints (§7)

| Endpoint | Status |
|----------|--------|
| `POST /vector/index` | ✅ |
| `POST /vector/search` | ✅ (`search_type`: semantic \| filtered \| hybrid) |
| `GET /vector/stats` | ✅ |

## Search modes (§6)

| Mode | How to call |
|------|-------------|
| Semantic | `"search_type": "semantic"` or `POST /vector/search/semantic` |
| Metadata-filtered | `"search_type": "filtered"` + `"filters": {...}` |
| Hybrid | `"search_type": "hybrid"` + optional `metadata_filters`, `keywords` |

## Tests

```bash
pytest tests/test_phase3_api.py -v
python tests/test_phase2.py
```

## Demo video (≤ 5 min) — suggested script

1. `docker-compose up --build`
2. Open `/docs` → index a sample document
3. Semantic search → show results
4. Filtered search (`source: ocr`)
5. Hybrid search (keyword + filter)
6. `GET /vector/stats`
7. Restart container → search still works (persistence)

## Push to GitHub

```bash
git add .
git commit -m "feat(phase3-4): vector search API, service layer, and tests"
git push origin main
```
