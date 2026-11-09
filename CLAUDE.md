# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Content-based movie recommender: TF-IDF + cosine similarity over movie metadata (genres, keywords, top-5 cast, director). Three parts that must run together:

- **`ML-model/`**: offline pipeline. `notebooks/build_model.py` writes everything the API needs to `ML-model/model_v2/` (vectorizer, sparse matrix, movies CSV, precomputed top-50 neighbor table, `model_meta.json`).
- **`backend/`**: Django + DRF. `POST /api/v1/recommend/` (legacy `/api/recommend/` is also mounted), Swagger UI at `/api/docs/`.
- **`frontend/`**: Next.js 16 App Router + TypeScript + Tailwind 3 + framer-motion. Calls the backend via `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`).
- **`evaluation/`**: `evaluate.py` compares the model against popularity, CF, SVD, embeddings and hybrids and writes `evaluation/results.md`.

## Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
python manage.py runserver        # http://127.0.0.1:8000
pytest                            # 33 tests (pytest.ini sets DJANGO_SETTINGS_MODULE); `python manage.py test` also works
```
Needs `backend/.env` (copy `.env.example`): `SECRET_KEY`; `DEBUG`/`ALLOWED_HOSTS` optional; `TMDB_API_KEY` optional (only `scripts/fetch_posters.py` and the live poster fallback use it).

### Lint / format (CI runs these from the repo root)
```bash
ruff check backend/ && black --check backend/     # config in root pyproject.toml, line length 100
```

### Frontend
```bash
cd frontend
npm install
npm run dev        # http://localhost:3000, uses --webpack
npm run build      # --webpack
npm run lint
npx tsc --noEmit
```

### Rebuild the model (run from anywhere; paths derive from `__file__`)
```bash
python ML-model/notebooks/build_model.py   # rewrites ML-model/model_v2/ including neighbor_table.npy
```
Rebuild whenever scikit-learn is upgraded or `merged_movies.csv` changes: the API refuses to load artifacts built by a different scikit-learn major.minor or with inconsistent sizes. Never rebuild only part of `model_v2/`.

`clean-merge.py` regenerates `merged_movies.csv` from raw data but **drops the `poster_path` column**; re-run `python scripts/fetch_posters.py` (needs `TMDB_API_KEY`) afterwards. It needs MovieLens `ratings.csv` etc. from `python scripts/download_data.py` plus the two TMDB 5000 CSVs (manual Kaggle download).

### Evaluation
```bash
python scripts/download_data.py            # ratings.csv (gitignored)
python evaluation/evaluate.py [--no-embeddings]   # embeddings row needs `pip install sentence-transformers`
```

## Architecture gotchas (non-obvious, span multiple files)

- **Single source of truth for the model is `model_v2/`.** `utils._load_artifacts()` validates sklearn version and that `matrix`, `movies.csv`, `model_meta.json` and `neighbor_table.npy` agree on size; it raises `ModelArtifactError` (view returns 503) rather than falling back. The old `model.pkl` and `build_neighbors.py` are gone.
- **The production text is the `metadata` column** (genres + keywords + cast + director, no overview), derived in `ML-model/recommender/data_loader.py::add_metadata_column` from the stringified list columns in the CSV. `combined` (which includes the overview) still exists in the CSV but is only used by evaluation variants M2-M5.
- **Title matching lives in `recommender_app/utils.py` (`resolve_title`).** Catalog titles are MovieLens style (`Matrix, The (1999)`, `Seven (a.k.a. Se7en) (1995)`). Order: exact lowercase title → normalized key / alias (year stripped, trailing article moved/dropped, accents folded, year in query disambiguates) → `difflib` fuzzy at `FUZZY_CUTOFF=0.75`. Do **not** fuzzy-match against raw titles: that made "The Matrix" resolve to "The Martian" and short titles ("Up", "Heat") fail.
- **`query` in the API response means "this was a fuzzy correction".** The view only sets it when `same_title(query, matched)` is false; the frontend shows its "no exact match" banner from that field. Don't compare strings in the UI.
- **Rerank happens only inside the 50 precomputed neighbors** (`score = sim × vote_average^0.3`, `RERANK_BETA`). `evaluation/evaluate.py` mirrors these constants (`RERANK_BETA`, `RERANK_POOL`); keep them in sync.
- **Posters are a dict lookup** built at load time from the `poster_path` column; `_fetch_poster_live` is only a cached last resort for ids missing from the CSV.
- **Model loads at startup** via `RecommenderAppConfig.ready()` (skipped under tests); tests patch `_load_model` / `get_poster`, and `RealCatalogTest` / `ArtifactGuardTest` use the real artifacts through `_load_artifacts()`.
- CORS/CSRF in `backend/backend/settings.py` are locked to `http://localhost:3000` / `http://127.0.0.1:3000`.
- **Frontend errors are typed** (`ApiError` / `describeError` in `frontend/lib/api.ts`): 404 → not found, 429 → rate limited, 503 → model unavailable, network failure → "is the backend running?". Keep featured chips in `page.tsx` to titles that exist in the catalog (it ends in 2016; "Parasite" is not in it).

## Evaluation gotchas

- Collaborative filtering and popularity baselines are built from **train users only** and every model is scored on the same held-out users. Building CF from all users leaks the test users' ratings and inflates it badly; don't reintroduce that.
- "Popularity: most liked" currently beats every query-aware model under this protocol. That reflects the "also liked" relevance definition favouring popular films, not a bug. Read `evaluation/results.md` before claiming the model "beats a baseline".
- All content-based variants are within each other's confidence intervals; don't claim one is better without checking the CIs.
