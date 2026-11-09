# Evaluation Results

Generated: 2026-11-09  |  Catalog: 3536 movies  |  Test users: 270 (x5 held-out seeds each)  |  Train users: 305

## Accuracy metrics

| Model | P@10 | R@10 | MAP@10 | NDCG@10 (95% CI) | Coverage | Diversity | Novelty |
|-------|------|------|--------|------------------|----------|-----------|---------|
| Popularity: most liked | 0.3791 | 0.0941 | 0.2855 | **0.4104** ± 0.0345 | 0.31% | 0.9814 | 1.22 |
| Popularity: top rated (>=10 votes) | 0.1118 | 0.0259 | 0.0341 | **0.0928** ± 0.0141 | 0.31% | 0.9818 | 3.65 |
| M1: TF-IDF metadata | 0.0626 | 0.0138 | 0.0346 | **0.0752** ± 0.0100 | 75.59% | 0.9429 | 6.43 |
| Production: M1 + rating rerank | 0.0673 | 0.0146 | 0.0370 | **0.0801** ± 0.0105 | 74.10% | 0.9425 | 6.28 |
| M2: TF-IDF + overview | 0.0541 | 0.0113 | 0.0313 | **0.0658** ± 0.0102 | 75.48% | 0.9619 | 6.38 |
| M3: TF-IDF tuned | 0.0571 | 0.0121 | 0.0325 | **0.0689** ± 0.0103 | 76.53% | 0.9594 | 6.32 |
| M4: SVD/LSA | 0.0573 | 0.0123 | 0.0324 | **0.0684** ± 0.0106 | 74.21% | 0.9464 | 6.32 |
| M5: Sentence embeddings | 0.0606 | 0.0146 | 0.0325 | **0.0720** ± 0.0089 | 68.44% | 0.9685 | 6.26 |
| M6: Item-item CF | 0.2115 | 0.0650 | 0.1389 | **0.2329** ± 0.0197 | 43.86% | 0.9873 | 4.74 |
| M7: Hybrid M3+CF (alpha=0.2) | 0.2119 | 0.0654 | 0.1391 | **0.2325** ± 0.0196 | 52.21% | 0.9749 | 6.01 |
| M7: Hybrid M3+CF (alpha=0.5) | 0.2085 | 0.0640 | 0.1333 | **0.2274** ± 0.0188 | 52.55% | 0.9743 | 6.01 |
| M7: Hybrid M3+CF (alpha=0.8) | 0.1524 | 0.0418 | 0.0826 | **0.1633** ± 0.0140 | 59.22% | 0.9681 | 6.09 |

## Interpretation

**Best overall**: Popularity: most liked (NDCG@10 = 0.4104).

**Caveat - popularity wins this protocol.** A list that ignores the query and always returns the same few well-liked films (0.31% of the catalog) outscores every query-aware model. Held-out 'also liked' items are dominated by globally popular films, so this metric mostly rewards popularity. It does not measure what the product promises - films similar to the one you typed - and the NDCG numbers below should be read as 'agreement with what users liked', not 'quality of similarity'. Popularity-corrected metrics (e.g. NDCG with items weighted by inverse popularity) would be the next step.

**Baseline check**: the strongest popularity baseline (Popularity: most liked) reaches NDCG@10 = 0.4104; M1 reaches 0.0752. The content model does NOT clearly beat popularity - treat its signal with caution.

**Overview text (M1 -> M2)**: NDCG@10 0.0752 -> 0.0658 (-12.5%), within the confidence intervals, i.e. not distinguishable from noise. The production model uses metadata only (M1).

**Tuned TF-IDF (M3)** vs M1: 0.0689 vs 0.0752 (-8.3%, within noise).

**SVD/LSA (M4)** built on M3: 0.0684 vs 0.0689 (-0.7%), diversity 0.946 vs 0.959.

**Sentence embeddings (M5)**: 0.0720 vs 0.0752 for M1 (-4.2%, within noise).

**Production rerank**: sim x vote_average^0.3 over the top-50 gives 0.0801 vs 0.0752 for plain M1 (+6.6%, within noise).

**Collaborative filtering (M6)**, trained on train users only: 0.2329 vs 0.0752 for the best content model (M1: TF-IDF metadata), +210%. Because CF never sees the test users, this gap is an out-of-sample result, not leakage.

**Hybrid (M7)**: best blend is M7: Hybrid M3+CF (alpha=0.2) at 0.2325 vs CF alone 0.2329 (within noise). Blending does not measurably beat the better single signal.

**Diversity / coverage** (diversity measured in the shared M1 feature space): lowest diversity Production: M1 + rating rerank (0.942), highest M6: Item-item CF (0.987); lowest coverage Popularity: most liked (0.3%), highest M3: TF-IDF tuned (76.5%).

## Personalization lift

Does telling the model more about the user help? Same held-out users and the same relevant films for every row (content model M1, no rating rerank). 264 users x 3 draws.

| Setup | P@10 | R@10 | NDCG@10 (95% CI) | Δ NDCG vs 1 film (paired 95% CI) |
|-------|------|------|------------------|-----------------------------------|
| 1 liked film (baseline) | 0.0530 | 0.0124 | **0.0655** ± 0.0098 | n/a |
| 3 liked films | 0.0879 | 0.0211 | **0.1061** ± 0.0141 | +0.0407 ± 0.0094 * |
| 5 liked films | 0.1154 | 0.0297 | **0.1369** ± 0.0172 | +0.0715 ± 0.0125 * |
| 3 liked + dislikes | 0.0859 | 0.0211 | **0.1049** ± 0.0141 | +0.0394 ± 0.0096 * |
| 3 liked + dislikes + hide watched | 0.0896 | 0.0218 | **0.1108** ± 0.0149 | +0.0453 ± 0.0100 * |
| 5 liked + dislikes + hide watched | 0.1183 | 0.0299 | **0.1443** ± 0.0180 | +0.0788 ± 0.0129 * |
| Popularity: most liked | 0.3559 | 0.0954 | **0.3894** ± 0.0342 | +0.3239 ± 0.0326 * |
| Popularity: most liked + hide watched | 0.3874 | 0.1033 | **0.4231** ± 0.0368 | +0.3576 ± 0.0349 * |

\* the paired 95% CI excludes zero.

- **3 liked films** vs 1 liked film (baseline): NDCG@10 0.1061 vs 0.0655 (+62.1%), a real gain.
- **5 liked films** vs 3 liked films: NDCG@10 0.1369 vs 0.1061 (+29.0%), a real gain.
- **3 liked + dislikes** vs 3 liked films: NDCG@10 0.1049 vs 0.1061 (-1.2%), within noise.
- **3 liked + dislikes + hide watched** vs 3 liked + dislikes: NDCG@10 0.1108 vs 0.1049 (+5.6%), a real gain.
- Hiding watched films also lifts the popularity baseline (0.3894 -> 0.4231), so part of that gain is not about taste.
- Even the richest profile (0.1443) stays far below query-ignoring popularity (0.4231) under this protocol; see the caveat above about what the metric rewards.

**Protocol.** For each test user, 5 liked films are revealed per draw and the user's *other* liked films are the relevant set (identical for every row). A row using fewer than 5 likes never recommends the unused revealed ones. *Dislikes* are up to 3 films the user rated <= 2.5. *Hide watched* removes every film the user rated except the held-out relevant ones, which is what a real user's own history would remove. The scoring function is `backend/recommender_app/profile.py`, loaded directly, so this measures the production code. Filters and in-session thumbs feedback have no offline ground truth and are not measured here.

## Evaluation methodology

- **Split**: users are shuffled (seed 42) and split 50/50 into test and train users. Collaborative filtering and the popularity baselines are built from **train users only**; content models use no ratings. All models are scored on the same test users, so no model sees the ratings it is graded against.
- **Protocol**: for each test user with >=20 ratings and >=2 liked catalog movies, up to 5 liked movies (>=4.0 stars) are held out one at a time as the query; the user's other liked catalog movies are the relevant set. The model returns its top-10, excluding the query.
- **Aggregation**: metrics are averaged over a user's seeds, then over users. The ± is a 95% normal-approximation CI over users; differences smaller than the combined CIs are reported as noise.
- **Metrics**: Precision/Recall/MAP/NDCG@10; catalog coverage (fraction of the library ever recommended); intra-list diversity (1 - mean pairwise cosine similarity within a list, always in the M1 TF-IDF space so models are comparable, 300 sampled queries); novelty (mean -log2 popularity; higher = more obscure).
- **Baselines**: *most liked* ranks by the number of train-user ratings >= 4.0; *top rated* ranks by mean rating among films with >=10 train-user votes. Both ignore the query.
- **Hybrid**: alpha x content + (1 - alpha) x CF over min-max normalised full score vectors. All three alphas are reported; none was selected on the test users.
- **Production row**: replicates what the API serves (M1 top-50 neighbors re-ranked by sim x vote_average^0.3).

## Known limitations

- Small catalog (MovieLens small ∩ TMDB 5000) and only ~610 MovieLens users; CF is trained on half of them, so its absolute numbers would improve with more data.
- Relevance is 'the user also liked it', which favours popular, well-connected films (a query-independent popularity list is a very strong baseline here) and rewards CF structurally; it does not measure how good a *content-similar* list feels (e.g. sequels are trivially similar but may not be what a user wants).
- Seeds are drawn from each user's liked films, so queries skew towards popular titles.
- Queries outside the catalog fall back to title matching, which the API now normalises and reports via the `query` field when it had to correct a typo.
