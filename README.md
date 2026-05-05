# Nemotron-Personas-Korea — Psychographic Clustering

> Can we build meaningful user clusters from synthetic Korean persona data **without demographic variables**?  
> This project applies NLP and unsupervised clustering to NVIDIA's [Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) dataset — 1M synthetic Korean persona profiles — to explore behavior- and values-based segmentation as an alternative to demographic stereotyping.

---

## Dataset

| | |
|---|---|
| Source | `nvidia/Nemotron-Personas-Korea` (HuggingFace) |
| Size | 1,000,000 rows · 26 columns |
| Language | Korean |
| Missing values | None |

Each row is a synthetic persona with structured demographic fields and 6 domain-specific text descriptions (professional, sports, arts, travel, culinary, hobbies) plus values/identity columns.

---

## Notebooks

### Notebook 01 · EDA & Keyword Analysis
[`notebook/01_korean-persona-nlp-analysis.ipynb`](notebook/01_korean-persona-nlp-analysis.ipynb)

Baseline exploration: who is in the dataset and what language do they use?

- Distribution of age group, gender, region, education level
- Text length statistics across all persona columns
- TF-IDF keyword extraction by **age group / region / gender / age × gender**
- Personality vs. interest keyword separation
- Domain keyword analysis across 6 domains (food, culture, hobbies, sports, arts, travel)
- WordCloud per age × gender × region combination

---

### Notebook 02 · Behavioral Clustering
[`notebook/02_behavioral-clustering.ipynb`](notebook/02_behavioral-clustering.ipynb)

**Research question**: Can behavioral text alone (no demographics) produce meaningful psychographic clusters?

| Step | Method |
|------|--------|
| Text | 5 behavioral columns concatenated |
| Morphology | `kiwipiepy` noun extraction |
| Embedding | SVD 300D (Option A) vs. Sentence Transformer `ko-sroberta-multitask` (Option B) |
| Dim reduction | UMAP 50D (clustering) + 2D (visualization) |
| Clustering | K-Means (K=4 selected) + HDBSCAN comparison |
| Interpretation | TF-IDF keywords · Radar Chart · Word Network |
| Reverse analysis | Cluster × age / gender / region Heatmap + Sankey |
| Behavior signals | 5 signals: 자율성·사회성·정보탐색·경험추구·루틴안정 |
| Agent mapping | Amershi G6·G11·G13 guideline mapping per cluster |

**Key findings:**
- Option B (Sentence Transformer) wins: Silhouette 0.41 vs. 0.34
- K=4 optimal: Silhouette 0.60, Davies-Bouldin 0.56 (both best simultaneously)
- HDBSCAN produced 28 clusters with 23.5% noise — K-Means adopted
- Cluster 2 stands out: 경주·유적지·역사·부여·불국사 (historic travel type)
- Social keywords (친구·가족) appear 4.40×/row vs. autonomy 0.28×/row — LLM generation bias confirmed

---

### Notebook 03 · Values-Based Clustering
[`notebook/03_values-based-clustering.ipynb`](notebook/03_values-based-clustering.ipynb)

**Research question**: Do values and life orientation (not behavior) form distinct clusters, and are they independent of demographics?

| Layer | Columns | Role |
|-------|---------|------|
| Layer A | `persona`, `cultural_background`, `career_goals_and_ambitions` | Clustering input |
| Layer B | `hobbies_and_interests`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona` | Comparison only |
| Demographics | `age_group`, `sex`, `province`, `education_level`, `occupation` | Reverse analysis only |

| Step | Method |
|------|--------|
| Morphology | `kiwipiepy` lemmatization (verb/adj → base form) |
| Embedding | Sentence Transformer `ko-sroberta-multitask` (768D) |
| Dim reduction | UMAP 50D (clustering) + 2D (visualization) |
| Clustering | K-Means + HDBSCAN comparison |
| Reverse A | Values clusters × demographics Heatmap + Sankey |
| Reverse B | Values clusters × behavioral clusters cross-Heatmap |
| Agent mapping | Amershi G6·G11·G13 per values-cluster profile |

**Core argument:**  
If values clusters distribute evenly across age/gender/region → demographic stereotyping in AI design is empirically refuted.  
If values and behavior clusters don't align → the two must be modeled as independent layers.

---

## Design Philosophy

> Demographics tell you *who* someone is. Behavior tells you *what* they do. Values tell you *why*.

This project tests whether LLM-generated synthetic personas encode psychographic structure beyond demographic labels — and whether that structure can ground fairer, more adaptive AI agent design.

**Amershi et al. (2019) guidelines addressed:**

| Guideline | How |
|-----------|-----|
| G6 · Mitigate social biases | Clusters built without demographic input |
| G11 · Explainability | TF-IDF keywords surfaced as cluster rationale |
| G13 · Learn from user behavior | Behavior signals inform per-cluster agent strategy |

---

## Repository Structure

```
notebook/
├── 01_korean-persona-nlp-analysis.ipynb
├── 02_behavioral-clustering.ipynb
└── 03_values-based-clustering.ipynb

src/
└── nemotron_theme.py          # 오방색 × HuggingFace palette + chart helpers

image/
├── nb01/                      # EDA & keyword visualizations
├── nb02/                      # Behavioral clustering outputs
└── nb03/                      # Values-based clustering outputs
```

---

## Setup

```bash
pip install datasets kiwipiepy scikit-learn sentence-transformers \
            umap-learn hdbscan wordcloud matplotlib seaborn plotly kaleido
```

Create a `.env` file in the project root:

```
HF_TOKEN=your_token_here
```

Run notebooks top to bottom in order (01 → 02 → 03).  
Morphological analysis and embedding cells take ~10–25 min each.

---

## Reference

Amershi, S., et al. (2019). Software engineering for machine learning: A case study. *ICSE-SEIP*.  
Dataset: NVIDIA. *Nemotron-Personas-Korea*. HuggingFace, 2024.

---

*Analysis by [lucytheboss](https://github.com/lucytheboss).*
