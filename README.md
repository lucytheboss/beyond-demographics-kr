# Nemotron-Personas-Korea — NLP Analysis

> Exploratory NLP analysis of NVIDIA's [Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) dataset: 1M synthetic Korean persona profiles analyzed by age, gender, and region.

---

## Dataset

| | |
|---|---|
| Source | `nvidia/Nemotron-Personas-Korea` (HuggingFace) |
| Size | 1,000,000 rows · 26 columns |
| Language | Korean |
| Missing values | None |

Each row is a synthetic persona with structured demographic fields and 6 domain-specific text descriptions: professional, sports, arts, travel, culinary, and family life.

---

## Analysis Overview

### 1. Exploratory Data Analysis
- Distribution of age, gender, marital status, education level, and region
- Text length statistics across all persona columns

### 2. Keyword Extraction by Group (TF-IDF)
Extracted group-distinctive keywords using TF-IDF — words frequent within a group but rare across others.

- **By age group** (20s–80s+)
- **By region** (top 8 provinces)
- **By gender**
- **Cross: age × gender**

### 3. Personality vs. Interest Separation
Split persona text into two domains:
- **Personality** — `persona` + `cultural_background`
- **Interests** — `hobbies_and_interests` + `sports` + `arts` + `travel` + `culinary`

### 4. Domain Keyword Analysis (Verb+Noun Phrases)
Used `kiwipiepy` morphological analyzer to extract **verb+noun phrases** (e.g., *배달_시키다*, *외식_하다*) instead of nouns alone — capturing behavioral direction, not just topics.

Analyzed across 6 domains per group:

| Domain | Column |
|--------|--------|
| 🍜 Food | `culinary_persona` |
| 🏛️ Cultural Background | `cultural_background` |
| 🎯 Hobbies & Interests | `hobbies_and_interests` |
| ⚽ Sports | `sports_persona` |
| 🎨 Arts | `arts_persona` |
| ✈️ Travel | `travel_persona` |

### 5. WordCloud Visualization
Generated per **age × gender × region** combination (14 figures × top-3 provinces × 6 domains).

### 6. Domain-level Verb+Noun Phrase Summaries
Generated natural-language persona summaries per group × domain using template sentences.

Analyzed across 4 groupings: age group, gender, region (top 8 provinces), gender × age cross.

### 7. Key Insights & Limitations

**Findings:**
- **Age**: 20–30s show high activity in sports/travel; 60s+ lean toward cultural background and nature keywords; 40–50s show food/family stability framing
- **Gender**: Male personas concentrate on sports and technical/career terms; Female personas show more relational and sensory language in arts, travel, and food
- **Region**: Seoul/Gyeonggi cluster around urban professional keywords; provincial areas (Gyeongsang, Jeolla, Chungcheong) surface regional food names and traditional/nature terms

**Limitations:**
- Dataset is **fully synthetic** — distributions reflect NVIDIA's generation process, not real Korean population behavior
- `kiwipiepy` morphological parsing can mis-segment compound words and neologisms
- TF-IDF suppresses cross-group common vocabulary, so findings describe *relative* distinctiveness, not absolute frequency
- Persona text columns have different average lengths; shorter columns (sports, arts) produce noisier TF-IDF signals

---

## Key Methods

| Method | Purpose |
|--------|---------|
| `kiwipiepy` | Korean morphological analysis |
| TF-IDF (`scikit-learn`) | Group-distinctive keyword extraction |
| Verb+Noun phrase extraction | Behavioral direction (not just topics) |
| WordCloud | Visual summary per group × domain |
| Cosine similarity | Pairwise group distinctiveness across keyword vectors |

---

## Output

```
image/
├── age_distribution.png
├── province_distribution.png
├── age_keywords_tfidf.png
├── age_wordcloud.png
├── province_keywords_tfidf.png
├── sex_keywords_tfidf.png
├── cross_age_sex_keywords.png
├── age_personality_vs_interests.png
├── wordcloud_3way_20대_남자.png
├── wordcloud_3way_20대_여자.png
└── ... (14 total)
```

---

## Requirements

```bash
pip install datasets kiwipiepy scikit-learn wordcloud matplotlib pandas
```

---

## Usage

Run `Nemotron-Personas-Korea.ipynb` top to bottom.  
Note: morphological analysis cells take ~25–35 min total on 1M rows (sampled per group).

---

*Dataset by NVIDIA. Analysis by [lucytheboss](https://github.com/lucytheboss).*
