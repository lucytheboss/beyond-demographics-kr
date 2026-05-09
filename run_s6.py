import os, sys, pickle, asyncio, json, re, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

import anthropic
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu

import nest_asyncio
nest_asyncio.apply()

# Constants
RANDOM_STATE = 42
HAIKU_INPUT_COST  = 0.80 / 1_000_000
HAIKU_OUTPUT_COST = 4.00 / 1_000_000
BUDGET_LIMIT = 10.0
CLUSTER_NAMES_A = {
    0:'자기계발·전환 모색형',
    1:'소박 일상·지역 공동체형',
    2:'현실적 목표 실践형',
    3:'관계·가족 중심형',
    4:'책임·안정 지향형'
}
TARGET_GENRES = ['Drama','Thriller','Action','Romance','Horror','Comedy','Animation']
cost_tracker = {"calls":0,"input_tokens":0,"output_tokens":0}

# ── Load data ────────────────────────────────────────────────────────────────
with open('data/cluster_profiles_a.pkl', 'rb') as f:
    cluster_profiles = pickle.load(f)

korean_movies_final = pd.read_parquet('data/korean_movies_sample.parquet')
if 'primary_genre' not in korean_movies_final.columns:
    korean_movies_final['primary_genre'] = korean_movies_final['genres'].str.split('|').str[0]

sim_df = pd.read_parquet('data/simulation_results.parquet')
if 'primary_genre' not in sim_df.columns:
    sim_df['primary_genre'] = sim_df['genres'].str.split('|').str[0]

ratings_raw = pd.read_csv('data/ml-latest/ratings.csv')
korean_ratings_filtered = ratings_raw[ratings_raw['movieId'].isin(korean_movies_final['movieId'])]
ml_df = korean_ratings_filtered[korean_ratings_filtered['movieId'].isin(korean_movies_final['movieId'])]

print(f"Loaded: {len(korean_movies_final)} movies, {len(sim_df)} sim rows, {len(ml_df)} ML ratings")

# ── Helpers ──────────────────────────────────────────────────────────────────
def estimate_cost(input_tok, output_tok):
    return input_tok * HAIKU_INPUT_COST + output_tok * HAIKU_OUTPUT_COST

def check_budget():
    cost = estimate_cost(cost_tracker['input_tokens'], cost_tracker['output_tokens'])
    if cost > BUDGET_LIMIT:
        raise RuntimeError(f"Budget exceeded: ${cost:.2f}")
    return cost

def parse_response(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass
    return None

def rating_to_dist(ratings, bins=np.arange(0.5, 5.6, 0.5)):
    """Convert rating array to probability distribution over bins."""
    hist, _ = np.histogram(ratings, bins=bins)
    total = hist.sum()
    if total == 0:
        return np.ones(len(hist)) / len(hist)
    return hist / total

# ── Step 1: Demo baseline simulation ─────────────────────────────────────────
DEMO_PROFILES = [
    {'age':'20대','gender':'여성'},
    {'age':'20대','gender':'남성'},
    {'age':'40대','gender':'여성'},
    {'age':'40대','gender':'남성'},
    {'age':'60대','gender':'여성'},
    {'age':'60대','gender':'남성'},
]
DEMO_SIM_PATH = 'data/demo_baseline.parquet'

if os.path.exists(DEMO_SIM_PATH):
    print("Loading cached demo_baseline.parquet ...")
    demo_df = pd.read_parquet(DEMO_SIM_PATH)
    if 'primary_genre' not in demo_df.columns:
        demo_df['primary_genre'] = demo_df['genres'].str.split('|').str[0]
else:
    print("Running async demo simulation ...")
    client = anthropic.AsyncAnthropic()

    async def simulate_profile(sem, profile_idx, profile, movies):
        results = []
        async with sem:
            for _, movie in movies.iterrows():
                check_budget()
                prompt = (
                    f"[나이/성별] {profile['age']} {profile['gender']}\n"
                    f"[영화] {movie['title']} / {movie['genres']}\n"
                    f'반드시 JSON만 반환: {{"watch":bool,"rating":1.0~5.0,"reason":"15자이내"}}'
                )
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=80,
                    temperature=0,
                    system="당신은 아래 정보를 가진 한국인입니다. JSON만 반환하세요.",
                    messages=[{"role":"user","content":prompt}]
                )
                cost_tracker['calls'] += 1
                cost_tracker['input_tokens'] += resp.usage.input_tokens
                cost_tracker['output_tokens'] += resp.usage.output_tokens
                parsed = parse_response(resp.content[0].text)
                if parsed:
                    results.append({
                        'profile_idx': profile_idx,
                        'age': profile['age'],
                        'gender': profile['gender'],
                        'movie_id': movie['movieId'],
                        'title': movie['title'],
                        'genres': movie['genres'],
                        'primary_genre': movie['primary_genre'],
                        'watch': parsed.get('watch', True),
                        'rating': float(parsed.get('rating', 3.0)),
                        'reason': parsed.get('reason',''),
                    })
        return results

    async def run_all():
        sem = asyncio.Semaphore(3)
        movies_sample = korean_movies_final.sample(min(50, len(korean_movies_final)), random_state=RANDOM_STATE)
        tasks = [simulate_profile(sem, i, p, movies_sample) for i, p in enumerate(DEMO_PROFILES)]
        all_results = await asyncio.gather(*tasks)
        flat = [r for sub in all_results for r in sub]
        return pd.DataFrame(flat)

    demo_df = asyncio.run(run_all())
    demo_df.to_parquet(DEMO_SIM_PATH, index=False)
    print(f"Saved demo_baseline.parquet ({len(demo_df)} rows)")

# ── Step 2: Compute JSD arrays ────────────────────────────────────────────────
# Real ratings: join ml_df with korean_movies_final to get genre
ml_with_genre = ml_df.merge(
    korean_movies_final[['movieId','primary_genre']], on='movieId', how='left'
)

bins = np.arange(0.5, 5.6, 0.5)

def genre_jsd(sim_ratings_by_genre, real_ratings_by_genre, genres):
    jsds = []
    for g in genres:
        real = real_ratings_by_genre.get(g, np.array([]))
        sim  = sim_ratings_by_genre.get(g, np.array([]))
        if len(real) < 5 or len(sim) < 5:
            jsds.append(np.nan)
        else:
            p = rating_to_dist(real, bins)
            q = rating_to_dist(sim, bins)
            jsds.append(jensenshannon(p, q))
    return np.array(jsds)

# Real by genre
real_by_genre = {g: ml_with_genre[ml_with_genre['primary_genre']==g]['rating'].values
                 for g in TARGET_GENRES}

# Demo JSD
demo_by_genre = {g: demo_df[demo_df['primary_genre']==g]['rating'].values
                 for g in TARGET_GENRES}
demo_jsd_by_genre = genre_jsd(demo_by_genre, real_by_genre, TARGET_GENRES)

# Values JSD (average across clusters)
cluster_jsds = []
for cid in range(5):
    cdf = sim_df[sim_df['cluster_id']==cid]
    c_by_genre = {g: cdf[cdf['primary_genre']==g]['rating'].values for g in TARGET_GENRES}
    cjsd = genre_jsd(c_by_genre, real_by_genre, TARGET_GENRES)
    cluster_jsds.append(cjsd)
cluster_jsds = np.array(cluster_jsds)  # shape (5, 7)
values_jsd_by_genre = np.nanmean(cluster_jsds, axis=0)

print(f"\nDemo JSD by genre: {dict(zip(TARGET_GENRES, demo_jsd_by_genre.round(3)))}")
print(f"Values JSD by genre: {dict(zip(TARGET_GENRES, values_jsd_by_genre.round(3)))}")

# ── Step 3: Visualization ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('image/nb04', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Demographic vs. Values-Based Simulation: JSD from Real Ratings', fontsize=13, fontweight='bold')

# --- Subplot 1: Overall bar with errorbars ---
ax1 = axes[0]
demo_mean = np.nanmean(demo_jsd_by_genre)
values_mean = np.nanmean(values_jsd_by_genre)
demo_std = np.nanstd(demo_jsd_by_genre)
values_std = np.nanstd(values_jsd_by_genre)

x = np.array([0, 1])
bars = ax1.bar(x, [demo_mean, values_mean], yerr=[demo_std, values_std],
               color=['#4472C4','#ED7D31'], capsize=8, width=0.5,
               error_kw={'elinewidth':2})
ax1.set_xticks(x)
ax1.set_xticklabels(['Demographic\n(6 profiles)', 'Values-Based\n(5 clusters)'], fontsize=11)
ax1.set_ylabel('Jensen-Shannon Divergence', fontsize=11)
ax1.set_title('Overall JSD (mean ± std)', fontsize=11)
ax1.set_ylim(0, max(demo_mean+demo_std, values_mean+values_std)*1.4 + 0.05)
for bar, val in zip(bars, [demo_mean, values_mean]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+demo_std+0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Subplot 2: Genre grouped bar ---
ax2 = axes[1]
n_genres = len(TARGET_GENRES)
x2 = np.arange(n_genres)
w = 0.35
b1 = ax2.bar(x2-w/2, demo_jsd_by_genre, w, label='Demographic', color='#4472C4', alpha=0.85)
b2 = ax2.bar(x2+w/2, values_jsd_by_genre, w, label='Values-Based', color='#ED7D31', alpha=0.85)
ax2.set_xticks(x2)
ax2.set_xticklabels(TARGET_GENRES, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('JSD', fontsize=11)
ax2.set_title('JSD by Genre', fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(np.nanmax(demo_jsd_by_genre), np.nanmax(values_jsd_by_genre))*1.3 + 0.05)

plt.tight_layout()
img_path = 'image/nb04/demographic_vs_values_redesigned.png'
plt.savefig(img_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved image to {img_path}")

# ── Step 4: Statistical validation ───────────────────────────────────────────
np.random.seed(RANDOM_STATE)
N_BOOT = 1000

def bootstrap_ci(data, n=N_BOOT, ci=0.95):
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan
    boots = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n)]
    lo = np.percentile(boots, (1-ci)/2*100)
    hi = np.percentile(boots, (1-(1-ci)/2)*100)
    return lo, hi

demo_ci = bootstrap_ci(demo_jsd_by_genre)
values_ci = bootstrap_ci(values_jsd_by_genre)

# Mann-Whitney U on per-genre JSD arrays (drop nans)
demo_valid = demo_jsd_by_genre[~np.isnan(demo_jsd_by_genre)]
values_valid = values_jsd_by_genre[~np.isnan(values_jsd_by_genre)]

if len(demo_valid) >= 2 and len(values_valid) >= 2:
    stat_mw, p_value = mannwhitneyu(demo_valid, values_valid, alternative='two-sided')
else:
    stat_mw, p_value = np.nan, 1.0

# Cohen's d
def cohens_d(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std

d = cohens_d(demo_jsd_by_genre, values_jsd_by_genre)

# Per-genre bootstrap CIs (non-overlapping count)
non_overlap_count = 0
print("\n--- Per-genre Bootstrap CI summary ---")
print(f"{'Genre':<12} {'Demo JSD':>10} {'Values JSD':>12} {'Demo CI':>20} {'Values CI':>20} Overlap?")
for i, g in enumerate(TARGET_GENRES):
    dv = demo_jsd_by_genre[i]
    vv = values_jsd_by_genre[i]
    # Bootstrap CI from the cluster data for values
    cluster_genre_jsds = cluster_jsds[:, i]
    d_ci = bootstrap_ci(np.array([dv]*3))  # single-point; use genre-level cluster variance
    v_ci = bootstrap_ci(cluster_genre_jsds)
    # For demo we only have 1 value per genre across 6 profiles — use demo_jsd directly
    # Non-overlap: check if demo point outside values CI
    overlap = v_ci[0] <= dv <= v_ci[1]
    if not overlap:
        non_overlap_count += 1
    print(f"{g:<12} {dv:>10.4f} {vv:>12.4f} ({v_ci[0]:.3f},{v_ci[1]:.3f}) overlap={'Yes' if overlap else 'No'}")

print(f"\nMann-Whitney U={stat_mw:.2f}, p={p_value:.4f}")
print(f"Cohen's d = {d:.4f}")
print(f"Bootstrap CI Demo: ({demo_ci[0]:.4f}, {demo_ci[1]:.4f})")
print(f"Bootstrap CI Values: ({values_ci[0]:.4f}, {values_ci[1]:.4f})")
print(f"Non-overlapping genres: {non_overlap_count}")

# Save text summary
with open('data/statistical_summary.txt', 'w') as f:
    f.write(f"Section 6 Statistical Validation Summary\n")
    f.write(f"=========================================\n")
    f.write(f"Demo JSD mean: {np.nanmean(demo_jsd_by_genre):.4f}\n")
    f.write(f"Values JSD mean: {np.nanmean(values_jsd_by_genre):.4f}\n")
    f.write(f"Mann-Whitney U={stat_mw:.2f}, p={p_value:.4f}\n")
    f.write(f"Cohen's d={d:.4f}\n")
    f.write(f"Non-overlapping genres: {non_overlap_count}\n")
print("\nSaved statistical_summary.txt")

# ── Run 5 checks ─────────────────────────────────────────────────────────────
demo_mean_val = np.nanmean(demo_jsd_by_genre)
values_mean_val = np.nanmean(values_jsd_by_genre)
diff_val = abs(demo_mean_val - values_mean_val)
img_exists = os.path.exists(img_path)

check1 = 0.05 <= demo_mean_val <= 0.60
check2 = 0.10 <= values_mean_val <= 0.70
check3 = diff_val > 0.01
check4 = (not np.isnan(p_value)) and (non_overlap_count >= 1)
check5 = img_exists

additional_cost = estimate_cost(cost_tracker['input_tokens'], cost_tracker['output_tokens'])
# Try to get cumulative cost from previous runs if tracked
total_cost = additional_cost  # simplified; no persistent tracker

print("\n" + "="*60)
print(f"{'✅' if check1 else '❌'} 체크 1: 인구통계 JSD = {demo_mean_val:.3f} ({'정상' if check1 else '이상'})")
print(f"{'✅' if check2 else '❌'} 체크 2: 가치관 JSD = {values_mean_val:.3f} ({'정상' if check2 else '이상'})")
print(f"{'✅' if check3 else '❌'} 체크 3: 두 값 차이 = {diff_val:.3f} ({'비교 유효' if check3 else '무효'})")
print(f"{'✅' if check4 else '❌'} 체크 4: Mann-Whitney p = {p_value:.3f}, CI 비겹침 장르 {non_overlap_count}개")
print(f"{'✅' if check5 else '❌'} 체크 5: 이미지 생성 {'확인' if check5 else '실패'}")
print(f"총 추가 비용: ${additional_cost:.2f} | 누적 총 비용: ${total_cost:.2f}")

all_passed = all([check1, check2, check3, check4, check5])
print(f"\n{'모든 체크 통과!' if all_passed else '일부 체크 실패 — 위 결과 확인 요망'}")
