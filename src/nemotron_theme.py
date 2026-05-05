import platform
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────
# 1. 색상 클래스 T
# ─────────────────────────────────────────────
class T:
    CHEONG   = '#2A5FA5'   # 청(靑) 쪽빛
    JEOK     = '#C94040'   # 적(赤) 단사
    HWANG    = '#D4941A'   # 황(黃) 황토+HF
    SONAMU   = '#5C8A3C'   # 소나무 연두
    JADAN    = '#7A3B6E'   # 자단목 자주
    CHEONGJA = '#5BA4C8'   # 청자 하늘

    # HuggingFace
    HF_GOLD   = '#FFD21E'
    HF_ORANGE = '#FF9D00'

    # UI
    BG       = '#FAFAF7'   # 배경 (한지)
    CARD     = '#F4F0E8'   # 카드 배경
    GRID     = '#DDD9C8'   # 그리드
    TEXT     = '#1E1E2E'   # 본문 텍스트 (먹)
    SUBTEXT  = '#6B6860'   # 보조 텍스트
    NOISE    = '#C8C4B0'   # 노이즈 포인트

    # 오방색 순서 리스트 (prop_cycle용)
    CYCLE = [CHEONG, HWANG, JEOK, JADAN, SONAMU, CHEONGJA, HF_ORANGE]


def color_list(n):
    """팔레트에서 n개 색상을 순서대로 반환. n > 팔레트 길이면 순환."""
    return [T.CYCLE[i % len(T.CYCLE)] for i in range(n)]


# ─────────────────────────────────────────────
# 3. 커스텀 colormap
# ─────────────────────────────────────────────
CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    'nemotron_heat',
    [T.BG, T.HF_ORANGE, T.JEOK]
)

CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'nemotron_div',
    [T.CHEONG, '#F5F0E8', T.JEOK]
)

CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'nemotron_seq',
    [T.BG, T.CHEONG]
)

# matplotlib에 등록
for _cmap in [CMAP_HEATMAP, CMAP_DIVERGING, CMAP_SEQUENTIAL]:
    try:
        mpl.colormaps.register(_cmap, force=True)
    except AttributeError:
        # matplotlib < 3.5 fallback
        mpl.cm.register_cmap(cmap=_cmap)


# ─────────────────────────────────────────────
# 4. rcParams — apply()
# ─────────────────────────────────────────────
def apply():
    font = 'AppleGothic' if platform.system() == 'Darwin' else 'NanumGothic'

    mpl.rcParams.update({
        'font.family':           font,
        'axes.unicode_minus':    False,
        'figure.facecolor':      'white',
        'axes.facecolor':        T.BG,
        'axes.edgecolor':        T.GRID,
        'axes.labelcolor':       T.TEXT,
        'axes.titlecolor':       T.TEXT,
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'axes.grid':             True,
        'grid.color':            T.GRID,
        'grid.linewidth':        0.6,
        'grid.alpha':            0.8,
        'xtick.color':           T.SUBTEXT,
        'ytick.color':           T.SUBTEXT,
        'text.color':            T.TEXT,
        'legend.framealpha':     0.85,
        'legend.edgecolor':      T.GRID,
        'axes.prop_cycle':       mpl.cycler('color', T.CYCLE),
        'savefig.dpi':           150,
        'savefig.bbox':          'tight',
        'savefig.facecolor':     'white',
    })


# ─────────────────────────────────────────────
# 5. 헬퍼 함수
# ─────────────────────────────────────────────
def fig(nrows=1, ncols=1, figsize=None, title=None):
    """Figure, Axes 생성. title 있으면 suptitle 적용."""
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    figure, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if title:
        figure.suptitle(title, fontsize=14, color=T.TEXT, y=1.01)
    return figure, axes


def save(figure, path):
    """tight_layout 후 저장, 경로 출력."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'저장: {path}')


def cluster_scatter(ax, X_2d, labels, names=None, s=4, alpha=0.4):
    """
    UMAP scatter — 군집별 오방색.
    labels: array-like (군집 번호, -1은 노이즈)
    """
    for lbl in sorted(set(labels)):
        mask = np.array(labels) == lbl
        if lbl == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       s=s, alpha=alpha * 0.5, color=T.NOISE, label='노이즈', zorder=1)
        else:
            color = T.CYCLE[lbl % len(T.CYCLE)]
            label = names.get(lbl, f'군집 {lbl}') if names else f'군집 {lbl}'
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       s=s, alpha=alpha, color=color, label=label, zorder=2)

    ax.set_xlabel('UMAP-1', color=T.SUBTEXT)
    ax.set_ylabel('UMAP-2', color=T.SUBTEXT)
    ax.legend(markerscale=4, fontsize=8, framealpha=0.85)


def metric_plot(axes_array, ks, metrics_dict, best_k=None):
    """
    K 선택 지표 3종 차트.
    metrics_dict 키: 'inertia', 'silhouette', 'db'
    best_k 있으면 수직선 + 강조점 표시.
    """
    configs = [
        ('inertia',    'Elbow — Inertia',      T.CHEONG),
        ('silhouette', 'Silhouette Score',      T.SONAMU),
        ('db',         'Davies-Bouldin Index',  T.JEOK),
    ]
    axes_flat = np.array(axes_array).flatten()

    for ax, (key, title, color) in zip(axes_flat, configs):
        values = metrics_dict[key]
        ax.plot(ks, values, 'o-', color=color, linewidth=2, markersize=5)
        if best_k is not None and best_k in ks:
            idx = list(ks).index(best_k)
            ax.axvline(best_k, color=T.HWANG, linestyle='--', linewidth=1.2, alpha=0.8)
            ax.scatter([best_k], [values[idx]], s=80, color=T.HWANG,
                       zorder=5, label=f'K={best_k}')
            ax.legend(fontsize=8)
        ax.set_title(title, color=T.TEXT)
        ax.set_xlabel('K', color=T.SUBTEXT)


def bar_keywords(ax, words, scores, color=None, title=''):
    """TF-IDF 수평 바차트 + 값 레이블."""
    if color is None:
        color = T.CHEONG
    words_r = words[::-1]
    scores_r = scores[::-1]
    bars = ax.barh(words_r, scores_r, color=color, alpha=0.85)
    for bar, score in zip(bars, scores_r):
        ax.text(score + max(scores_r) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=7, color=T.SUBTEXT)
    ax.set_xlabel('TF-IDF', color=T.SUBTEXT)
    ax.set_title(title, color=T.TEXT)
    ax.set_xlim(0, max(scores_r) * 1.15)


def radar(ax, values_dict, labels, names=None, fill_alpha=0.12):
    """
    Spider chart — 군집별 오방색.
    values_dict: {cluster_id: [v1, v2, ...]} (정규화된 값)
    labels: 축 레이블 리스트
    """
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=T.TEXT)
    ax.set_facecolor(T.BG)
    ax.yaxis.grid(True, color=T.GRID, linewidth=0.6)
    ax.spines['polar'].set_color(T.GRID)

    for cluster_id, vals in values_dict.items():
        v = list(vals) + [vals[0]]
        color = T.CYCLE[cluster_id % len(T.CYCLE)]
        name  = names.get(cluster_id, f'군집 {cluster_id}') if names else f'군집 {cluster_id}'
        ax.plot(angles, v, 'o-', linewidth=2, color=color, label=name)
        ax.fill(angles, v, alpha=fill_alpha, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)


def heatmap_style():
    """seaborn.heatmap에 넘길 kwargs 반환."""
    return dict(
        cmap=CMAP_HEATMAP,
        annot=True,
        fmt='.1f',
        annot_kws={'size': 8, 'color': T.TEXT},
        linewidths=0.4,
        linecolor=T.GRID,
        cbar_kws={'label': '비율 (%)'},
    )


def sankey_node_colors(n_clusters, n_demo):
    """
    plotly Sankey node color 리스트.
    앞 n_clusters개: 군집 색, 뒤 n_demo개: 보조 텍스트 색
    """
    cluster_colors = color_list(n_clusters)
    demo_colors    = [T.CHEONGJA] * n_demo
    return cluster_colors + demo_colors


# ─────────────────────────────────────────────
# 6. 치트시트
# ─────────────────────────────────────────────
def cheatsheet():
    print("""
─────────────────────────────────────────────────────
nemotron_theme 치트시트
─────────────────────────────────────────────────────
[setup]
  apply()                           전역 스타일 적용 (노트북 첫 셀)

[색상]
  T.CHEONG / T.JEOK / T.HWANG      청·적·황
  T.SONAMU / T.JADAN / T.CHEONGJA  소나무·자단·청자
  T.BG / T.TEXT / T.GRID           UI 색상

[군집 색상]
  color_list(n)                     팔레트에서 앞 n개 색상 리스트

[colormap]
  CMAP_HEATMAP   → 'nemotron_heat'  아이보리→오렌지→단사
  CMAP_DIVERGING → 'nemotron_div'   쪽빛↔단사
  CMAP_SEQUENTIAL→ 'nemotron_seq'   아이보리→쪽빛

[figure 유틸]
  fig(nrows, ncols, figsize, title) Figure, Axes 생성
  save(fig, path)                   저장 + 경로 출력

[시각화 함수]
  cluster_scatter(ax, X_2d, labels, names, s, alpha)
  metric_plot(axes, ks, metrics_dict, best_k)
    metrics_dict 키: 'inertia', 'silhouette', 'db'
  bar_keywords(ax, words, scores, color, title)
  radar(ax, values_dict, labels, names, fill_alpha)
  heatmap_style()                   → seaborn.heatmap kwargs
  sankey_node_colors(n_clusters, n_demo)

[예시]
  figure, ax = fig(1, 1, title='UMAP')
  cluster_scatter(ax, X_2d, labels)
  save(figure, '../image/nb02/umap.png')
─────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────
# 7. __main__ — 팔레트 미리보기
# ─────────────────────────────────────────────
if __name__ == '__main__':
    apply()

    fig_p, axes_p = plt.subplots(2, 1, figsize=(10, 5))
    fig_p.suptitle('nemotron_theme 팔레트 미리보기', fontsize=14)

    # 오방색 스와치
    swatches = [
        (T.CHEONG, '청 #2A5FA5'),
        (T.JEOK,   '적 #C94040'),
        (T.HWANG,  '황 #D4941A'),
        (T.SONAMU, '소나무 #5C8A3C'),
        (T.JADAN,  '자단 #7A3B6E'),
        (T.CHEONGJA,'청자 #5BA4C8'),
        (T.HF_GOLD, 'HF골드 #FFD21E'),
        (T.HF_ORANGE,'HF오렌지 #FF9D00'),
    ]
    ax0 = axes_p[0]
    ax0.set_xlim(0, len(swatches))
    ax0.set_ylim(0, 1)
    ax0.axis('off')
    ax0.set_title('오방색 + HuggingFace 팔레트', color=T.TEXT)
    for i, (color, label) in enumerate(swatches):
        ax0.add_patch(plt.Rectangle((i, 0.2), 0.85, 0.6, color=color))
        ax0.text(i + 0.425, 0.08, label, ha='center', va='top',
                 fontsize=7, color=T.TEXT, rotation=20)

    # colormap 프리뷰
    ax1 = axes_p[1]
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    combined = np.vstack([gradient, gradient, gradient])
    ax1.imshow(combined, aspect='auto', cmap=CMAP_HEATMAP)
    ax1.set_title('CMAP_HEATMAP (nemotron_heat)', color=T.TEXT)
    ax1.axis('off')

    plt.tight_layout()
    plt.show()
