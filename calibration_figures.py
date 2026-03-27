"""
Calibration Study: Publication Figures (Individual)
====================================================
Generates four standalone figures from annotator JSON responses.

Figure 8a: Per-annotator threshold boxplots (Q1 and Q2)
Figure 8b: Aggregate threshold distributions with calibrated boundaries
Figure 9a: Intra-rater reliability on duplicate panels
Figure 9b: Q2-Q1 gap vs model probability (identifies task misunderstanding)

Annotator mapping (internal only, not shown in figures):
  Annotator 1 = Shawn Lindner
  Annotator 2 = merbo
  Annotator 3 = David Moosmann
  Annotator 4 = paufran
  Annotator 5 = Pascal (excluded from threshold derivation)

Output: Individual PNGs at 300 DPI.

Usage:
    python calibration_figures.py
    python calibration_figures.py --data_dir /path/to/json/files
    python calibration_figures.py --output_dir /path/to/output
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

DPI = 300
FMT = 'png'

SINGLE_COL = 3.5
DOUBLE_COL = 7.0

ANNOTATOR_ORDER = [
    'Shawn_Lindner',
    'merbo',
    'David_Moosmann',
    'paufran',
    'Pascal',
]

ANNOTATOR_LABELS = {
    'Shawn_Lindner':   'Annotator 1',
    'merbo':           'Annotator 2',
    'David_Moosmann':  'Annotator 3',
    'paufran':         'Annotator 4',
    'Pascal':          'Annotator 5',
}

THRESHOLD_ANNOTATORS = ['Shawn_Lindner', 'merbo', 'David_Moosmann', 'paufran']

ANNOTATOR_COLOURS = {
    'Shawn_Lindner':   '#1f77b4',
    'merbo':           '#2ca02c',
    'David_Moosmann':  '#d62728',
    'paufran':         '#ff7f0e',
    'Pascal':          '#7f7f7f',
}

ANNOTATOR_MARKERS = {
    'Shawn_Lindner':   'o',
    'merbo':           's',
    'David_Moosmann':  'D',
    'paufran':         '^',
    'Pascal':          'x',
}

Q1_COLOUR = '#1f77b4'
Q2_COLOUR = '#d62728'


# ═══════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════

def set_style():
    plt.rcParams.update({
        'font.family':        'DejaVu Sans',
        'font.size':          10,
        'axes.titlesize':     11,
        'axes.titleweight':   'bold',
        'axes.labelsize':     11,
        'axes.labelweight':   'bold',
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'legend.fontsize':    9,
        'legend.framealpha':  1.0,
        'legend.edgecolor':   '#333333',
        'legend.fancybox':    False,
        'figure.dpi':         DPI,
        'savefig.dpi':        DPI,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.12,
        'axes.linewidth':     1.2,
        'axes.spines.top':    True,
        'axes.spines.right':  True,
        'axes.spines.bottom': True,
        'axes.spines.left':   True,
        'xtick.top':          False,
        'xtick.bottom':       True,
        'ytick.right':        False,
        'ytick.left':         True,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
        'xtick.major.size':   5,
        'ytick.major.size':   5,
        'axes.grid':          False,
        'image.interpolation':'none',
    })


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════

def load_annotator_data(data_dir):
    """Load all annotator JSON files. Returns dict keyed by annotator name."""
    data_dir = Path(data_dir)
    annotators = {}

    for json_path in sorted(data_dir.glob('calibration_*.json')):
        with open(json_path) as f:
            raw = json.load(f)

        annotator_key = raw.get('annotator', '').replace(' ', '_')

        if not annotator_key:
            stem = json_path.stem
            parts = stem.split('_')
            name_parts = []
            for p in parts[1:]:
                if p.isdigit() and len(p) > 6:
                    break
                name_parts.append(p)
            annotator_key = '_'.join(name_parts)

        originals = [r for r in raw['responses']
                     if not r.get('is_duplicate') and r.get('q1_label')]
        duplicates = [r for r in raw['responses']
                      if r.get('is_duplicate')]

        annotators[annotator_key] = {
            'raw': raw,
            'originals': originals,
            'duplicates': duplicates,
            'orig_by_panel': {r['panel_name']: r for r in originals},
        }

    return annotators


def extract_thresholds(annotators, annotator_key):
    """Extract Q1 and Q2 threshold lists for one annotator."""
    orig = annotators[annotator_key]['originals']
    q1 = [r['q1_threshold'] for r in orig if r['q1_threshold'] is not None]
    q2 = [r['q2_threshold'] for r in orig if r['q2_threshold'] is not None]
    return q1, q2


def build_common_panel_data(annotators):
    """
    Build per-panel data for panels reviewed by all annotators.
    Returns dict: panel_name -> {mean_prob, colour, annotator_name: {q1, q2}}.
    """
    panel_data = {}
    for key in ANNOTATOR_ORDER:
        if key not in annotators:
            continue
        for r in annotators[key]['originals']:
            pn = r['panel_name']
            if pn not in panel_data:
                panel_data[pn] = {
                    'mean_prob': r.get('mean_model_probability'),
                    'colour': r.get('colour'),
                }
            panel_data[pn][key] = {
                'q1': r['q1_threshold'],
                'q2': r['q2_threshold'],
            }

    loaded = [k for k in ANNOTATOR_ORDER if k in annotators]
    common = {
        pn: pd for pn, pd in panel_data.items()
        if all(k in pd for k in loaded)
    }
    return common


# ═══════════════════════════════════════════════════════════
# Figure 8a: Per-annotator threshold boxplots
# ═══════════════════════════════════════════════════════════

def figure_8a_threshold_boxplots(annotators, output_dir):
    """
    Paired boxplots showing Q1 and Q2 threshold distributions
    per annotator. Threshold annotators only (excluding Pascal).
    """
    set_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.2, 3.5))

    n_ann = len(THRESHOLD_ANNOTATORS)
    spacing = 3.0
    positions_q1 = np.arange(n_ann) * spacing
    positions_q2 = positions_q1 + 1.0

    bp_q1_data = []
    bp_q2_data = []
    labels = []

    for key in THRESHOLD_ANNOTATORS:
        q1, q2 = extract_thresholds(annotators, key)
        bp_q1_data.append(q1)
        bp_q2_data.append(q2)
        labels.append(ANNOTATOR_LABELS[key])

    bp_props = dict(
        widths=0.7,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=4, alpha=0.6),
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    bp1 = ax.boxplot(bp_q1_data, positions=positions_q1, **bp_props)
    bp2 = ax.boxplot(bp_q2_data, positions=positions_q2, **bp_props)

    for patch in bp1['boxes']:
        patch.set_facecolor(Q1_COLOUR)
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor(Q2_COLOUR)
        patch.set_alpha(0.6)

    ax.set_xticks(positions_q1 + 0.5)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Selected threshold')
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    legend_q1 = Patch(facecolor=Q1_COLOUR, alpha=0.6, edgecolor='black',
                      linewidth=0.5, label='Q1: Clean vs fouled')
    legend_q2 = Patch(facecolor=Q2_COLOUR, alpha=0.6, edgecolor='black',
                      linewidth=0.5, label='Q2: Incipient vs macro')
    ax.legend(handles=[legend_q1, legend_q2], loc='upper left', fontsize=9)

    fig.tight_layout()

    out = Path(output_dir) / f'fig8a_threshold_boxplots.{FMT}'
    fig.savefig(out, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════
# Figure 8b: Aggregate threshold distributions
# ═══════════════════════════════════════════════════════════

def figure_8b_aggregate_histograms(annotators, output_dir):
    """
    Aggregate histograms of Q1 and Q2 thresholds across all
    threshold annotators, with median lines and calibrated
    boundary annotation.
    """
    set_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.2, 3.5))

    all_q1, all_q2 = [], []
    for key in THRESHOLD_ANNOTATORS:
        q1, q2 = extract_thresholds(annotators, key)
        all_q1.extend(q1)
        all_q2.extend(q2)

    all_q1 = np.array(all_q1)
    all_q2 = np.array(all_q2)

    bins = np.arange(0.05, 1.0, 0.1)  # centered on 0.1, 0.2, ..., 0.9

    ax.hist(all_q1, bins=bins, color=Q1_COLOUR, alpha=0.50,
            edgecolor='white', linewidth=0.5, label='Q1: Clean vs fouled')
    ax.hist(all_q2, bins=bins, color=Q2_COLOUR, alpha=0.50,
            edgecolor='white', linewidth=0.5, label='Q2: Incipient vs macro')

    q1_med = np.median(all_q1)
    q2_med = np.median(all_q2)
    q1_25, q1_75 = np.percentile(all_q1, 25), np.percentile(all_q1, 75)
    q2_25, q2_75 = np.percentile(all_q2, 25), np.percentile(all_q2, 75)

    ax.axvline(q1_med, ymin=0, ymax=0.65, color=Q1_COLOUR, linewidth=2.0,
               linestyle='--', alpha=0.9, label=f'Q1 median = {q1_med:.2f}')
    ax.axvline(q2_med, ymin=0, ymax=0.65, color=Q2_COLOUR, linewidth=2.0,
               linestyle='--', alpha=0.9, label=f'Q2 median = {q2_med:.2f}')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(fontsize=9, loc='upper left', ncol=1)

    fig.tight_layout()

    out = Path(output_dir) / f'fig8b_aggregate_histograms.{FMT}'
    fig.savefig(out, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')
    print(f'    Q1 median={q1_med:.2f} IQR=[{q1_25:.2f}, {q1_75:.2f}]')
    print(f'    Q2 median={q2_med:.2f} IQR=[{q2_25:.2f}, {q2_75:.2f}]')
    print(f'    N(Q1)={len(all_q1)}, N(Q2)={len(all_q2)}')

    return q1_med, q2_med


# ═══════════════════════════════════════════════════════════
# Figure 9a: Intra-rater reliability
# ═══════════════════════════════════════════════════════════

def figure_9a_intrarater_reliability(annotators, output_dir):
    """
    Grouped bar chart showing exact and near (within 0.1) agreement
    rates on duplicate panels for all five annotators.
    """
    set_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.5, 3.8))

    ann_labels = []
    q1_exact_rates = []
    q2_exact_rates = []
    q1_close_rates = []
    q2_close_rates = []

    for key in ANNOTATOR_ORDER:
        data = annotators[key]
        orig_by_panel = data['orig_by_panel']
        dups = data['duplicates']

        q1_ex, q2_ex, q1_cl, q2_cl, checked = 0, 0, 0, 0, 0

        for dup in dups:
            orig = orig_by_panel.get(dup['panel_name'])
            if orig and dup.get('q1_label'):
                checked += 1

                if dup['q1_threshold'] == orig['q1_threshold']:
                    q1_ex += 1
                if dup['q2_threshold'] == orig['q2_threshold']:
                    q2_ex += 1

                t1a, t1b = dup['q1_threshold'], orig['q1_threshold']
                if t1a is not None and t1b is not None:
                    if abs(t1a - t1b) <= 0.1:
                        q1_cl += 1
                elif t1a is None and t1b is None:
                    q1_cl += 1

                t2a, t2b = dup['q2_threshold'], orig['q2_threshold']
                if t2a is not None and t2b is not None:
                    if abs(t2a - t2b) <= 0.1:
                        q2_cl += 1
                elif t2a is None and t2b is None:
                    q2_cl += 1

        if checked > 0:
            q1_exact_rates.append(q1_ex / checked * 100)
            q2_exact_rates.append(q2_ex / checked * 100)
            q1_close_rates.append(q1_cl / checked * 100)
            q2_close_rates.append(q2_cl / checked * 100)
        else:
            q1_exact_rates.append(0)
            q2_exact_rates.append(0)
            q1_close_rates.append(0)
            q2_close_rates.append(0)

        ann_labels.append(ANNOTATOR_LABELS[key])

    x = np.arange(len(ann_labels))
    width = 0.18
    gap = 0.03

    bars1 = ax.bar(x - 1.5 * width - gap, q1_exact_rates, width,
                   color=Q1_COLOUR, alpha=0.85, edgecolor='black',
                   linewidth=0.6, label='Q1 exact')
    bars2 = ax.bar(x - 0.5 * width, q1_close_rates, width,
                   color=Q1_COLOUR, alpha=0.35, edgecolor='black',
                   linewidth=0.6, label='Q1 \u00b10.1')
    bars3 = ax.bar(x + 0.5 * width, q2_exact_rates, width,
                   color=Q2_COLOUR, alpha=0.85, edgecolor='black',
                   linewidth=0.6, label='Q2 exact')
    bars4 = ax.bar(x + 1.5 * width + gap, q2_close_rates, width,
                   color=Q2_COLOUR, alpha=0.35, edgecolor='black',
                   linewidth=0.6, label='Q2 \u00b10.1')

    all_bars = [bars1, bars2, bars3, bars4]
    for bar_idx, bars in enumerate(all_bars):
        for ann_idx, bar in enumerate(bars):
            h = bar.get_height()
            if h == 0:
                continue
            duplicate = False
            for prev_idx in range(bar_idx):
                prev_h = all_bars[prev_idx][ann_idx].get_height()
                if abs(prev_h - h) < 0.5:
                    duplicate = True
                    break
            if not duplicate:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    short_labels = [f'Ann. {i+1}' for i in range(len(ann_labels))]
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel('Agreement (%)')
    ax.set_ylim(0, 125)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(fontsize=8, loc='upper right', ncol=2,
              handletextpad=0.3, columnspacing=0.8)

    fig.tight_layout()

    out = Path(output_dir) / f'fig9a_intrarater_reliability.{FMT}'
    fig.savefig(out, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════
# Figure 9b: Q2-Q1 gap vs model probability
# ═══════════════════════════════════════════════════════════

def figure_9b_threshold_gap(annotators, output_dir):
    """
    Scatter of Q2-Q1 gap vs model mean probability for each
    panel, per annotator. X-axis is the model output (fouling
    gradient), making it possible to see whether annotator
    separation between Q1 and Q2 is consistent across the
    full range of fouling intensity.

    All five annotators shown with distinct markers.
    Only panels where both Q1 and Q2 are non-None are plotted.
    """
    set_style()

    common = build_common_panel_data(annotators)

    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.5, 3.8))

    rng = np.random.RandomState(42)

    for key in ANNOTATOR_ORDER:
        colour = ANNOTATOR_COLOURS[key]
        marker = ANNOTATOR_MARKERS[key]
        short = f'Ann. {ANNOTATOR_ORDER.index(key) + 1}'

        probs = []
        gaps = []

        for pn, pd in common.items():
            if key not in pd:
                continue
            t1 = pd[key]['q1']
            t2 = pd[key]['q2']
            mp = pd.get('mean_prob')
            if t1 is None or t2 is None or mp is None:
                continue
            probs.append(mp)
            gaps.append(t2 - t1)

        probs = np.array(probs)
        gaps = np.array(gaps)

        # Small jitter to reduce overplotting on the 0.1 grid
        gaps_j = gaps + rng.uniform(-0.012, 0.012, len(gaps))
        probs_j = probs + rng.uniform(-0.4, 0.4, len(probs))

        if key == 'Pascal':
            ax.scatter(probs_j, gaps_j, c=colour, s=28, alpha=0.7,
                       marker=marker, linewidths=0.8, label=short, zorder=5)
        else:
            ax.scatter(probs_j, gaps_j, c=colour, s=28, alpha=0.55,
                       marker=marker, edgecolors='none', label=short, zorder=3)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

    ax.set_xlabel('Model mean probability (%)')
    ax.set_ylabel('Q2 \u2212 Q1 (threshold gap)')
    ax.set_ylim(-0.65, 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    ax.legend(fontsize=9, loc='upper left', ncol=2,
              handletextpad=0.4, columnspacing=1.0,
              markerscale=1.3)

    fig.tight_layout()

    out = Path(output_dir) / f'fig9b_threshold_gap.{FMT}'
    fig.savefig(out, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')

    for key in ANNOTATOR_ORDER:
        gaps_all = []
        for pn, pd in common.items():
            if key not in pd:
                continue
            t1, t2 = pd[key]['q1'], pd[key]['q2']
            if t1 is not None and t2 is not None:
                gaps_all.append(t2 - t1)
        inv = sum(1 for g in gaps_all if g < 0)
        zero = sum(1 for g in gaps_all if g == 0)
        print(f'    {ANNOTATOR_LABELS[key]:14s}  n={len(gaps_all):2d}  '
              f'inversions={inv}  zero_gap={zero}  '
              f'median_gap={np.median(gaps_all):.2f}')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / 'calibration_figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Data dir:   {data_dir}')
    print(f'Output dir: {output_dir}')

    print('Loading annotator data...')
    annotators = load_annotator_data(data_dir)
    print(f'  Found {len(annotators)} annotators: {list(annotators.keys())}')

    for key in ANNOTATOR_ORDER:
        if key not in annotators:
            print(f'  WARNING: {key} not found in data directory')

    print('\nFigure 8a: Per-annotator threshold boxplots')
    figure_8a_threshold_boxplots(annotators, output_dir)

    print('\nFigure 8b: Aggregate threshold distributions')
    q1_med, q2_med = figure_8b_aggregate_histograms(annotators, output_dir)

    print('\nFigure 9a: Intra-rater reliability')
    figure_9a_intrarater_reliability(annotators, output_dir)

    print('\nFigure 9b: Q2-Q1 threshold gap vs model probability')
    figure_9b_threshold_gap(annotators, output_dir)

    print(f'\nAll figures saved to: {output_dir}')
    print(f'\nFinal calibrated thresholds (4 annotators, excluding Annotator 5):')
    print(f'  Clean:     P < {q1_med:.2f}')
    print(f'  Incipient: {q1_med:.2f} <= P < {q2_med:.2f}')
    print(f'  Macro:     P >= {q2_med:.2f}')


if __name__ == '__main__':
    main()