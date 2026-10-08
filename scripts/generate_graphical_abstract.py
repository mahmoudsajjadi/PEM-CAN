"""
Publication-grade script to generate an improved, modern, icon-rich PEM-CAN Graphical Abstract.
Features:
- Large readable typography (no fine print)
- Minimal, high-impact text and key statistics
- Custom vector icons:
  1. Retinal Oculomics: Eye with iris, pupil, and reflection
  2. Wearable Telemetry: Smartwatch with active ECG pulse
  3. Frozen Foundation: Padlock icon
  4. Temporal Tokenizer: Multi-frequency waveform waves
  5. Stiefel LoRA: Orthonormal axes and adapter matrices
  6. Clinical Protocol: Medical cross / clinical data registry
  7. Adjudicated Targets: Target bullseye
  8. Diagnostic Discrimination: High-AUROC target badge
  9. Efficiency: Microprocessor chip
  10. Edge & Federated: Multi-node distributed network
- Modern clinical & engineering color palette (Slate, Indigo, Emerald, Amber, Rose)
- Dimensions: 13.28 x 5.31 inches at 300 DPI (Elsevier CAS standard 5:2 ratio)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle, 
                                PathPatch, Rectangle, Polygon)
from matplotlib.path import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PAPER_DIR, 'fig')
KBS_FIG_DIR = os.path.join(PAPER_DIR, 'Knowledge-Based Systems', 'fig')
ROOT_DIR = os.path.dirname(PAPER_DIR)

for d in [FIG_DIR, KBS_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Canvas Setup
W, H = 13.28, 5.31
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')

# Clean neutral background
ax.fill_between([0, W], 0, H, color='#F8FAFC', zorder=0)

# Color Palette
DARK_HEADER = '#0F172A'
TEXT_MUTED  = '#475569'
BORDER_COL  = '#CBD5E1'

# Theme Colors
C_ROSE       = '#E11D48'
C_ROSE_BG    = '#FFF1F2'
C_ROSE_BDR   = '#FECDD3'

C_BLUE       = '#2563EB'
C_BLUE_BG    = '#EFF6FF'
C_BLUE_BDR   = '#BFDBFE'

C_INDIGO     = '#4F46E5'
C_INDIGO_BG  = '#EEF2FF'
C_INDIGO_BDR = '#C7D2FE'

C_EMERALD    = '#059669'
C_EMERALD_BG = '#ECFDF5'
C_EMERALD_BDR= '#A7F3D0'

C_AMBER      = '#D97706'
C_AMBER_BG   = '#FFFBEB'
C_AMBER_BDR  = '#FDE68A'

def draw_card(ax, x, y, w, h, bg_color, border_color=BORDER_COL, radius=0.14, zorder=1, lw=1.2):
    card = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=bg_color, edgecolor=border_color,
                          linewidth=lw, zorder=zorder)
    ax.add_patch(card)
    return card

def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', width=2.2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=6,head_width=4',
                            color=color, linewidth=width, zorder=6)
    ax.add_patch(arrow)

# ==============================================================================
# VECTOR ICON DRAWING FUNCTIONS
# ==============================================================================

def draw_eye_icon(ax, cx, cy, r=0.22, color='#E11D48'):
    """Ophthalmic fundus eye icon."""
    verts = [
        (cx - r*1.3, cy),
        (cx, cy + r*0.9),
        (cx + r*1.3, cy),
        (cx, cy - r*0.9),
        (cx - r*1.3, cy)
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='white', edgecolor=color, lw=1.8, zorder=4)
    ax.add_patch(patch)
    iris = Circle((cx, cy), r*0.48, facecolor=color, edgecolor='none', zorder=5)
    ax.add_patch(iris)
    pupil = Circle((cx, cy), r*0.22, facecolor='#0F172A', edgecolor='none', zorder=6)
    ax.add_patch(pupil)
    hl = Circle((cx + r*0.12, cy + r*0.12), r*0.08, facecolor='white', edgecolor='none', zorder=7)
    ax.add_patch(hl)

def draw_smartwatch_icon(ax, cx, cy, w=0.30, h=0.38, color='#D97706'):
    """Wearable smartwatch with active pulse wave."""
    strap_w = w * 0.55
    strap_h = h * 0.28
    s_top = FancyBboxPatch((cx - strap_w/2, cy + h/2 - 0.02), strap_w, strap_h,
                           boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor='#78350F', edgecolor='none', zorder=3)
    s_bot = FancyBboxPatch((cx - strap_w/2, cy - h/2 - strap_h + 0.02), strap_w, strap_h,
                           boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor='#78350F', edgecolor='none', zorder=3)
    ax.add_patch(s_top)
    ax.add_patch(s_bot)
    case = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0,rounding_size=0.08",
                          facecolor='#1E293B', edgecolor=color, lw=1.6, zorder=4)
    ax.add_patch(case)
    xs = np.linspace(cx - w*0.38, cx + w*0.38, 25)
    ys = cy + np.array([0, 0, 0.02, 0, -0.03, 0.12, -0.08, 0.03, 0, 0, 0, 0.04, -0.05, 0.14, -0.10, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * (h / 0.4)
    ax.plot(xs, ys, color='#FDE047', lw=1.5, zorder=5)

def draw_lock_icon(ax, cx, cy, s=0.16, color='#2563EB'):
    """Padlock representing frozen foundation weights."""
    verts = [
        (cx - s*0.45, cy),
        (cx - s*0.45, cy + s*0.75),
        (cx + s*0.45, cy + s*0.75),
        (cx + s*0.45, cy)
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    shackle = PathPatch(Path(verts, codes), facecolor='none', edgecolor=color, lw=2.0, zorder=4)
    ax.add_patch(shackle)
    body = FancyBboxPatch((cx - s*0.6, cy - s*0.65), s*1.2, s*0.75,
                          boxstyle="round,pad=0,rounding_size=0.04",
                          facecolor=color, edgecolor='none', zorder=5)
    ax.add_patch(body)
    kh_c = Circle((cx, cy - s*0.22), s*0.12, facecolor='white', zorder=6)
    ax.add_patch(kh_c)

def draw_waves_icon(ax, cx, cy, w=0.35, color='#059669'):
    """Multi-frequency temporal biosignal wave lines."""
    xs = np.linspace(cx - w/2, cx + w/2, 40)
    y1 = cy + 0.06 * np.sin(np.linspace(0, 4*np.pi, 40))
    y2 = cy - 0.08 + 0.04 * np.sin(np.linspace(0, 6*np.pi, 40))
    ax.plot(xs, y1, color=color, lw=1.8, zorder=5)
    ax.plot(xs, y2, color='#34D399', lw=1.4, zorder=5)

def draw_stiefel_icon(ax, cx, cy, s=0.18, color='#4F46E5'):
    """Orthonormal subspace axes."""
    ax.annotate('', xy=(cx + s*0.85, cy - s*0.4), xytext=(cx - s*0.3, cy - s*0.4),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=5)
    ax.annotate('', xy=(cx - s*0.3, cy + s*0.75), xytext=(cx - s*0.3, cy - s*0.4),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=5)
    ra = PathPatch(Path([(cx - s*0.3, cy - s*0.1), (cx, cy - s*0.1), (cx, cy - s*0.4)]),
                   facecolor='none', edgecolor=color, lw=1.2, zorder=5)
    ax.add_patch(ra)

def draw_medical_cross_icon(ax, cx, cy, s=0.18, color='#0F172A'):
    """Medical cohort badge."""
    c = Circle((cx, cy), s, facecolor='#E2E8F0', edgecolor=color, lw=1.4, zorder=4)
    ax.add_patch(c)
    # Cross bars
    w_bar = s * 0.35
    l_bar = s * 1.15
    b1 = Rectangle((cx - w_bar/2, cy - l_bar/2), w_bar, l_bar, facecolor=color, edgecolor='none', zorder=5)
    b2 = Rectangle((cx - l_bar/2, cy - w_bar/2), l_bar, w_bar, facecolor=color, edgecolor='none', zorder=5)
    ax.add_patch(b1)
    ax.add_patch(b2)

def draw_target_icon(ax, cx, cy, r=0.20, color='#059669'):
    """Target bullseye badge."""
    c1 = Circle((cx, cy), r, facecolor=C_EMERALD_BG, edgecolor=color, lw=1.8, zorder=4)
    c2 = Circle((cx, cy), r*0.65, facecolor='white', edgecolor=color, lw=1.4, zorder=5)
    c3 = Circle((cx, cy), r*0.30, facecolor=color, edgecolor='none', zorder=6)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)

def draw_chip_icon(ax, cx, cy, s=0.20, color='#059669'):
    """Microprocessor chip."""
    chip = FancyBboxPatch((cx - s*0.6, cy - s*0.6), s*1.2, s*1.2,
                          boxstyle="round,pad=0,rounding_size=0.04",
                          facecolor='#1E293B', edgecolor=color, lw=1.6, zorder=4)
    ax.add_patch(chip)
    die = Rectangle((cx - s*0.3, cy - s*0.3), s*0.6, s*0.6, facecolor=color, edgecolor='none', zorder=5)
    ax.add_patch(die)
    pin_len = s * 0.25
    for offset in [-s*0.35, 0, s*0.35]:
        ax.plot([cx + offset, cx + offset], [cy + s*0.6, cy + s*0.6 + pin_len], color=color, lw=1.4, zorder=3)
        ax.plot([cx + offset, cx + offset], [cy - s*0.6, cy - s*0.6 - pin_len], color=color, lw=1.4, zorder=3)
        ax.plot([cx - s*0.6 - pin_len, cx - s*0.6], [cy + offset, cy + offset], color=color, lw=1.4, zorder=3)
        ax.plot([cx + s*0.6, cx + s*0.6 + pin_len], [cy + offset, cy + offset], color=color, lw=1.4, zorder=3)

def draw_network_icon(ax, cx, cy, r=0.20, color='#D97706'):
    """Decentralized federated edge network."""
    hub = Circle((cx, cy), r*0.32, facecolor=color, edgecolor='white', lw=1.5, zorder=6)
    ax.add_patch(hub)
    angles = [45, 135, 225, 315]
    for ang in angles:
        rad = np.radians(ang)
        px = cx + r * 1.05 * np.cos(rad)
        py = cy + r * 1.05 * np.sin(rad)
        ax.plot([cx, px], [cy, py], color='#CBD5E1', lw=1.2, linestyle='--', zorder=4)
        node = Circle((px, py), r*0.22, facecolor='#B45309', edgecolor='white', lw=1.2, zorder=5)
        ax.add_patch(node)


# ==============================================================================
# HEADER STRIP (Large, clean, impactful)
# ==============================================================================
ax.text(W/2, 4.96, "PEM-CAN: Parameter-Efficient Multimodal Diabetic Neuropathy Screening",
        ha='center', va='center', fontsize=15.0, fontweight='bold', color=DARK_HEADER)
ax.text(W/2, 4.65, "Orthogonal Low-Rank Subspace Cross-Attention over Retinal Imaging & Wearable Biosignals",
        ha='center', va='center', fontsize=10.2, color=TEXT_MUTED, style='italic')


# ==============================================================================
# 4 MAIN FUNCTIONAL PANELS
# ==============================================================================
PANEL_Y = 0.35
PANEL_H = 4.05

# ------------------------------------------------------------------------------
# PANEL 1: MULTIMODAL SENSORS
# ------------------------------------------------------------------------------
P1_X, P1_W = 0.35, 2.80
draw_card(ax, P1_X, PANEL_Y, P1_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P1_X + P1_W/2, 4.15, "1. MULTIMODAL SENSORS", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_ROSE)

# Card 1A: Retinal Window
draw_card(ax, P1_X + 0.15, 2.65, P1_W - 0.30, 1.30, C_ROSE_BG, border_color=C_ROSE_BDR)
draw_eye_icon(ax, P1_X + 0.50, 3.48, r=0.22, color=C_ROSE)
ax.text(P1_X + 0.85, 3.55, "Retinal Oculomics", fontsize=9.6, fontweight='bold', color='#9F1239')
ax.text(P1_X + 0.85, 3.35, "Cumulative Microvascular State", fontsize=7.6, color='#BE123C')
ax.text(P1_X + 0.22, 2.92, "• High-resolution fundus photography\n• Arteriolar narrowing & capillary rarefaction\n• Spatial tokens: M = 576",
        fontsize=8.2, color='#881337', va='center')

# Card 1B: Wearable Biosignals
draw_card(ax, P1_X + 0.15, 1.20, P1_W - 0.30, 1.35, C_AMBER_BG, border_color=C_AMBER_BDR)
draw_smartwatch_icon(ax, P1_X + 0.50, 2.02, w=0.30, h=0.38, color=C_AMBER)
ax.text(P1_X + 0.85, 2.10, "Wearable Telemetry", fontsize=9.6, fontweight='bold', color='#92400E')
ax.text(P1_X + 0.85, 1.90, "Dynamic Autonomic Excursions", fontsize=7.6, color='#B45309')
ax.text(P1_X + 0.22, 1.48, "• 7-day CGM (5-min, T = 2,016 steps)\n• 5-channel actigraphy: HRV, sleep, stress\n• Multiscale Conv1D tokens: L = 252",
        fontsize=8.2, color='#78350F', va='center')

# Bottom Challenge Callout
draw_card(ax, P1_X + 0.15, 0.45, P1_W - 0.30, 0.65, '#F1F5F9', border_color=BORDER_COL)
ax.text(P1_X + P1_W/2, 0.80, "Clinical Challenge", ha='center', fontsize=8.2, fontweight='bold', color='#334155')
ax.text(P1_X + P1_W/2, 0.60, "Visual dominance & quadratic attention blowup", ha='center', fontsize=7.4, color='#64748B')

draw_arrow(ax, P1_X + P1_W, 2.37, P1_X + P1_W + 0.20, 2.37, color=C_BLUE, width=2.4)


# ------------------------------------------------------------------------------
# PANEL 2: PEM-CAN ARCHITECTURE
# ------------------------------------------------------------------------------
P2_X, P2_W = 3.35, 3.65
draw_card(ax, P2_X, PANEL_Y, P2_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P2_X + P2_W/2, 4.15, "2. PEM-CAN SUBSPACE ADAPTERS", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_BLUE)

# Backbone 1: Frozen ViT
draw_card(ax, P2_X + 0.15, 2.95, 1.45, 0.95, C_BLUE_BG, border_color=C_BLUE_BDR)
draw_lock_icon(ax, P2_X + 0.40, 3.52, s=0.16, color=C_BLUE)
ax.text(P2_X + 0.68, 3.55, "Frozen ViT-B/16", fontsize=8.6, fontweight='bold', color='#1E40AF')
ax.text(P2_X + 0.22, 3.16, "86.6M Base Params\n(Zero Backprop)", fontsize=7.8, color='#1E3A8A')

# Backbone 2: Temporal Conv1D
draw_card(ax, P2_X + 0.15, 1.80, 1.45, 0.95, C_EMERALD_BG, border_color=C_EMERALD_BDR)
draw_waves_icon(ax, P2_X + 0.40, 2.45, w=0.30, color=C_EMERALD)
ax.text(P2_X + 0.68, 2.48, "Temporal Conv", fontsize=8.6, fontweight='bold', color='#065F46')
ax.text(P2_X + 0.22, 2.05, "Multiscale Tokenizer\n(L = 252 steps)", fontsize=7.8, color='#047857')

# Adapter Box: Stiefel LoRA
draw_card(ax, P2_X + 0.15, 0.50, 1.45, 1.15, C_INDIGO_BG, border_color=C_INDIGO_BDR)
draw_stiefel_icon(ax, P2_X + 0.35, 1.25, s=0.17, color=C_INDIGO)
ax.text(P2_X + 0.65, 1.30, "Stiefel LoRA", fontsize=8.6, fontweight='bold', color='#3730A3')
ax.text(P2_X + 0.18, 0.82, "||AAᵀ - I_r||_F² Penalty\nRank r = 8 Subspace\n98.3k Adapter Weights", fontsize=7.4, color='#312E81')

# Fusion Engine Card
draw_card(ax, P2_X + 1.80, 1.70, 1.70, 2.20, '#EEF2FF', border_color=C_INDIGO_BDR)
ax.text(P2_X + 2.65, 3.65, "Bidirectional Cross-Attn", fontsize=9.0, fontweight='bold', color='#3730A3', ha='center')
ax.text(P2_X + 2.65, 3.40, "S(Z_v, Z_s) & S(Z_s, Z_v)", fontsize=7.8, color='#4338CA', ha='center')
ax.text(P2_X + 1.95, 2.85, "• Linear Complexity: O(M·L)\n• Prevents Modality Collapse\n• Zero-Overhead Inference\n• Adapters fold into W_0",
        fontsize=8.0, color='#312E81', va='center')
ax.text(P2_X + 2.65, 1.95, "L_total = L_focal + L_orth + L_align", fontsize=6.8, color='#4F46E5', ha='center')

# Multi-Task Classifier Head
draw_card(ax, P2_X + 1.80, 0.50, 1.70, 1.05, '#1E293B', border_color='#0F172A')
ax.text(P2_X + 2.65, 1.28, "Multi-Task Head", fontsize=8.8, fontweight='bold', color='#FFFFFF', ha='center')
ax.text(P2_X + 2.65, 0.85, "DAN (Autonomic) | DR (Retina)\nCKD (Renal) | DPN (Sensory)", fontsize=7.5, color='#CBD5E1', ha='center')

# Connecting Arrows
draw_arrow(ax, P2_X + 1.60, 3.35, P2_X + 1.80, 3.10, color=C_BLUE, width=1.4)
draw_arrow(ax, P2_X + 1.60, 2.30, P2_X + 1.80, 2.50, color=C_EMERALD, width=1.4)
draw_arrow(ax, P2_X + 1.60, 1.10, P2_X + 1.80, 1.90, color=C_INDIGO, width=1.4)
draw_arrow(ax, P2_X + 2.65, 1.70, P2_X + 2.65, 1.55, color='#0F172A', width=1.4)

draw_arrow(ax, P2_X + P2_W, 2.37, P2_X + P2_W + 0.20, 2.37, color=C_EMERALD, width=2.4)


# ------------------------------------------------------------------------------
# PANEL 3: BENCHMARK COHORT
# ------------------------------------------------------------------------------
P3_X, P3_W = 7.20, 2.70
draw_card(ax, P3_X, PANEL_Y, P3_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P3_X + P3_W/2, 4.15, "3. BENCHMARK COHORT", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=C_EMERALD)

# Protocol Card
draw_card(ax, P3_X + 0.15, 2.30, P3_W - 0.30, 1.65, '#F8FAFC', border_color=BORDER_COL)
draw_medical_cross_icon(ax, P3_X + 0.45, 3.52, s=0.18, color='#0F172A')
ax.text(P3_X + 0.75, 3.62, "AI-READI T2D Protocol", fontsize=9.4, fontweight='bold', color='#0F172A')
ax.text(P3_X + 0.75, 3.40, "N = 2,840 Patients", fontsize=8.0, color='#475569')
ax.text(P3_X + 0.22, 2.82, "• Statistically mapped to NIH atlas\n• All confirmed Type 2 Diabetes\n• Stratified 5-Fold Cross-Validation\n• White (60%), Black (25%), Lat (15%)",
        fontsize=8.0, color='#334155', va='center')

# Endpoints Card
draw_card(ax, P3_X + 0.15, 0.50, P3_W - 0.30, 1.65, C_EMERALD_BG, border_color=C_EMERALD_BDR)
draw_target_icon(ax, P3_X + 0.45, 1.76, r=0.18, color=C_EMERALD)
ax.text(P3_X + 0.75, 1.84, "Adjudicated Targets", fontsize=9.4, fontweight='bold', color='#065F46')
ax.text(P3_X + 0.75, 1.64, "Clinically Validated Labels", fontsize=7.8, color='#047857')
ax.text(P3_X + 0.22, 1.05, "• DAN (Autonomic): 22.5% Prev.\n• DR (Retinopathy): 18.2% Prev.\n• CKD (Renal): 14.1% Prev.\n• DPN (Peripheral): 26.8% Prev.",
        fontsize=8.0, color='#064E3B', va='center')

draw_arrow(ax, P3_X + P3_W, 2.37, P3_X + P3_W + 0.20, 2.37, color=C_BLUE, width=2.4)


# ------------------------------------------------------------------------------
# PANEL 4: QUANTIFIED RESULTS (Stat Cards with large bold numbers)
# ------------------------------------------------------------------------------
P4_X, P4_W = 10.10, 2.83
draw_card(ax, P4_X, PANEL_Y, P4_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P4_X + P4_W/2, 4.15, "4. QUANTIFIED PERFORMANCE", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=DARK_HEADER)

# Card 4A: Accuracy
draw_card(ax, P4_X + 0.12, 2.90, P4_W - 0.24, 1.05, C_BLUE_BG, border_color=C_BLUE_BDR)
draw_target_icon(ax, P4_X + 0.40, 3.42, r=0.18, color=C_BLUE)
ax.text(P4_X + 0.70, 3.58, "0.934 AUROC", fontsize=12.5, fontweight='bold', color='#1E40AF')
ax.text(P4_X + 0.70, 3.34, "90.5% Accuracy | 0.803 F1", fontsize=8.4, fontweight='bold', color='#1E3A8A')
ax.text(P4_X + 0.20, 3.08, "• Oracle Ceiling: 0.957  • ECE: 0.038", fontsize=7.8, color='#1D4ED8')

# Card 4B: Efficiency
draw_card(ax, P4_X + 0.12, 1.70, P4_W - 0.24, 1.05, C_EMERALD_BG, border_color=C_EMERALD_BDR)
draw_chip_icon(ax, P4_X + 0.40, 2.22, s=0.18, color=C_EMERALD)
ax.text(P4_X + 0.70, 2.38, "1.75% Weights (2.6M)", fontsize=11.5, fontweight='bold', color='#047857')
ax.text(P4_X + 0.70, 2.14, "8.4 GB VRAM (vs. 32.4 GB)", fontsize=8.4, fontweight='bold', color='#065F46')
ax.text(P4_X + 0.20, 1.88, "• 0.861 AUROC under 50% sensor loss", fontsize=7.8, color='#047857')

# Card 4C: Edge & Federated
draw_card(ax, P4_X + 0.12, 0.50, P4_W - 0.24, 1.05, C_AMBER_BG, border_color=C_AMBER_BDR)
draw_network_icon(ax, P4_X + 0.40, 1.02, r=0.18, color=C_AMBER)
ax.text(P4_X + 0.70, 1.18, "20.2 ms Edge Latency", fontsize=11.5, fontweight='bold', color='#92400E')
ax.text(P4_X + 0.70, 0.94, "NVIDIA Jetson Orin Nano (1.18 GB)", fontsize=8.2, fontweight='bold', color='#B45309')
ax.text(P4_X + 0.20, 0.68, "• 50-Node FL: 98.25% less bandwidth", fontsize=7.8, color='#78350F')


# ==============================================================================
# SAVE & EXPORT
# ==============================================================================
out_paths = [
    (os.path.join(FIG_DIR, 'graphical_abstract.png'), os.path.join(FIG_DIR, 'graphical_abstract.pdf')),
    (os.path.join(KBS_FIG_DIR, 'graphical_abstract.png'), os.path.join(KBS_FIG_DIR, 'graphical_abstract.pdf')),
    (os.path.join(ROOT_DIR, 'graphical_abstract.png'), os.path.join(ROOT_DIR, 'graphical_abstract.pdf'))
]

for png_p, pdf_p in out_paths:
    plt.savefig(png_p, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
    plt.savefig(pdf_p, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
    print(f"Saved: {png_p}")

plt.close()
print("Enhanced graphical abstract generated successfully!")
