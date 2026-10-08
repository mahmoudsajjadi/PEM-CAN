import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle, 
                                PathPatch, Rectangle)
from matplotlib.path import Path

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PAPER_DIR, 'fig')
KBS_FIG_DIR = os.path.join(PAPER_DIR, 'Knowledge-Based Systems', 'fig')
ROOT_DIR = os.path.dirname(PAPER_DIR)
SUBMISSION_DIR = os.path.join(PAPER_DIR, 'Knowledge-Based Systems', 'submission_files')

for d in [FIG_DIR, KBS_FIG_DIR, SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)

# Canvas Setup: 13.5 x 5.4 inches at 300 DPI (Elsevier CAS standard 5:2 ratio)
W, H = 13.50, 5.40
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

def draw_card(ax, x, y, w, h, bg_color, border_color=BORDER_COL, radius=0.12, zorder=1, lw=1.2):
    card = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=bg_color, edgecolor=border_color,
                          linewidth=lw, zorder=zorder)
    ax.add_patch(card)
    return card

def draw_pill(ax, cx, cy, w, h, bg_color, border_color=BORDER_COL, text="", text_color='#1E293B', fontsize=7.5, fontweight='bold', zorder=3):
    pill = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={min(h/2, 0.12)}",
                          facecolor=bg_color, edgecolor=border_color,
                          linewidth=0.9, zorder=zorder)
    ax.add_patch(pill)
    if text:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder+1)

def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', width=2.4):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=7,head_width=4.5',
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
        (cx - r*1.3, cy),
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='white', edgecolor=color, lw=1.6, zorder=4)
    ax.add_patch(patch)
    iris = Circle((cx, cy), r*0.55, facecolor=color, edgecolor='none', zorder=5)
    pupil = Circle((cx, cy), r*0.28, facecolor='#1E293B', edgecolor='none', zorder=6)
    glint = Circle((cx + r*0.14, cy + r*0.14), r*0.10, facecolor='white', edgecolor='none', zorder=7)
    ax.add_patch(iris)
    ax.add_patch(pupil)
    ax.add_patch(glint)

def draw_smartwatch_icon(ax, cx, cy, w=0.28, h=0.36, color='#D97706'):
    """Smartwatch with biosensor display."""
    strap_top = Rectangle((cx - w*0.35, cy + h*0.48), w*0.7, h*0.32,
                          facecolor='#B45309', edgecolor='none', zorder=3)
    strap_bot = Rectangle((cx - w*0.35, cy - h*0.80), w*0.7, h*0.32,
                          facecolor='#B45309', edgecolor='none', zorder=3)
    ax.add_patch(strap_top)
    ax.add_patch(strap_bot)
    case = FancyBboxPatch((cx - w*0.5, cy - h*0.5), w, h,
                          boxstyle="round,pad=0,rounding_size=0.08",
                          facecolor='#1E293B', edgecolor=color, lw=1.5, zorder=4)
    ax.add_patch(case)
    pts_x = [cx - w*0.35, cx - w*0.15, cx - w*0.05, cx + w*0.08, cx + w*0.20, cx + w*0.35]
    pts_y = [cy, cy, cy + h*0.25, cy - h*0.28, cy + h*0.10, cy]
    ax.plot(pts_x, pts_y, color=color, lw=1.6, zorder=5)

def draw_lock_icon(ax, cx, cy, s=0.18, color='#2563EB'):
    """Padlock icon for frozen foundation backbone."""
    shackle = FancyBboxPatch((cx - s*0.4, cy), s*0.8, s*0.7,
                             boxstyle="round,pad=0,rounding_size=0.12",
                             facecolor='none', edgecolor=color, lw=1.8, zorder=4)
    ax.add_patch(shackle)
    body = FancyBboxPatch((cx - s*0.55, cy - s*0.6), s*1.1, s*0.75,
                          boxstyle="round,pad=0,rounding_size=0.06",
                          facecolor=color, edgecolor='white', lw=1.2, zorder=5)
    ax.add_patch(body)
    hole = Circle((cx, cy - s*0.25), s*0.15, facecolor='white', edgecolor='none', zorder=6)
    ax.add_patch(hole)

def draw_waves_icon(ax, cx, cy, w=0.32, color='#059669'):
    """Multi-frequency temporal biosignal wave."""
    xs = np.linspace(cx - w*0.6, cx + w*0.6, 60)
    ys1 = cy + 0.08 * np.sin(np.linspace(0, 3*np.pi, 60))
    ys2 = cy - 0.07 * np.cos(np.linspace(0, 4*np.pi, 60))
    ax.plot(xs, ys1, color=color, lw=1.8, zorder=4)
    ax.plot(xs, ys2, color='#10B981', lw=1.2, linestyle='--', zorder=4)

def draw_stiefel_icon(ax, cx, cy, s=0.18, color='#4F46E5'):
    """Orthonormal subspace axes."""
    ax.annotate('', xy=(cx + s*0.9, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=5)
    ax.annotate('', xy=(cx, cy + s*0.9), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=5)
    sq = Rectangle((cx, cy), s*0.25, s*0.25, facecolor='none', edgecolor=color, lw=1.0, zorder=4)
    ax.add_patch(sq)

def draw_medical_cross_icon(ax, cx, cy, s=0.18, color='#0F172A'):
    """Medical clinical cross badge."""
    bg = Circle((cx, cy), s*1.15, facecolor='#E2E8F0', edgecolor=BORDER_COL, lw=1.2, zorder=3)
    ax.add_patch(bg)
    b1 = Rectangle((cx - s*0.22, cy - s*0.7), s*0.44, s*1.4, facecolor=color, edgecolor='none', zorder=4)
    b2 = Rectangle((cx - s*0.7, cy - s*0.22), s*1.4, s*0.44, facecolor=color, edgecolor='none', zorder=4)
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
# HEADER STRIP (Clear, impactful, well-spaced)
# ==============================================================================
ax.text(W/2, 5.06, "PEM-CAN: Parameter-Efficient Multimodal Diabetic Neuropathy Screening",
        ha='center', va='center', fontsize=16.0, fontweight='bold', color=DARK_HEADER)
ax.text(W/2, 4.75, "Orthogonal Low-Rank Subspace Cross-Attention over Retinal Imaging & Wearable Biosignals",
        ha='center', va='center', fontsize=10.5, color=TEXT_MUTED, style='italic')


# ==============================================================================
# 4 MAIN FUNCTIONAL PANELS
# ==============================================================================
PANEL_Y = 0.35
PANEL_H = 4.10

# ------------------------------------------------------------------------------
# PANEL 1: MULTIMODAL INPUTS (Width = 2.80)
# ------------------------------------------------------------------------------
P1_X, P1_W = 0.40, 2.80
draw_card(ax, P1_X, PANEL_Y, P1_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P1_X + P1_W/2, 4.22, "1. MULTIMODAL INPUTS", ha='center', va='center',
        fontsize=11.2, fontweight='bold', color=C_ROSE)

# Card 1A: Retinal Fundus (Microvascular)
draw_card(ax, P1_X + 0.16, 2.55, P1_W - 0.32, 1.45, C_ROSE_BG, border_color=C_ROSE_BDR)
draw_eye_icon(ax, P1_X + 0.50, 3.52, r=0.22, color=C_ROSE)
ax.text(P1_X + 0.84, 3.60, "Retinal Oculomics", fontsize=10.2, fontweight='bold', color='#9F1239')
ax.text(P1_X + 0.84, 3.38, "Microvascular Architecture", fontsize=8.0, color='#BE123C')
ax.text(P1_X + P1_W/2, 3.00, "M = 576 Spatial Tokens", fontsize=8.8, fontweight='bold', color='#881337', ha='center')
draw_pill(ax, P1_X + P1_W/2, 2.72, 2.10, 0.28, 'white', border_color=C_ROSE_BDR,
          text="Color Fundus Photography (2D)", text_color='#9F1239', fontsize=7.6)

# Card 1B: Wearable Biosignals (Dynamic Autonomic)
draw_card(ax, P1_X + 0.16, 0.95, P1_W - 0.32, 1.45, C_AMBER_BG, border_color=C_AMBER_BDR)
draw_smartwatch_icon(ax, P1_X + 0.50, 1.92, w=0.28, h=0.36, color=C_AMBER)
ax.text(P1_X + 0.84, 2.00, "Wearable Telemetry", fontsize=10.2, fontweight='bold', color='#92400E')
ax.text(P1_X + 0.84, 1.78, "Autonomic Dynamics", fontsize=8.0, color='#B45309')
ax.text(P1_X + P1_W/2, 1.40, "L = 252 Signal Steps", fontsize=8.8, fontweight='bold', color='#78350F', ha='center')
draw_pill(ax, P1_X + P1_W/2, 1.12, 2.10, 0.28, 'white', border_color=C_AMBER_BDR,
          text="Continuous CGM & Actigraphy (1D)", text_color='#92400E', fontsize=7.6)

# Bottom Callout: Modality Asymmetry Challenge
draw_pill(ax, P1_X + P1_W/2, 0.60, P1_W - 0.32, 0.36, '#F1F5F9', border_color=BORDER_COL,
          text="Clinical Challenge: Modality Asymmetry", text_color='#475569', fontsize=7.6)

draw_arrow(ax, P1_X + P1_W + 0.04, 2.40, P1_X + P1_W + 0.24, 2.40, color=C_BLUE, width=2.4)


# ------------------------------------------------------------------------------
# PANEL 2: PEM-CAN ARCHITECTURE (Width = 3.65) - TOP-TO-BOTTOM INTUITIVE FLOW
# ------------------------------------------------------------------------------
P2_X, P2_W = 3.48, 3.65
draw_card(ax, P2_X, PANEL_Y, P2_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P2_X + P2_W/2, 4.22, "2. PEM-CAN ARCHITECTURE", ha='center', va='center',
        fontsize=11.2, fontweight='bold', color=C_BLUE)

# Row 1: Dual Pretrained Encoders (Side-by-side)
ENC_W = 1.60
# Left Encoder: Frozen ViT
draw_card(ax, P2_X + 0.16, 2.92, ENC_W, 1.08, C_BLUE_BG, border_color=C_BLUE_BDR)
draw_lock_icon(ax, P2_X + 0.40, 3.60, s=0.15, color=C_BLUE)
ax.text(P2_X + 0.62, 3.62, "Frozen ViT-B/16", fontsize=8.6, fontweight='bold', color='#1E40AF')
ax.text(P2_X + 0.16 + ENC_W/2, 3.32, "86.6M Frozen Weights", fontsize=7.6, color='#1E3A8A', ha='center')
draw_pill(ax, P2_X + 0.16 + ENC_W/2, 3.08, 1.34, 0.24, 'white', border_color=C_BLUE_BDR,
          text="Vision Tokens (M)", text_color='#1E40AF', fontsize=7.0)

# Right Encoder: Temporal ConvNet
draw_card(ax, P2_X + P2_W - 0.16 - ENC_W, 2.92, ENC_W, 1.08, C_EMERALD_BG, border_color=C_EMERALD_BDR)
draw_waves_icon(ax, P2_X + P2_W - 0.16 - ENC_W + 0.24, 3.60, w=0.24, color=C_EMERALD)
ax.text(P2_X + P2_W - 0.16 - ENC_W + 0.44, 3.62, "Temporal ConvNet", fontsize=8.6, fontweight='bold', color='#065F46')
ax.text(P2_X + P2_W - 0.16 - ENC_W/2, 3.32, "Multiscale Tokenizer", fontsize=7.6, color='#047857', ha='center')
draw_pill(ax, P2_X + P2_W - 0.16 - ENC_W/2, 3.08, 1.34, 0.24, 'white', border_color=C_EMERALD_BDR,
          text="Signal Tokens (L)", text_color='#065F46', fontsize=7.0)

# Downward Arrows from Encoders to Cross-Attention Core
draw_arrow(ax, P2_X + 0.16 + ENC_W/2, 2.92, P2_X + 0.16 + ENC_W/2, 2.68, color=C_BLUE, width=1.8)
draw_arrow(ax, P2_X + P2_W - 0.16 - ENC_W/2, 2.92, P2_X + P2_W - 0.16 - ENC_W/2, 2.68, color=C_EMERALD, width=1.8)

# Row 2: Orthogonal Subspace Cross-Attention (Wide central card)
CAN_W = P2_W - 0.32
draw_card(ax, P2_X + 0.16, 1.45, CAN_W, 1.22, '#EEF2FF', border_color=C_INDIGO_BDR)
draw_stiefel_icon(ax, P2_X + 0.40, 2.38, s=0.15, color=C_INDIGO)
ax.text(P2_X + P2_W/2 + 0.10, 2.42, "Orthogonal Subspace Cross-Attention", fontsize=9.2, fontweight='bold', color='#3730A3', ha='center')
ax.text(P2_X + P2_W/2, 2.22, "Bidirectional Token Interaction with Stiefel Constraint", fontsize=7.6, color='#4338CA', ha='center', style='italic')

# 2 Horizontal Badges inside CAN
draw_pill(ax, P2_X + 0.16 + 0.85, 1.88, 1.45, 0.30, 'white', border_color=C_INDIGO_BDR,
          text="Stiefel LoRA (r = 8)", text_color='#3730A3', fontsize=7.6)
draw_pill(ax, P2_X + P2_W - 0.16 - 0.85, 1.88, 1.45, 0.30, 'white', border_color=C_INDIGO_BDR,
          text="Linear Complexity O(M·L)", text_color='#3730A3', fontsize=7.6)
ax.text(P2_X + P2_W/2, 1.60, "Adapters Fold into Base Weights (Zero Extra Latency)", fontsize=7.2, color='#312E81', ha='center')

# Downward Arrow to Multi-Task Head
draw_arrow(ax, P2_X + P2_W/2, 1.45, P2_X + P2_W/2, 1.20, color='#0F172A', width=1.8)

# Row 3: Multi-Task Clinical Head (Dark Slate)
draw_card(ax, P2_X + 0.16, 0.46, CAN_W, 0.74, '#1E293B', border_color='#0F172A')
ax.text(P2_X + P2_W/2, 0.98, "Multi-Task Clinical Screening Head", fontsize=9.2, fontweight='bold', color='#FFFFFF', ha='center')
ax.text(P2_X + P2_W/2, 0.78, "DAN (Autonomic)   •   DPN (Peripheral)", fontsize=7.6, fontweight='bold', color='#38BDF8', ha='center')
ax.text(P2_X + P2_W/2, 0.60, "DR (Retinopathy)   •   CKD (Nephropathy)", fontsize=7.6, fontweight='bold', color='#38BDF8', ha='center')

draw_arrow(ax, P2_X + P2_W + 0.04, 2.40, P2_X + P2_W + 0.24, 2.40, color=C_EMERALD, width=2.4)


# ------------------------------------------------------------------------------
# PANEL 3: CLINICAL BENCHMARK (Width = 2.70)
# ------------------------------------------------------------------------------
P3_X, P3_W = 7.42, 2.70
draw_card(ax, P3_X, PANEL_Y, P3_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P3_X + P3_W/2, 4.22, "3. CLINICAL BENCHMARK", ha='center', va='center',
        fontsize=11.2, fontweight='bold', color=C_EMERALD)

# AI-READI Cohort Card
draw_card(ax, P3_X + 0.15, 2.28, P3_W - 0.30, 1.72, '#F8FAFC', border_color=BORDER_COL)
draw_medical_cross_icon(ax, P3_X + 0.46, 3.62, s=0.17, color='#0F172A')
ax.text(P3_X + 0.74, 3.70, "AI-READI Cohort", fontsize=10.0, fontweight='bold', color='#0F172A')
ax.text(P3_X + 0.74, 3.48, "NIH Multi-Ethnic Protocol", fontsize=7.8, color='#64748B')
ax.text(P3_X + P3_W/2, 3.08, "N = 2,840 Patients", fontsize=12.5, fontweight='bold', color='#0F172A', ha='center')
draw_pill(ax, P3_X + P3_W/2, 2.66, 2.05, 0.28, 'white', border_color=BORDER_COL,
          text="Confirmed Type 2 Diabetes", text_color='#334155', fontsize=7.6)
draw_pill(ax, P3_X + P3_W/2, 2.40, 2.05, 0.24, 'white', border_color=BORDER_COL,
          text="Stratified 5-Fold Validation", text_color='#475569', fontsize=7.2)

# Clinical Endpoints Card
draw_card(ax, P3_X + 0.15, 0.46, P3_W - 0.30, 1.68, C_EMERALD_BG, border_color=C_EMERALD_BDR)
draw_target_icon(ax, P3_X + 0.46, 1.78, r=0.17, color=C_EMERALD)
ax.text(P3_X + 0.74, 1.86, "Target Complications", fontsize=10.0, fontweight='bold', color='#065F46')
ax.text(P3_X + 0.74, 1.64, "Clinically Validated Labels", fontsize=7.8, color='#047857')

# 4 Complication badges in 2x2 grid
badge_w, badge_h = 1.05, 0.32
b_x1 = P3_X + 0.15 + 0.58
b_x2 = P3_X + P3_W - 0.15 - 0.58
draw_pill(ax, b_x1, 1.20, badge_w, badge_h, 'white', border_color=C_EMERALD_BDR,
          text="DAN (22.5%)", text_color='#065F46', fontsize=7.8)
draw_pill(ax, b_x2, 1.20, badge_w, badge_h, 'white', border_color=C_EMERALD_BDR,
          text="DPN (26.8%)", text_color='#065F46', fontsize=7.8)
draw_pill(ax, b_x1, 0.76, badge_w, badge_h, 'white', border_color=C_EMERALD_BDR,
          text="DR (18.2%)", text_color='#065F46', fontsize=7.8)
draw_pill(ax, b_x2, 0.76, badge_w, badge_h, 'white', border_color=C_EMERALD_BDR,
          text="CKD (14.1%)", text_color='#065F46', fontsize=7.8)

draw_arrow(ax, P3_X + P3_W + 0.04, 2.40, P3_X + P3_W + 0.24, 2.40, color=C_BLUE, width=2.4)


# ------------------------------------------------------------------------------
# PANEL 4: QUANTIFIED RESULTS (Width = 2.70)
# ------------------------------------------------------------------------------
P4_X, P4_W = 10.40, 2.70
draw_card(ax, P4_X, PANEL_Y, P4_W, PANEL_H, '#FFFFFF', border_color=BORDER_COL)
ax.text(P4_X + P4_W/2, 4.22, "4. QUANTIFIED FINDINGS", ha='center', va='center',
        fontsize=11.2, fontweight='bold', color=DARK_HEADER)

# Card 4A: Diagnostic Discrimination
draw_card(ax, P4_X + 0.14, 2.90, P4_W - 0.28, 1.10, C_BLUE_BG, border_color=C_BLUE_BDR)
draw_target_icon(ax, P4_X + 0.42, 3.45, r=0.17, color=C_BLUE)
ax.text(P4_X + 0.70, 3.65, "0.934 AUROC", fontsize=13.0, fontweight='bold', color='#1E40AF')
ax.text(P4_X + 0.70, 3.40, "90.5% Acc  |  0.803 F1", fontsize=8.6, fontweight='bold', color='#1E3A8A')
draw_pill(ax, P4_X + P4_W/2, 3.08, 2.15, 0.26, 'white', border_color=C_BLUE_BDR,
          text="Surpasses Full Fine-Tuning (0.912)", text_color='#1E40AF', fontsize=7.2)

# Card 4B: Parameter & Memory Efficiency
draw_card(ax, P4_X + 0.14, 1.68, P4_W - 0.28, 1.10, C_EMERALD_BG, border_color=C_EMERALD_BDR)
draw_chip_icon(ax, P4_X + 0.42, 2.23, s=0.17, color=C_EMERALD)
ax.text(P4_X + 0.70, 2.43, "1.75% Weights", fontsize=13.0, fontweight='bold', color='#047857')
ax.text(P4_X + 0.70, 2.18, "2.6M Tunable (vs 86.6M)", fontsize=8.6, fontweight='bold', color='#065F46')
draw_pill(ax, P4_X + P4_W/2, 1.86, 2.15, 0.26, 'white', border_color=C_EMERALD_BDR,
          text="8.4 GB VRAM Footprint (-74%)", text_color='#047857', fontsize=7.2)

# Card 4C: Edge Latency & Federated Scaling
draw_card(ax, P4_X + 0.14, 0.46, P4_W - 0.28, 1.10, C_AMBER_BG, border_color=C_AMBER_BDR)
draw_network_icon(ax, P4_X + 0.42, 1.01, r=0.17, color=C_AMBER)
ax.text(P4_X + 0.70, 1.21, "20.2 ms / Patient", fontsize=13.0, fontweight='bold', color='#92400E')
ax.text(P4_X + 0.70, 0.96, "Jetson Orin Nano (Edge)", fontsize=8.6, fontweight='bold', color='#B45309')
draw_pill(ax, P4_X + P4_W/2, 0.64, 2.15, 0.26, 'white', border_color=C_AMBER_BDR,
          text="98.25% Bandwidth Reduction", text_color='#92400E', fontsize=7.2)


# ==============================================================================
# SAVE & EXPORT ACROSS ALL LOCATIONS
# ==============================================================================
out_paths = [
    (os.path.join(FIG_DIR, 'graphical_abstract.png'), os.path.join(FIG_DIR, 'graphical_abstract.pdf')),
    (os.path.join(KBS_FIG_DIR, 'graphical_abstract.png'), os.path.join(KBS_FIG_DIR, 'graphical_abstract.pdf')),
    (os.path.join(ROOT_DIR, 'graphical_abstract.png'), os.path.join(ROOT_DIR, 'graphical_abstract.pdf')),
    (os.path.join(SUBMISSION_DIR, 'Graphical_Abstract.png'), os.path.join(SUBMISSION_DIR, 'Graphical_Abstract.pdf'))
]

for png_p, pdf_p in out_paths:
    plt.savefig(png_p, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
    plt.savefig(pdf_p, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
    print(f"Saved: {png_p}")

plt.close()
print("Clean, spacious graphical abstract generated successfully!")
