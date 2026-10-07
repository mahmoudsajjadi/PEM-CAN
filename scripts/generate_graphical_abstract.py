"""
Standalone publication-grade script to generate Elsevier-compliant Graphical Abstract.
Dimensions: 1328 x 531 aspect ratio, 300 DPI, modern vector aesthetics.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import matplotlib.patheffects as pe

# Ensure output directory exists
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'fig')
os.makedirs(FIG_DIR, exist_ok=True)

# Elsevier standard ratio: 1328 x 531 (w x h)
FIG_WIDTH = 13.28
FIG_HEIGHT = 5.31

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13.28)
ax.set_ylim(0, 5.31)
ax.axis('off')

# Background
ax.fill_between([0, 13.28], 0, 5.31, color='#F8FAFC', zorder=0)

# Color Palette (Modern Nature/Elsevier Clinical Aesthetics)
COL_HEADER = '#0F172A'
COL_SUB = '#475569'
COL_BLUE = '#2563EB'
COL_BLUE_LIGHT = '#EFF6FF'
COL_EMERALD = '#059669'
COL_EMERALD_LIGHT = '#ECFDF5'
COL_AMBER = '#D97706'
COL_AMBER_LIGHT = '#FFFBEB'
COL_ROSE = '#E11D48'
COL_ROSE_LIGHT = '#FFF1F2'
COL_BORDER = '#CBD5E1'

# Helper function for drawing elegant cards
def draw_card(ax, x, y, w, h, bg_color, border_color='#CBD5E1', radius=0.15, zorder=1):
    card = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=bg_color, edgecolor=border_color,
                          linewidth=1.2, zorder=zorder)
    ax.add_patch(card)
    return card

def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', width=1.8, style='simple'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=5,head_width=3',
                            color=color, linewidth=width, zorder=5)
    ax.add_patch(arrow)

# ==============================================================================
# HEADER STRIP
# ==============================================================================
ax.text(6.64, 4.95, "BioSync: Wide-and-Deep Multimodal Token Fusion for Edge Wearables",
        ha='center', va='center', fontsize=14, fontweight='bold', color=COL_HEADER)
ax.text(6.64, 4.65, "Ultra-Lightweight Attention & Linear Bypass over Heterogeneous Ambient Biosignals",
        ha='center', va='center', fontsize=9.5, color=COL_SUB, style='italic')

# Column Boundaries (4 Primary Narrative Panels)
# Panel 1: (0.35 -> 3.25)  Problem & Sensor Realities
# Panel 2: (3.45 -> 7.05)  Wide-and-Deep Token Architecture
# Panel 3: (7.25 -> 9.95)  Clinical Schemas & AI-READI Mapping
# Panel 4: (10.15 -> 12.95) Quantified Results & Edge Footprint

# ==============================================================================
# PANEL 1: THE CLINICAL PROBLEM & EDGE CONSTRAINTS
# ==============================================================================
draw_card(ax, 0.35, 0.35, 2.95, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(1.82, 4.15, "1. AMBIENT SENSING REALITY", ha='center', va='center',
        fontsize=9, fontweight='bold', color=COL_ROSE)

# Sub-boxes for Problem Points
draw_card(ax, 0.50, 2.95, 2.65, 0.95, COL_ROSE_LIGHT, border_color='#FECDD3')
ax.text(0.60, 3.65, "Signal Artifacts & Dropouts", fontsize=8, fontweight='bold', color='#9F1239')
ax.text(0.60, 3.35, "• Motion artifacts in dry-EEG & PPG\n• 10%–50% packet missingness\n• Loose wristbands & detachment",
        fontsize=7, color='#881337', va='center')

draw_card(ax, 0.50, 1.70, 2.65, 1.05, COL_AMBER_LIGHT, border_color='#FDE68A')
ax.text(0.60, 2.50, "Edge Gateway Bottlenecks", fontsize=8, fontweight='bold', color='#92400E')
ax.text(0.60, 2.10, "• Microcontroller SRAM < 5 kB\n• Quadratic Transformer OOM\n• Battery & thermal budgets",
        fontsize=7, color='#78350F', va='center')

draw_card(ax, 0.50, 0.55, 2.65, 0.95, '#F1F5F9', border_color=COL_BORDER)
ax.text(0.60, 1.25, "Architectural Gap", fontsize=8, fontweight='bold', color='#334155')
ax.text(0.60, 0.95, "• Early fusion: Modality collapse\n• Late fusion: Discards cross-talk\n• Need: Linear baseline fallback",
        fontsize=7, color='#475569', va='center')

draw_arrow(ax, 3.30, 2.37, 3.45, 2.37, color=COL_BLUE, width=2.0)

# ==============================================================================
# PANEL 2: WIDE-AND-DEEP TOKEN ARCHITECTURE
# ==============================================================================
draw_card(ax, 3.45, 0.35, 3.65, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(5.27, 4.15, "2. BIOSYNC WIDE-AND-DEEP FUSION", ha='center', va='center',
        fontsize=9, fontweight='bold', color=COL_BLUE)

# Input Modality Tokens (d=8)
tokens = [("PPG/HRV", 3.25), ("Dry EEG", 2.65), ("Actigraphy", 2.05), ("Vocal Feat.", 1.45)]
for label, y_pos in tokens:
    draw_card(ax, 3.60, y_pos, 1.15, 0.42, COL_BLUE_LIGHT, border_color='#BFDBFE')
    ax.text(4.17, y_pos + 0.21, label, ha='center', va='center', fontsize=6.8, fontweight='bold', color='#1E40AF')
    draw_arrow(ax, 4.75, y_pos + 0.21, 5.05, y_pos + 0.21, color='#94A3B8', width=1.0)

ax.text(4.17, 1.05, "Modality Tokens\n(Shared dim d=8)", ha='center', va='center',
        fontsize=7, color='#64748B', style='italic')

# Linear & Attention Split Blocks
draw_card(ax, 5.05, 2.50, 1.85, 1.35, '#FEF3C7', border_color='#FCD34D')
ax.text(5.97, 3.60, "Deep Attention Path", ha='center', fontsize=7.5, fontweight='bold', color='#B45309')
ax.text(5.97, 3.20, "• [CLS] Token (e_cls)\n• Multi-Head Self-Attn\n• Dynamic weights α_m",
        ha='center', fontsize=6.5, color='#78350F')

draw_card(ax, 5.05, 0.95, 1.85, 1.25, '#ECFDF5', border_color='#A7F3D0')
ax.text(5.97, 1.95, "Wide Linear Bypass", ha='center', fontsize=7.5, fontweight='bold', color='#047857')
ax.text(5.97, 1.55, "• Direct Concatenation\n• Retains Linear Subset\n• Baseline Fallback",
        ha='center', fontsize=6.5, color='#065F46')

# Fusion Concatenator
draw_arrow(ax, 6.00, 2.50, 6.00, 2.20, color='#94A3B8', width=1.0)
draw_card(ax, 5.15, 0.50, 1.65, 0.35, '#1E293B', border_color='#0F172A')
ax.text(5.97, 0.67, "Unified Classifier (y_hat)", ha='center', va='center',
        fontsize=7, fontweight='bold', color='#FFFFFF')

draw_arrow(ax, 7.10, 2.37, 7.25, 2.37, color=COL_EMERALD, width=2.0)

# ==============================================================================
# PANEL 3: HETEROGENEOUS DATASETS & AI-READI SCHEMA
# ==============================================================================
draw_card(ax, 7.25, 0.35, 2.75, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(8.62, 4.15, "3. MULTI-COHORT SCHEMAS", ha='center', va='center',
        fontsize=9, fontweight='bold', color=COL_EMERALD)

# Cognitive Cohort Card
draw_card(ax, 7.40, 2.35, 2.45, 1.55, '#F8FAFC', border_color='#E2E8F0')
ax.text(7.50, 3.65, "Cognitive-Decline Cohort", fontsize=7.8, fontweight='bold', color='#0F172A')
ax.text(7.50, 3.45, "Controlled Sensor Benchmark", fontsize=6.8, color='#64748B')
ax.text(7.50, 3.00, "• M = 4 tokens (22 features)\n• N = 360 primary cohort\n• Synthetic N = 1,440 scale",
        fontsize=6.8, color='#334155')

# AI-READI Schema Card
draw_card(ax, 7.40, 0.55, 2.45, 1.65, COL_EMERALD_LIGHT, border_color='#6EE7B7')
ax.text(7.50, 1.95, "NIH AI-READI Schema", fontsize=7.8, fontweight='bold', color='#065F46')
ax.text(7.50, 1.75, "Garmin Vivosmart 5 Telemetry", fontsize=6.8, color='#047857')
ax.text(7.50, 1.15, "• Resting Heart Rate\n• Respiration Rate\n• Device Stress Score\n• Step Count & Sleep Stages",
        fontsize=6.6, color='#064E3B')

draw_arrow(ax, 10.00, 2.37, 10.15, 2.37, color=COL_BLUE, width=2.0)

# ==============================================================================
# PANEL 4: CONCRETE RESULTS & HARDWARE PROOF
# ==============================================================================
draw_card(ax, 10.15, 0.35, 2.78, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(11.54, 4.15, "4. QUANTIFIED PERFORMANCE", ha='center', va='center',
        fontsize=9, fontweight='bold', color=COL_HEADER)

# Metric Result 1: Primary AUC
draw_card(ax, 10.30, 2.85, 2.48, 1.05, COL_BLUE_LIGHT, border_color='#BFDBFE')
ax.text(10.40, 3.65, "Cognitive AUC (N=360)", fontsize=7.8, fontweight='bold', color='#1E40AF')
ax.text(10.40, 3.35, "• BioSync: 0.928 ± 0.006\n• Concatenation: 0.926\n• Pure Attention: 0.911",
        fontsize=7, color='#1E3A8A')

# Metric Result 2: Interaction Synergy
draw_card(ax, 10.30, 1.60, 2.48, 1.10, '#ECFDF5', border_color='#A7F3D0')
ax.text(10.40, 2.45, "Synergy Scaling (N=1,440)", fontsize=7.8, fontweight='bold', color='#047857')
ax.text(10.40, 2.10, "• Unlocks Interaction Features\n• BioSync: 0.945 AUC\n• Concat: 0.892 (Delta +0.053)",
        fontsize=7, color='#065F46')

# Metric Result 3: Embedded Footprint
draw_card(ax, 10.30, 0.50, 2.48, 0.95, '#FFFBEB', border_color='#FDE68A')
ax.text(10.40, 1.20, "MCU SRAM Footprint", fontsize=7.8, fontweight='bold', color='#92400E')
ax.text(10.40, 0.88, "• ~1,000 Total Parameters\n• 3.4 kB – 4.5 kB Memory\n• Operates on Cortex-M4/ESP32",
        fontsize=7, color='#78350F')

# Export high-res outputs
out_png = os.path.join(FIG_DIR, 'graphical_abstract.png')
out_pdf = os.path.join(FIG_DIR, 'graphical_abstract.pdf')

plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()

print(f"Graphical Abstract generated successfully:")
print(f"PNG: {out_png}")
print(f"PDF: {out_pdf}")
