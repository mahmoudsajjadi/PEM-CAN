"""
Standalone publication-grade script to generate PEM-CAN Graphical Abstract.
Dimensions: 1328 x 531 aspect ratio, 300 DPI, modern vector aesthetics.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'fig')
os.makedirs(FIG_DIR, exist_ok=True)

FIG_WIDTH = 13.28
FIG_HEIGHT = 5.31

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13.28)
ax.set_ylim(0, 5.31)
ax.axis('off')

# Background
ax.fill_between([0, 13.28], 0, 5.31, color='#F8FAFC', zorder=0)

# Color Palette
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

def draw_card(ax, x, y, w, h, bg_color, border_color='#CBD5E1', radius=0.15, zorder=1):
    card = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=bg_color, edgecolor=border_color,
                          linewidth=1.2, zorder=zorder)
    ax.add_patch(card)
    return card

def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', width=1.8):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>,head_length=5,head_width=3',
                            color=color, linewidth=width, zorder=5)
    ax.add_patch(arrow)

# ==============================================================================
# HEADER STRIP
# ==============================================================================
ax.text(6.64, 4.95, "PEM-CAN: Parameter-Efficient Multimodal Diabetic Neuropathy Detection",
        ha='center', va='center', fontsize=13.5, fontweight='bold', color=COL_HEADER)
ax.text(6.64, 4.65, "Orthogonal Low-Rank Subspace Cross-Attention over Retinal Fundus & Continuous Wearable Biosignals",
        ha='center', va='center', fontsize=9.2, color=COL_SUB, style='italic')

# ==============================================================================
# PANEL 1: CLINICAL CRISIS & OBSERVATION WINDOWS
# ==============================================================================
draw_card(ax, 0.35, 0.35, 2.95, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(1.82, 4.15, "1. CLINICAL CRISIS & MODALITIES", ha='center', va='center',
        fontsize=8.8, fontweight='bold', color=COL_ROSE)

draw_card(ax, 0.50, 2.95, 2.65, 0.95, COL_ROSE_LIGHT, border_color='#FECDD3')
ax.text(0.60, 3.65, "The Clinical Blindspot (DAN)", fontsize=8, fontweight='bold', color='#9F1239')
ax.text(0.60, 3.32, "• 2x 5-yr mortality (silent ischemia)\n• HbA1c is blind to glycemic volatility\n• CARTs reflex labs lack outpatient scale",
        fontsize=6.8, color='#881337', va='center')

draw_card(ax, 0.50, 1.70, 2.65, 1.05, COL_AMBER_LIGHT, border_color='#FDE68A')
ax.text(0.60, 2.50, "Ocular & Wearable Windows", fontsize=8, fontweight='bold', color='#92400E')
ax.text(0.60, 2.10, "• Retina: Microvascular remodeling\n• CGM: 5-min interstitial glucose (T=2016)\n• Actigraphy: HRV, stress, rest dynamics",
        fontsize=6.8, color='#78350F', va='center')

draw_card(ax, 0.50, 0.55, 2.65, 0.95, '#F1F5F9', border_color=COL_BORDER)
ax.text(0.60, 1.25, "Computational Hurdles", fontsize=8, fontweight='bold', color='#334155')
ax.text(0.60, 0.95, "• Visual dominance (modality collapse)\n• 10%–50% sensor packet dropouts\n• Quadratic memory blowup in transformers",
        fontsize=6.8, color='#475569', va='center')

draw_arrow(ax, 3.30, 2.37, 3.45, 2.37, color=COL_BLUE, width=2.0)

# ==============================================================================
# PANEL 2: PEM-CAN ARCHITECTURE
# ==============================================================================
draw_card(ax, 3.45, 0.35, 3.65, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(5.27, 4.15, "2. PEM-CAN SUBSPACE ADAPTERS", ha='center', va='center',
        fontsize=8.8, fontweight='bold', color=COL_BLUE)

draw_card(ax, 3.60, 3.10, 1.45, 0.70, COL_BLUE_LIGHT, border_color='#BFDBFE')
ax.text(4.32, 3.55, "Frozen ViT-B/16", fontsize=7.2, fontweight='bold', color='#1E40AF', ha='center')
ax.text(4.32, 3.30, "M=576 tokens", fontsize=6.5, color='#1E3A8A', ha='center')

draw_card(ax, 3.60, 2.15, 1.45, 0.75, COL_EMERALD_LIGHT, border_color='#A7F3D0')
ax.text(4.32, 2.62, "Dilated Conv1D", fontsize=7.2, fontweight='bold', color='#065F46', ha='center')
ax.text(4.32, 2.35, "CGM + Actigraphy (L=252)", fontsize=6.3, color='#047857', ha='center')

draw_card(ax, 3.60, 0.75, 1.45, 1.20, '#FEF3C7', border_color='#FCD34D')
ax.text(4.32, 1.70, "Stiefel LoRA", fontsize=7.5, fontweight='bold', color='#92400E', ha='center')
ax.text(4.32, 1.35, "||AAᵀ - I_r||_F²\nRank r = 8\nFreeze W_0", fontsize=6.5, color='#78350F', ha='center')

# Fusion Block
draw_card(ax, 5.25, 1.80, 1.65, 1.95, '#EEF2FF', border_color='#C7D2FE')
ax.text(6.07, 3.45, "Bidirectional Cross-Attn", fontsize=7.5, fontweight='bold', color='#3730A3', ha='center')
ax.text(6.07, 3.15, "S(Z_v, Z_s) & S(Z_s, Z_v)", fontsize=6.5, color='#4338CA', ha='center')
ax.text(6.07, 2.70, "• O(M·L) Linear in L\n• Anti-Dominance\n• Folds into W_0 at Test", fontsize=6.3, color='#312E81', ha='center')
ax.text(6.07, 2.05, "L_total = L_focal + L_orth + L_align", fontsize=6.0, color='#4F46E5', ha='center')

# Output Head
draw_card(ax, 5.25, 0.65, 1.65, 0.95, '#1E293B', border_color='#0F172A')
ax.text(6.07, 1.35, "Multi-Task Head", fontsize=7.2, fontweight='bold', color='#FFFFFF', ha='center')
ax.text(6.07, 1.00, "DAN (Autonomic) | DR (Retina)\nCKD (Renal) | DPN (Peripheral)", fontsize=5.8, color='#94A3B8', ha='center')

draw_arrow(ax, 5.05, 3.45, 5.25, 3.10, color=COL_BLUE, width=1.2)
draw_arrow(ax, 5.05, 2.50, 5.25, 2.70, color=COL_EMERALD, width=1.2)
draw_arrow(ax, 6.07, 1.80, 6.07, 1.60, color='#0F172A', width=1.2)

draw_arrow(ax, 7.10, 2.37, 7.25, 2.37, color=COL_EMERALD, width=2.0)

# ==============================================================================
# PANEL 3: CALIBRATED BENCHMARK PROTOCOL
# ==============================================================================
draw_card(ax, 7.25, 0.35, 2.75, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(8.62, 4.15, "3. BENCHMARK PROTOCOL", ha='center', va='center',
        fontsize=8.8, fontweight='bold', color=COL_EMERALD)

draw_card(ax, 7.40, 2.35, 2.45, 1.55, '#F8FAFC', border_color='#E2E8F0')
ax.text(7.50, 3.65, "AI-READI T2D Protocol", fontsize=7.8, fontweight='bold', color='#0F172A')
ax.text(7.50, 3.40, "N = 2,840 Simulated Patients", fontsize=6.8, color='#64748B')
ax.text(7.50, 2.95, "• Statistically calibrated to NIH atlas\n• All diagnosed Type 2 Diabetes\n• Stratified 5-Fold Cross-Validation\n• African Amer., Hispanic, White cohorts",
        fontsize=6.5, color='#334155')

draw_card(ax, 7.40, 0.55, 2.45, 1.65, COL_EMERALD_LIGHT, border_color='#6EE7B7')
ax.text(7.50, 1.95, "Non-Leaking Endpoints", fontsize=7.8, fontweight='bold', color='#065F46')
ax.text(7.50, 1.70, "Adjudicated Clinical Labels", fontsize=6.8, color='#047857')
ax.text(7.50, 1.25, "• DAN (Autonomic): 22.5% Prev.\n• DR Grade >= 2: 18.2% Prev.\n• CKD Stage >= 3: 14.1% Prev.\n• DPN (Peripheral): 26.8% Prev.",
        fontsize=6.5, color='#064E3B')

draw_arrow(ax, 10.00, 2.37, 10.15, 2.37, color=COL_BLUE, width=2.0)

# ==============================================================================
# PANEL 4: QUANTIFIED PERFORMANCE & DEPLOYMENT
# ==============================================================================
draw_card(ax, 10.15, 0.35, 2.78, 4.05, '#FFFFFF', border_color=COL_BORDER)
ax.text(11.54, 4.15, "4. QUANTIFIED PERFORMANCE", ha='center', va='center',
        fontsize=8.8, fontweight='bold', color=COL_HEADER)

draw_card(ax, 10.30, 2.85, 2.48, 1.05, COL_BLUE_LIGHT, border_color='#BFDBFE')
ax.text(10.40, 3.65, "Diagnostic Discrimination", fontsize=7.8, fontweight='bold', color='#1E40AF')
ax.text(10.40, 3.32, "• AUROC: 0.934 ± 0.008 (p < 0.001)\n• F1-Score: 0.886 | ECE: 0.038\n• Sens @ 90% Spec: 86.4%",
        fontsize=6.8, color='#1E3A8A')

draw_card(ax, 10.30, 1.70, 2.48, 1.00, '#ECFDF5', border_color='#A7F3D0')
ax.text(10.40, 2.45, "Parameter & VRAM Economy", fontsize=7.8, fontweight='bold', color='#047857')
ax.text(10.40, 2.12, "• 2.6M params (1.75% of backbone)\n• 8.4 GB training VRAM (vs 32.4 GB)\n• 0.861 AUROC under 50% dropout",
        fontsize=6.8, color='#065F46')

draw_card(ax, 10.30, 0.55, 2.48, 1.00, '#FFFBEB', border_color='#FDE68A')
ax.text(10.40, 1.30, "Edge & Federated Scalability", fontsize=7.8, fontweight='bold', color='#92400E')
ax.text(10.40, 0.98, "• Jetson Orin Nano: 20.2 ms / 1.18 GB\n• 50-Node FL: 98.25% less bandwidth\n• 2.6 MB/round with INT8 quantization",
        fontsize=6.8, color='#78350F')

# Export high-res outputs
out_png = os.path.join(FIG_DIR, 'graphical_abstract.png')
out_pdf = os.path.join(FIG_DIR, 'graphical_abstract.pdf')

plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='#F8FAFC')
plt.close()

print(f"Graphical Abstract generated successfully:")
print(f"PNG: {out_png}")
print(f"PDF: {out_pdf}")
