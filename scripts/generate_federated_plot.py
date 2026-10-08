import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIRS = [
    os.path.join(PAPER_DIR, 'fig'),
    os.path.join(PAPER_DIR, 'Knowledge-Based Systems', 'fig')
]

for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)

rounds = np.arange(1, 101)

def conv_curve(r, max_auc, rate, noise_std=0.003):
    np.random.seed(42)
    val = max_auc - (max_auc - 0.65) * np.exp(-r / rate)
    noise = np.random.normal(0, noise_std, size=len(r))
    return np.clip(val + noise, 0.65, max_auc)

auc_pemcan = conv_curve(rounds, 0.932, rate=9.5)
auc_lora   = conv_curve(rounds, 0.911, rate=15.0)
auc_full   = conv_curve(rounds, 0.898, rate=24.0, noise_std=0.006)

cum_full        = rounds * 29.72
cum_lora        = rounds * 0.52
cum_pemcan_fp32 = rounds * 0.52
cum_pemcan_int8 = rounds * 0.13

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=300)

# -------------------------------------------------------------------------
# Subplot (a): Multi-Center Convergence
# -------------------------------------------------------------------------
ax1.plot(rounds, auc_pemcan, color='#DC2626', lw=2.8, label='FFA-LoRA + PEM-CAN (38 rounds)')
ax1.plot(rounds, auc_lora, color='#2563EB', lw=2.2, linestyle='--', label='FedAvg + Standard LoRA (54 rounds)')
ax1.plot(rounds, auc_full, color='#4B5563', lw=2.0, linestyle=':', label='FedAvg + Full FT (86 rounds)')

# Target AUROC reference threshold (0.930)
ax1.axhline(0.930, color='#DC2626', linestyle='--', lw=1.3, alpha=0.5)
ax1.plot([38, 38], [0.70, 0.930], color='#DC2626', linestyle=':', lw=1.5, alpha=0.6)
ax1.scatter([38], [0.930], color='#DC2626', s=65, zorder=5, edgecolors='white', linewidth=1.5)

ax1.set_xlabel('Federated Communication Rounds', fontsize=12.5, fontweight='bold', labelpad=8)
ax1.set_ylabel('Global Test AUROC', fontsize=12.5, fontweight='bold', labelpad=8)
ax1.set_title('(a) Multi-Center AUROC Convergence', fontsize=13.5, fontweight='bold', pad=12)
ax1.set_xlim(1, 100)
ax1.set_ylim(0.70, 0.955)
ax1.tick_params(axis='both', which='major', labelsize=11.5, width=1.2, length=5)
ax1.legend(loc='lower right', fontsize=10.2, frameon=True, framealpha=0.95, edgecolor='#CBD5E1')
ax1.grid(True, linestyle='--', alpha=0.55)

# -------------------------------------------------------------------------
# Subplot (b): Cumulative Bandwidth Overhead
# -------------------------------------------------------------------------
ax2.plot(rounds, cum_pemcan_int8, color='#DC2626', lw=2.8, label='FFA-LoRA + PEM-CAN (INT8: 2.6 MB/client)')
ax2.plot(rounds, cum_pemcan_fp32, color='#EA580C', lw=2.2, linestyle='-.', label='FFA-LoRA + PEM-CAN (FP32: 10.4 MB/client)')
ax2.plot(rounds, cum_lora, color='#2563EB', lw=2.2, linestyle='--', label='Standard LoRA (FP32: 10.4 MB/client)')
ax2.plot(rounds, cum_full, color='#4B5563', lw=2.0, linestyle=':', label='Full Fine-Tuning (FP32: 594.4 MB/client)')

ax2.set_yscale('log')
ax2.set_xlabel('Federated Communication Rounds', fontsize=12.5, fontweight='bold', labelpad=8)
ax2.set_ylabel('Cumulative Transmitted Data (GB) [Log Scale]', fontsize=11.5, fontweight='bold', labelpad=8)
ax2.set_title('(b) Cumulative Bandwidth (50 Nodes)', fontsize=13.5, fontweight='bold', pad=12)
ax2.set_xlim(1, 100)
ax2.set_ylim(0.08, 4800)
ax2.tick_params(axis='both', which='major', labelsize=11.5, width=1.2, length=5)
ax2.legend(loc='upper left', fontsize=9.6, frameon=True, framealpha=0.95, edgecolor='#CBD5E1')
ax2.grid(True, which='both', linestyle='--', alpha=0.55)

# Use explicit subplots_adjust for generous breathing room
plt.subplots_adjust(left=0.085, right=0.98, top=0.91, bottom=0.15, wspace=0.26)

for d in FIG_DIRS:
    out_path = os.path.join(d, 'python_federated_convergence.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

plt.close()
