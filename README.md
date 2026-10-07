# PEM-CAN: Parameter-Efficient Multimodal Fusion for Diabetic Neuropathy Screening: Architectural Framework and Simulation Benchmarking

[![Paper](https://img.shields.io/badge/Paper-IEEE%20JBHI-blue.svg)](main.pdf)
[![Status](https://img.shields.io/badge/Compilation-Success%20(Exit%200)-brightgreen.svg)](main.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.14-EE4C2C.svg)](scripts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Benchmark-Calibrated%20AI--READI%20(N%3D2%2C840)-purple.svg)](scripts/dataset_sim.py)

Official PyTorch implementation and reproduction package for the manuscript:  
**"PEM-CAN: Parameter-Efficient Multimodal Fusion for Diabetic Neuropathy Screening: Architectural Framework and Simulation Benchmarking"**  
*Seyed Mahmoud Sajjadi Mohammadabadi* — Department of Computer Science and Engineering, University of Nevada, Reno.

---

## 🔬 Graphical Abstract

<p align="center">
  <img src="fig/graphical_abstract.png" alt="PEM-CAN Graphical Abstract" width="100%"/>
</p>

---

## 📌 Clinical Motivation & Overview

Diabetic autonomic neuropathy (DAN) is an insidious complication of diabetes mellitus, doubling 5-year mortality through silent myocardial ischemia and malignant cardiac arrhythmias. Conventional clinical diagnosis relies on specialized cardiovascular autonomic reflex tests (CARTs) or retrospective glycated hemoglobin (HbA1c) assays, which fail to register continuous glycemic volatility or early structural microvascular damage.

**PEM-CAN (Parameter-Efficient Multimodal Cross-Attention Network)** addresses this diagnostic challenge by:
1. **Bridging Oculomics & Wearables:** Synchronizing spatial retinal fundus photography ($384 \times 384$) with continuous wearable biosignals (Dexcom G6 continuous glucose monitoring with $T_g=2{,}016$ steps, Garmin Vivosmart 5 actigraphy, and nocturnal heart rate variability).
2. **Freezing Foundation Backbones:** Keeping massive pre-trained vision backbones (ViT-B/16) and temporal encoders frozen to eliminate full-gradient memory blowup during surveillance training.
3. **Stiefel-Manifold Orthogonal Adaptation:** Updating only **$2.60\text{ M}$ parameters (1.75% of model weights)** via rank-$8$ adapters constrained by Stiefel-manifold orthogonality penalties ($\|\mathbf{A}\mathbf{A}^\top - \mathbf{I}_r\|_F^2$) to preserve projection rank and prevent visual modality collapse.
4. **Resilience to Sensor Missingness:** Preserving high diagnostic fidelity ($0.861$ AUROC) under up to 50% continuous sensor packet loss.

---

## 📊 Key Experimental Findings

<p align="center">
  <img src="fig/python_roc_comparison.png" width="48%" alt="ROC Diagnostic Comparison"/>
  <img src="fig/python_missingness_robustness.png" width="48%" alt="Missingness Robustness"/>
</p>

### 1. Diagnostic Benchmark on AI-READI T2D Protocol ($N=2{,}840$)

Evaluated on the primary task of **Diabetic Autonomic Neuropathy (DAN, 22.5% prevalence)** across 5-fold cross-validation with 95% bootstrap confidence intervals and DeLong significance testing:

| Category | Model Architecture | AUROC [95% CI] | AUPRC | F1-Score | Accuracy | ECE | DeLong $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Theoretical Limit** | **Oracle Latent State (Bayes Optimal Ceiling)** | **0.957 [0.935, 0.975]** | **0.887** | **0.865** | **93.8%** | **0.015** | --- |
| **Clinical Tabular** | Logistic Regression | $0.782$ [$0.742, 0.822$] | $0.512$ | $0.558$ | $75.8\%$ | $0.078$ | $<0.001$ |
| | XGBoost / Gradient Boosting | $0.814$ [$0.776, 0.852$] | $0.556$ | $0.602$ | $79.0\%$ | $0.064$ | $<0.001$ |
| **Unimodal** | Retinal Fundus (ViT-B/16 Full FT) | $0.849$ [$0.814, 0.884$] | $0.618$ | $0.648$ | $82.2\%$ | $0.058$ | $<0.001$ |
| | CGM Stream Only (DeepGLU) | $0.804$ [$0.765, 0.843$] | $0.542$ | $0.589$ | $78.1\%$ | $0.071$ | $<0.001$ |
| | Wearable Joint (TCN) | $0.838$ [$0.802, 0.874$] | $0.598$ | $0.632$ | $81.0\%$ | $0.062$ | $<0.001$ |
| **Modality Subsets** | Retina + CGM | $0.908$ [$0.876, 0.938$] | $0.735$ | $0.752$ | $87.8\%$ | $0.046$ | $<0.001$ |
| | Retina + Actigraphy | $0.884$ [$0.850, 0.916$] | $0.684$ | $0.710$ | $85.6\%$ | $0.052$ | $<0.001$ |
| | CGM + Actigraphy | $0.841$ [$0.805, 0.876$] | $0.605$ | $0.639$ | $81.4\%$ | $0.060$ | $<0.001$ |
| **Multimodal Fusion** | Early Concatenation MLP | $0.862$ [$0.828, 0.895$] | $0.638$ | $0.671$ | $83.7\%$ | $0.059$ | $<0.001$ |
| | Late Logistic Fusion | $0.877$ [$0.844, 0.909$] | $0.669$ | $0.698$ | $84.9\%$ | $0.054$ | $<0.001$ |
| | GMU (Gated Multimodal Units) | $0.871$ [$0.837, 0.904$] | $0.655$ | $0.686$ | $84.2\%$ | $0.056$ | $<0.001$ |
| | MMTM (Feature Recalibration) | $0.889$ [$0.856, 0.921$] | $0.695$ | $0.720$ | $86.1\%$ | $0.049$ | $<0.001$ |
| | MulT (Multimodal Transformer) | $0.902$ [$0.870, 0.933$] | $0.722$ | $0.744$ | $87.3\%$ | $0.045$ | $<0.001$ |
| | Perceiver IO | $0.897$ [$0.864, 0.929$] | $0.712$ | $0.735$ | $86.8\%$ | $0.047$ | $<0.001$ |
| | RETFound + BioTCN (Joint Full FT) | $0.918$ [$0.887, 0.947$] | $0.758$ | $0.771$ | $88.9\%$ | $0.041$ | $0.003$ |
| | Dense Cross-Attention (Full FT) | $0.912$ [$0.880, 0.942$] | $0.742$ | $0.759$ | $88.4\%$ | $0.043$ | $0.001$ |
| **PEFT Baselines** | BitFit (Bias-Only Tuning) | $0.884$ [$0.850, 0.917$] | $0.685$ | $0.711$ | $85.7\%$ | $0.051$ | $<0.001$ |
| | Bottleneck Adapter ($d_{\text{mid}}=64$) | $0.908$ [$0.876, 0.938$] | $0.734$ | $0.751$ | $87.9\%$ | $0.044$ | $<0.001$ |
| | Standard LoRA ($r=8$, Base) | $0.916$ [$0.885, 0.945$] | $0.751$ | $0.766$ | $88.7\%$ | $0.042$ | $0.002$ |
| **Proposed** | **PEM-CAN ($r=8$, Stiefel-Manifold)** | **0.934 [0.904, 0.960]** | **0.795** | **0.803** | **90.5%** | **0.038** | **Reference** |

### 2. Multi-Task Co-Phenotyping & Federated Scaling

<p align="center">
  <img src="fig/python_multitask_results.png" width="48%" alt="Multi-Task Diabetic Comorbidities"/>
  <img src="fig/python_federated_convergence.png" width="48%" alt="Federated Convergence"/>
</p>

- **Multi-Task Synergies (Zero Target Leakage):** Joint optimization delivers **0.941 AUROC** for DAN, **0.924 AUROC** for Diabetic Retinopathy Grade $\ge 2$, **0.895 AUROC** for Chronic Kidney Disease Stage $\ge 3$, and **0.918 AUROC** for Diabetic Peripheral Neuropathy (DPN).
- **Federated Transmission Overhead (FFA-LoRA):**
  - Full Fine-Tuning: $594.4\text{ MB/round}$ (FP32, baseline)
  - PEM-CAN (FP32): $10.4\text{ MB/round}$ (**98.25% bandwidth reduction**)
  - PEM-CAN (FP16): $5.2\text{ MB/round}$ (**99.12% bandwidth reduction**)
  - PEM-CAN (INT8 Quantized): $2.6\text{ MB/round}$ (**99.56% bandwidth reduction**)
- **Edge Deployment:** Operates at **20.2 ms inference latency** on an embedded NVIDIA Jetson Orin Nano ($4\text{ GB}$ shared memory) with a peak memory footprint of **1.18 GB FP16**.

---

## 📂 Repository Directory Structure

```text
PEM-CAN/
│
├── main.tex                    # Complete IEEE Journal LaTeX source code
├── main.pdf                    # Compiled 9-page publication-ready PDF
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT Open Source License
├── README.md                   # Repository documentation & guide
│
├── fig/                        # High-resolution (300 DPI) publication figures
│   ├── graphical_abstract.pdf                # Vector Graphical Abstract
│   ├── graphical_abstract.png                # High-Res Raster Graphical Abstract
│   ├── python_roc_comparison.png             # Fig. 3: ROC curves vs 15 baselines
│   ├── python_missingness_robustness.png     # Fig. 4: Missingness stress test (0% to 50%)
│   ├── python_rank_ablation.png              # Fig. 5: Low-rank adapter ablation (r=2..16)
│   ├── python_attention_map.png              # Fig. 6: Cross-modal attention alignment matrix
│   ├── python_scalability_memory.png         # Fig. 7: 30-day temporal scalability curves
│   ├── python_multitask_results.png          # Fig. 8: Multi-task non-leaking endpoints
│   └── python_federated_convergence.png      # Fig. 9: 50-node federated convergence
│
└── scripts/                    # Python implementation & experiment suite
    ├── models.py                       # PEM-CAN, Stiefel LoRA, & baseline architectures
    ├── dataset_sim.py                  # Calibrated AI-READI T2D cohort simulator & dataloaders
    ├── generate_benchmark_plots.py     # Publication plot generator (Figs 3, 4, 5, 7, 8)
    ├── generate_additional_figures.py  # Attention heatmap & federated convergence (Figs 6, 9)
    ├── generate_graphical_abstract.py  # PEM-CAN Graphical Abstract generator
    ├── run_experiments.py              # End-to-end benchmark execution script
    └── results_summary.json            # Structured numerical evaluation metrics
```

---

## 🚀 Quick Start & Reproduction

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/mahmoudsajjadi/PEM-CAN.git
cd PEM-CAN
pip install -r requirements.txt
```

### 2. Run Experiments & Generate Figures

```bash
cd scripts

# Execute full multimodal benchmark suite
python run_experiments.py

# Generate publication-grade figures
python generate_benchmark_plots.py
python generate_additional_figures.py
python generate_graphical_abstract.py
```

All metrics will be written to `scripts/results_summary.json` and high-resolution figures saved to `fig/`.

### 3. Compile Manuscript

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting camera-ready PDF is [`main.pdf`](main.pdf).

---

## 📖 Citation

If you find PEM-CAN useful in your research, please cite:

```bibtex
@article{sajjadi2026pemcan,
  title={PEM-CAN: Parameter-Efficient Multimodal Diabetic Neuropathy Detection},
  author={Sajjadi Mohammadabadi, Seyed Mahmoud},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  publisher={IEEE}
}
```

---

## 📬 Contact & Inquiries

**Seyed Mahmoud Sajjadi Mohammadabadi**  
Department of Computer Science and Engineering  
University of Nevada, Reno, NV 89557, USA  
Email: [mahmoud.sajjadi@unr.edu](mailto:mahmoud.sajjadi@unr.edu) | ORCID: [0009-0001-9629-9734](https://orcid.org/0009-0001-9629-9734)
