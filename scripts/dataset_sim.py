"""
Statistically Calibrated AI-READI Multimodal Clinical Dataset Simulator.
Generates paired retinal fundus embeddings, 7-day CGM, wearable actigraphy,
and clinical tabular biomarkers calibrated to the NIH Bridge2AI AI-READI T2D protocol.

Grounding:
A 4-dimensional latent physiological disease profile z ~ N(0, Sigma) governs:
  z_0 (z_v): Retinal microvascular damage / capillary rarefaction
  z_1 (z_g): Glycemic dysregulation, variability (MAGE), and sustained hyperglycemia
  z_2 (z_a): Cardiovascular autonomic neuropathy (vagal denervation, sympathovagal imbalance)
  z_3 (z_c): Circadian rest-activity rhythm disruption

Labels are multi-factorial combinations of these latent pathophysiological processes.
Oracle Bayes Optimal AUROC is mathematically derived and reported as the theoretical ceiling.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

class AIREADIDataset(Dataset):
    """
    Calibrated benchmark generator modeling the NIH Bridge2AI AI-READI study distribution.
    All simulated subjects represent diagnosed Type 2 Diabetes patients across varying severity strata.
    """
    def __init__(self, n_samples: int = 2840, n_patches: int = 16, embed_dim: int = 768,
                 t_cgm: int = 2016, t_act: int = 2016, missing_rate: float = 0.0, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.n_samples = n_samples
        self.missing_rate = missing_rate

        # 1. 4-dimensional latent physiological disease profile: z ~ N(0, Sigma)
        cov_latent = np.array([
            [1.00, 0.40, 0.35, 0.30],
            [0.40, 1.00, 0.45, 0.40],
            [0.35, 0.45, 1.00, 0.35],
            [0.30, 0.40, 0.35, 1.00]
        ], dtype=np.float32)
        self.z = np.random.multivariate_normal(np.zeros(4), cov_latent, size=n_samples).astype(np.float32)
        z_v = self.z[:, 0] # Retinal microvascular
        z_g = self.z[:, 1] # Glycemic
        z_a = self.z[:, 2] # Autonomic
        z_c = self.z[:, 3] # Circadian

        # 2. Demographic distributions matching AI-READI cohort
        # 0: Non-Hispanic White (60%), 1: Black/African American (25%), 2: Hispanic/Latino (15%)
        self.demographics = np.random.choice([0, 1, 2], size=n_samples, p=[0.60, 0.25, 0.15])
        
        # Clinical covariates
        self.age = (52.0 + 11.0 * np.random.randn(n_samples) + 2.0 * z_v).astype(np.float32)
        self.sex = np.random.binomial(1, 0.48, size=n_samples).astype(np.float32) # 0: Female, 1: Male
        self.diabetes_duration = (4.0 + 3.0 * z_g + np.random.exponential(4.0, size=n_samples)).astype(np.float32)
        self.hba1c = (6.8 + 0.9 * z_g + np.random.normal(0, 1.2, size=n_samples)).astype(np.float32)
        self.bmi = (28.0 + 1.2 * z_g + np.random.normal(0, 4.5, size=n_samples)).astype(np.float32)
        self.sbp = (125.0 + 5.0 * z_v + np.random.normal(0, 14.0, size=n_samples)).astype(np.float32)

        # 3. Retinal visual embeddings: spatial feature vectors (representing ViT-B/16 tokens)
        # Fundus pigmentation contrast attenuation factor by ethnicity (stress test for CLAHE)
        contrast_factor = np.ones(n_samples, dtype=np.float32)
        contrast_factor[self.demographics == 1] = 0.90 # Darker fundus background
        contrast_factor[self.demographics == 2] = 0.94

        self.x_v = np.random.randn(n_samples, n_patches, embed_dim).astype(np.float32)
        for i in range(n_samples):
            # Macular & temporal vascular arcade patches (0..7) reflect microvascular rarefaction
            self.x_v[i, :8, :] += float(z_v[i] * contrast_factor[i]) * 0.85

        # 4. Continuous Glucose Monitoring (CGM): 7 days at 5-min intervals (T = 2016)
        time_steps = np.linspace(0, 7 * 2 * np.pi, t_cgm, dtype=np.float32)
        self.x_g = np.zeros((n_samples, 1, t_cgm), dtype=np.float32)
        self.cgm_mean = np.zeros(n_samples, dtype=np.float32)
        self.cgm_cv = np.zeros(n_samples, dtype=np.float32)
        self.cgm_tir = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            base_glucose = 105.0 + float(z_g[i]) * 35.0
            circadian = 12.0 * np.sin(time_steps) * (1.0 + 0.3 * float(z_c[i]))
            # High-frequency glycemic volatility
            noise = np.random.normal(0, 14.0 + max(0.0, float(z_g[i])) * 18.0, size=t_cgm).astype(np.float32)
            series = base_glucose + circadian + noise
            self.x_g[i, 0, :] = ((series - 120.0) / 40.0).astype(np.float32)
            self.cgm_mean[i] = float(np.mean(series))
            self.cgm_cv[i] = float(np.std(series) / max(1.0, np.mean(series)) * 100.0)
            self.cgm_tir[i] = float(np.mean((series >= 70.0) & (series <= 180.0)) * 100.0)

        # 5. Tri-axial actigraphy & autonomic metrics (Garmin Vivosmart 5 schema)
        act_attenuation = np.clip(1.0 - 0.25 * z_c - 0.15 * z_a, 0.4, 1.6)[:, None, None]
        self.x_a = (np.random.randn(n_samples, 3, t_act).astype(np.float32) * act_attenuation).astype(np.float32)
        self.hrv_sdnn = (38.0 - 5.5 * z_a + np.random.normal(0, 11.0, size=n_samples)).astype(np.float32)
        self.hrv_rmssd = (28.0 - 4.5 * z_a + np.random.normal(0, 9.5, size=n_samples)).astype(np.float32)
        self.resting_hr = (74.0 + 4.0 * z_a + np.random.normal(0, 10.0, size=n_samples)).astype(np.float32)

        # Tabular clinical risk factor feature vector (10 standard clinical variables)
        self.tabular_features = np.stack([
            self.age, self.sex, self.diabetes_duration, self.hba1c, self.bmi, self.sbp,
            self.cgm_mean, self.cgm_cv, self.hrv_sdnn, self.hrv_rmssd
        ], axis=-1)

        # Apply structured missingness to temporal streams if requested
        if missing_rate > 0.0:
            drop_len = int(t_cgm * missing_rate)
            for i in range(n_samples):
                start_idx = np.random.randint(0, max(1, t_cgm - drop_len))
                self.x_g[i, :, start_idx:start_idx + drop_len] = 0.0
                self.x_a[i, :, start_idx:start_idx + drop_len] = 0.0

        # 6. Multi-task diagnostic clinical endpoints (calibrated to target prevalences):
        # Task 1: Diabetic Autonomic Neuropathy (DAN) - prevalence 22.5%
        self.logits_dan = -3.65 + 2.18 * z_a + 1.42 * z_g + 1.65 * z_v + 0.98 * z_c
        # Task 2: Diabetic Retinopathy (DR Grade >= 2) - prevalence 18.2%
        self.logits_dr  = -3.65 + 2.65 * z_v + 1.25 * z_g + 0.55 * z_a
        # Task 3: Chronic Kidney Disease (CKD Stage >= 3) - prevalence 14.1%
        self.logits_ckd = -4.18 + 1.65 * z_v + 1.85 * z_g + 1.15 * z_a
        # Task 4: Diabetic Peripheral Neuropathy (DPN) - prevalence 26.8%
        self.logits_dpn = -2.52 + 2.25 * z_a + 1.45 * z_g + 0.85 * z_v

        p_dan = 1.0 / (1.0 + np.exp(-self.logits_dan))
        p_dr  = 1.0 / (1.0 + np.exp(-self.logits_dr))
        p_ckd = 1.0 / (1.0 + np.exp(-self.logits_ckd))
        p_dpn = 1.0 / (1.0 + np.exp(-self.logits_dpn))

        y_dan = (np.random.rand(n_samples) < p_dan).astype(np.float32)
        y_dr  = (np.random.rand(n_samples) < p_dr).astype(np.float32)
        y_ckd = (np.random.rand(n_samples) < p_ckd).astype(np.float32)
        y_dpn = (np.random.rand(n_samples) < p_dpn).astype(np.float32)

        self.targets = np.stack([y_dan, y_dr, y_ckd, y_dpn], axis=-1).astype(np.float32)

        # Theoretical Bayes Optimal Oracle AUROC calculations
        self.oracle_auroc = {
            'DAN': float(roc_auc_score(y_dan, self.logits_dan)),
            'DR': float(roc_auc_score(y_dr, self.logits_dr)),
            'CKD': float(roc_auc_score(y_ckd, self.logits_ckd)),
            'DPN': float(roc_auc_score(y_dpn, self.logits_dpn))
        }
        self.oracle_auprc = {
            'DAN': float(average_precision_score(y_dan, self.logits_dan)),
            'DR': float(average_precision_score(y_dr, self.logits_dr)),
            'CKD': float(average_precision_score(y_ckd, self.logits_ckd)),
            'DPN': float(average_precision_score(y_dpn, self.logits_dpn))
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'x_v': torch.tensor(self.x_v[idx], dtype=torch.float32),
            'x_g': torch.tensor(self.x_g[idx], dtype=torch.float32),
            'x_a': torch.tensor(self.x_a[idx], dtype=torch.float32),
            'tabular': torch.tensor(self.tabular_features[idx], dtype=torch.float32),
            'y': torch.tensor(self.targets[idx], dtype=torch.float32),
            'demo': int(self.demographics[idx])
        }

def get_dataloaders(n_samples: int = 2840, batch_size: int = 32, missing_rate: float = 0.0, seed: int = 42):
    dataset = AIREADIDataset(n_samples=n_samples, missing_rate=missing_rate, seed=seed)
    train_size = int(0.70 * n_samples)
    val_size   = int(0.10 * n_samples)
    test_size  = n_samples - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
