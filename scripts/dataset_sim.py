"""
Simulated AI-READI Multimodal Clinical Dataset Generator.
Provides paired retinal fundus embeddings, 7-day CGM, and actigraphy with multi-task labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AIREADIDataset(Dataset):
    def __init__(self, n_samples: int = 1000, n_patches: int = 16, embed_dim: int = 768,
                 t_cgm: int = 2016, t_act: int = 2016, missing_rate: float = 0.0, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.n_samples = n_samples
        self.missing_rate = missing_rate

        # 1. Latent Disease Risk Factors (underlying diabetes severity: 0 to 1)
        self.severity = np.random.beta(a=2, b=3, size=n_samples).astype(np.float32)

        # 2. Retinal visual features: correlated with severity
        self.x_v = np.random.randn(n_samples, n_patches, embed_dim).astype(np.float32)
        for i in range(n_samples):
            self.x_v[i, :4, :] += float(self.severity[i]) * 1.5

        # 3. Continuous Glucose Monitoring (CGM): 7 days (5 min uniform intervals = 2016)
        time_steps = np.linspace(0, 7 * 2 * np.pi, t_cgm, dtype=np.float32)
        circadian = 10.0 * np.sin(time_steps)
        self.x_g = np.zeros((n_samples, 1, t_cgm), dtype=np.float32)
        for i in range(n_samples):
            base_glucose = 90.0 + float(self.severity[i]) * 90.0
            noise = np.random.normal(0, 10.0 + float(self.severity[i]) * 25.0, size=t_cgm).astype(np.float32)
            series = base_glucose + circadian + noise
            self.x_g[i, 0, :] = ((series - 120.0) / 40.0).astype(np.float32)

        # 4. Tri-axial actigraphy (x, y, z): physical exertion inverse to severity
        self.x_a = (np.random.randn(n_samples, 3, t_act).astype(np.float32) *
                    (1.2 - 0.4 * self.severity[:, None, None])).astype(np.float32)

        # Apply structured missingness to temporal streams if requested
        if missing_rate > 0.0:
            drop_len = int(t_cgm * missing_rate)
            for i in range(n_samples):
                start_idx = np.random.randint(0, max(1, t_cgm - drop_len))
                self.x_g[i, :, start_idx:start_idx + drop_len] = 0.0
                self.x_a[i, :, start_idx:start_idx + drop_len] = 0.0

        # 5. Multi-task diagnostic clinical targets:
        p_dan = 1.0 / (1.0 + np.exp(-(3.5 * self.severity - 1.4)))
        p_dr  = 1.0 / (1.0 + np.exp(-(3.8 * self.severity - 1.2)))
        p_ckd = 1.0 / (1.0 + np.exp(-(3.0 * self.severity - 1.5)))
        p_agv = 1.0 / (1.0 + np.exp(-(4.2 * self.severity - 1.6)))

        y_dan = (np.random.rand(n_samples) < p_dan).astype(np.float32)
        y_dr  = (np.random.rand(n_samples) < p_dr).astype(np.float32)
        y_ckd = (np.random.rand(n_samples) < p_ckd).astype(np.float32)
        y_agv = (np.random.rand(n_samples) < p_agv).astype(np.float32)

        self.targets = np.stack([y_dan, y_dr, y_ckd, y_agv], axis=-1).astype(np.float32)

        # Demographics (0: Caucasian, 1: African American, 2: Hispanic)
        self.demographics = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'x_v': torch.tensor(self.x_v[idx], dtype=torch.float32),
            'x_g': torch.tensor(self.x_g[idx], dtype=torch.float32),
            'x_a': torch.tensor(self.x_a[idx], dtype=torch.float32),
            'y': torch.tensor(self.targets[idx], dtype=torch.float32),
            'demo': int(self.demographics[idx])
        }

def get_dataloaders(n_samples: int = 1000, batch_size: int = 32, missing_rate: float = 0.0):
    dataset = AIREADIDataset(n_samples=n_samples, missing_rate=missing_rate)
    train_size = int(0.7 * n_samples)
    val_size = int(0.1 * n_samples)
    test_size = n_samples - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
