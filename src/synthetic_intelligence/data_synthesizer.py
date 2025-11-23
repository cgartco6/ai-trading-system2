import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

class MarketDataSynthesizer:
    """Synthetic market data generator using GANs"""
    
    def __init__(self, seq_len: int = 60, feature_dim: int = 5):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.generator = None
        self.discriminator = None
        
    def build_models(self):
        """Build generator and discriminator models"""
        self.generator = Generator(self.seq_len, self.feature_dim)
        self.discriminator = Discriminator(self.seq_len, self.feature_dim)
        
    def fit(self, real_data: np.ndarray, epochs: int = 1000, batch_size: int = 32):
        """Train the GAN on real market data"""
        real_data = self._preprocess_data(real_data)
        
        # Training loop
        for epoch in range(epochs):
            for _ in range(len(real_data) // batch_size):
                # Train discriminator
                real_batch = self._get_real_batch(real_data, batch_size)
                fake_batch = self.generator.generate(batch_size)
                
                d_loss_real = self.discriminator(real_batch)
                d_loss_fake = self.discriminator(fake_batch)
                
                d_loss = -torch.mean(torch.log(d_loss_real) + torch.log(1 - d_loss_fake))
                
                # Train generator
                fake_batch = self.generator.generate(batch_size)
                g_loss = -torch.mean(torch.log(self.discriminator(fake_batch)))
                
                # Update models (simplified)
                self._update_models(d_loss, g_loss)
                
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")
    
    def generate_synthetic_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic market data"""
        with torch.no_grad():
            synthetic_data = self.generator.generate(n_samples)
        return synthetic_data.cpu().numpy()
    
    def generate_stress_scenarios(self, 
                                 base_data: pd.DataFrame,
                                 scenario_type: str = "crash") -> pd.DataFrame:
        """Generate market stress scenarios"""
        if scenario_type == "crash":
            return self._simulate_market_crash(base_data)
        elif scenario_type == "volatility":
            return self._simulate_high_volatility(base_data)
        elif scenario_type == "flash_crash":
            return self._simulate_flash_crash(base_data)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def _simulate_market_crash(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simulate market crash scenario"""
        synthetic = data.copy()
        crash_start = len(synthetic) // 2
        
        # Apply crash pattern
        for i in range(crash_start, len(synthetic)):
            progress = (i - crash_start) / (len(synthetic) - crash_start)
            crash_factor = 1 - 0.4 * progress  # 40% drawdown
            synthetic.iloc[i] = synthetic.iloc[i] * crash_factor
            
        return synthetic
    
    def _simulate_high_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simulate high volatility regime"""
        synthetic = data.copy()
        returns = synthetic.pct_change()
        
        # Increase volatility
        high_vol_returns = returns * 3 + np.random.normal(0, 0.01, len(returns))
        
        # Reconstruct price series
        synthetic = synthetic.iloc[0] * (1 + high_vol_returns).cumprod()
        return synthetic

class Generator(nn.Module):
    """Generator network for synthetic data"""
    
    def __init__(self, seq_len: int, feature_dim: int, latent_dim: int = 100):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * feature_dim),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        output = self.net(z)
        return output.view(batch_size, self.seq_len, self.feature_dim)
    
    def generate(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim)
        return self.forward(z)

class Discriminator(nn.Module):
    """Discriminator network for synthetic data"""
    
    def __init__(self, seq_len: int, feature_dim: int):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        self.net = nn.Sequential(
            nn.Linear(seq_len * feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.net(x)
