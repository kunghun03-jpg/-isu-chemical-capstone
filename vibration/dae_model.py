"""Legacy denoising AE architecture; compatible with its raw state_dict files."""

from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 2, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (1, 800):
            raise ValueError("DAE 입력은 (N, 1, 800)이어야 합니다.")
        return self.decoder(self.encoder(x))
