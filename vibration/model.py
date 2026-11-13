# CH3 단일 채널 스펙트럼을 압축하고 복원하는 1차원 합성곱 오토엔코더입니다.

from torch import nn
from vibration.channel import INPUT_SHAPE


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 스펙트럼 길이와 채널 수를 줄여 정상 패턴을 압축합니다.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )

        # 압축된 특징에서 원래 CH3 단일 채널 스펙트럼을 복원합니다.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected CH3 model input (N, 1, 3201), got {tuple(x.shape)}")
        reconstructed = self.decoder(self.encoder(x))

        # 복원 과정에서 늘어난 끝부분을 잘라 입력 길이에 맞춥니다.
        return reconstructed[:, :, : x.shape[-1]]
