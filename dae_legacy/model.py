import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 800개 값을 50개 길이까지 압축
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(64, 2, kernel_size=3, stride=2, padding=1)
        )
        # Decoder: 압축한 값을 다시 800개로 복원
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                2, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16, 1, kernel_size=4, stride=2, padding=1
            )
        )
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
if __name__ == "__main__":
    model = ConvAutoencoder()

    # 테스트용 가짜 데이터 1개
    x = torch.randn(1, 1, 800)

    z = model.encoder(x)
    y = model(x)

    print("입력 크기:", x.shape)
    print("압축된 데이터 크기:", z.shape)
    print("복원된 데이터 크기:", y.shape)
