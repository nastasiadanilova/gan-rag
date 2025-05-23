
import torch
import torch.nn as nn
import torch.optim as optim
import random


vocab = []
with open('/content/gan_vocab_extended.txt', 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f.readlines() if line.strip()]

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(vocab)

## 🧠 Глубокая архитектура GAN

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


gen = Generator()
disc = Discriminator()
loss_fn = nn.BCELoss()
opt_g = optim.Adam(gen.parameters(), lr=0.01)
opt_d = optim.Adam(disc.parameters(), lr=0.01)

real_data = [
    [0, 6, 12, 18],
    [1, 7, 13, 19],
    [2, 8, 14, 20],
    [3, 9, 15, 21],
    [4, 10, 16, 22],
    [5, 11, 17, 23]
]

for epoch in range(300):
    for real in real_data:
        real_tensor = torch.tensor(real, dtype=torch.float32)
        pred_real = disc(real_tensor)
        loss_d_real = loss_fn(pred_real, torch.ones(1))

        noise = torch.randn(16)
        fake = gen(noise)
        pred_fake = disc(fake.detach())
        loss_d_fake = loss_fn(pred_fake, torch.zeros(1))

        loss_d = loss_d_real + loss_d_fake
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        pred_fake = disc(fake)
        loss_g = loss_fn(pred_fake, torch.ones(1))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

with torch.no_grad():
    z = torch.randn(16)
    generated = gen(z).detach().numpy()
    tokens = [int(abs(i)) % VOCAB_SIZE for i in generated]
    print("\n✅ Сгенерированная логическая матрица:")
    for i, idx in enumerate(tokens):
        print(f"{i+1}. {idx2word[idx]}")