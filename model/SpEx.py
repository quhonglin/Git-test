import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32000, 8000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(8000, 32000),
            nn.ReLU()
        )

    def forward(self, mixture):
        """
            mixture: [batch_size, 1, 32000]
        """

        enhanced = self.fc1(mixture)
        enhanced = self.fc2(enhanced)

        return enhanced


def test_32000():
    model = Model()
    mix = torch.rand(2, 1, 32000)
    print(model(mix).size())


if __name__ == '__main__':
    test_32000()
