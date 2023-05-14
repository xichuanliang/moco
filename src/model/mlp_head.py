from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size=512, projection_size=128):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# mlp_hidden_size=512 projection_size=128

