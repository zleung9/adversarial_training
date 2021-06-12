from torch import nn
import torch

class Apdf2Lat(nn.Module):
    def __init__(self, output_size=3, dropout_rate=0.2):
        super(Apdf2Lat, self).__init__()
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(1, affine=False),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,1), stride=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(4, affine=False),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3,2), stride=(2,1)),
            nn.ReLU(),
            nn.BatchNorm2d(4, affine=False),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(4,2), stride=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(4, affine=False),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3,3), stride=(2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(8, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.main(x)
        # out = out.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # out = out.squeeze(dim=1)
        return out