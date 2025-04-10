import torch.nn as nn
from torchvision import models

# Define EfficientNet-B0 Model
class EfficientNetLSTM(nn.Module):
    def __init__(self):
        super(EfficientNetLSTM, self).__init__()
        self.efficientnet = models.efficientnet_b4(pretrained=True)
        self.efficientnet.classifier = nn.Identity()
        self.lstm = nn.LSTM(input_size=1792, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, 2)
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.efficientnet(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x