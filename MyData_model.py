import torch
import torch.nn as nn
from norse.torch.module.lif import LIFRecurrentCell
from norse.torch import LICell, LIFParameters
import torch.nn.functional as F

# ------------------ LeNet ------------------
# class MyData_LeNet(nn.Module):
#     def __init__(self, num_classes):
#         super(MyData_LeNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, 5, stride=2),  # assuming input: (1, 64, 64)
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 96, 3),
#             nn.ReLU(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(96 * 13 * 13, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

class MyData_LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=(3, 3)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))   # <--- Force final shape (96, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):          # x: (B, 200, 4004)
        x = x.unsqueeze(1)         # (B, 1, 200, 4004)
        x = self.encoder(x)        # (B, 96, 4, 4)
        x = torch.flatten(x, 1)    # (B, 1536)
        return self.fc(x)



# ------------------ Basic ResNet Blocks ------------------
class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x.clone()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.i_downsample:
            identity = self.i_downsample(identity)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x.clone()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.i_downsample:
            identity = self.i_downsample(identity)
        out += identity
        return self.relu(out)


class MyData_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(MyData_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, num_blocks, out_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, downsample, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ResNet Variants
def MyData_ResNet18(num_classes):
    return MyData_ResNet(Block, [2, 2, 2, 2], num_classes)

def MyData_ResNet50(num_classes):
    return MyData_ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def MyData_ResNet101(num_classes):
    return MyData_ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# ------------------ GRU ------------------
# class MyData_GRU(nn.Module):
#     def __init__(self, num_classes):
#         super(MyData_GRU, self).__init__()
#         self.gru = nn.GRU(input_size=4, hidden_size=64, num_layers=1, batch_first=False)
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # expected input shape: (batch, time_steps, features) = (batch, 1001, 4)
#         x = x.permute(1, 0, 2)  # (time_steps, batch, features)
#         _, h_n = self.gru(x)
#         return self.fc(h_n[-1])

class MyData_GRU(nn.Module):
    def __init__(self, num_classes, input_dim=(1000,4004), reduced_dim=512, hidden_dim=128):
        super().__init__()
        self.reduce = nn.Linear(input_dim, reduced_dim)
        self.gru = nn.GRU(input_size=reduced_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):          # x: (B, 200, 4004)
        x = self.reduce(x)         # (B, 200, 512)
        _, h_n = self.gru(x)       # h_n: (1, B, 128)
        return self.fc(h_n[-1])    # (B, num_classes)




# ------------------ CNN + GRU ------------------
# class MyData_CNN_GRU(nn.Module):
#     def __init__(self, num_classes):
#         super(MyData_CNN_GRU, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 16, 16, stride=8),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, 8, stride=4),
#             nn.ReLU(),
#         )
#         self.gru = nn.GRU(32, 128, num_layers=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         # input: (batch, 4004)
#         # x = x.view(x.size(0), 1, 4004)
#         x = self.encoder(x)  # shape: (batch, channels, seq_len)
#         x = x.permute(2, 0, 1)  # (seq_len, batch, channels)
#         _, h_n = self.gru(x)
#         return self.classifier(h_n[-1])

class MyData_CNN_GRU(nn.Module):
    def __init__(self, num_classes):
        super(MyData_CNN_GRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),  # (1,64,64) → (32,30,30)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (32,15,15)
            nn.Conv2d(32, 64, 3),           # (64,13,13)
            nn.ReLU()
        )

        # Infer flattened size from dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            dummy_feat = self.cnn(dummy)
            self.feature_size = dummy_feat.view(1, -1).shape[1]

        self.gru = nn.GRU(input_size=self.feature_size, hidden_size=128, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):  # x: (B, T, 1, 64, 64)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)
        _, h_n = self.gru(x)
        return self.classifier(h_n[-1])



# ---------------------- SNN ----------------------
class MyData_SNN(nn.Module):
    def __init__(self, num_classes, dt=0.001):
        super(MyData_SNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),  # (1, 64, 64) → (32, 30, 30)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),           # (64, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                # (64, 14, 14)
            nn.Conv2d(64, 96, 3),           # (96, 12, 12)
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            enc = self.encoder(dummy)
            self.flattened_size = enc.view(1, -1).shape[1]

        self.recurrent = LIFRecurrentCell(
            input_size=self.flattened_size,
            hidden_size=128,
            p=LIFParameters(alpha=10.0, v_th=torch.tensor(0.4)),
            dt=dt
        )

        self.readout_fc = nn.Linear(128, num_classes)
        self.readout_cell = LICell(dt=dt)

    def forward(self, x):  # x: (B, T, 1, 64, 64)
        B, T, C, H, W = x.shape
        s_recur = s_read = None
        voltages = []

        for t in range(T):
            frame = x[:, t]                  # (B, 1, 64, 64)
            z = self.encoder(frame)         # (B, conv_features)
            z = z.view(B, -1)
            z, s_recur = self.recurrent(z, s_recur)
            z = self.readout_fc(z)
            vo, s_read = self.readout_cell(z, s_read)
            voltages.append(vo)

        voltages = torch.stack(voltages)       # (T, B, num_classes)
        output, _ = torch.max(voltages, dim=0) # (B, num_classes)
        return F.log_softmax(output, dim=1)
