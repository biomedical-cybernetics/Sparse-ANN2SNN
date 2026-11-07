import torch.nn as nn

class MyCifarANNDense(nn.Module):
    """Dense (non-CHT) version of user-provided CIFAR architecture.

    Structure:
    Block1: 3 -> 128 -> 128 -> MaxPool (32->16)
    Block2: 128 -> 256 -> 256 -> MaxPool (16->8)
    Block3: 256 -> 512 -> 512 (stay 8x8)
    Head: Flatten -> FC 512*8*8 -> 512 -> ReLU -> (optional hidden fc) -> last_layer.
    """
    def __init__(self, num_classes: int, dropout: float = 0.0, one_fc: bool = False):
        super().__init__()
        layers = []
        # Block 1
        layers += [
            nn.Conv2d(3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.MaxPool2d(2)
        ]
        # Block 2
        layers += [
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.MaxPool2d(2)
        ]
        # Block 3
        layers += [
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout(dropout)
        ]
        self.features = nn.Sequential(*layers)
        # Mirror SD-SNN: Flatten -> Linear(512*8*8 -> 512) -> ReLU; last_layer: 512 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
        )
        self.last_layer = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # classifier performs Flatten internally
        x = self.classifier(x)
        x = self.last_layer(x)
        return x
