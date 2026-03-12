import torch
import torch.nn as nn


class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def make_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(f"Trước khi flatten: {x.shape}")
        x = x.view(x.shape[0], -1)
        # print(f"Sau khi flatten: {x.shape}")
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = SkinDiseaseModel()
    input_data = torch.rand(8, 3, 224, 224)
    while True:
        output = model(input_data)
        print(output.shape)
        break
