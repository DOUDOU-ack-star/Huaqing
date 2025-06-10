import torch
from torch import nn


class alex(nn.Module):
    def __init__(self, num_classes=10):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            # 第一层卷积和池化，减小步长以保留更多空间信息
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 输入:32x32, 输出:32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出:16x16

            # 第二层卷积和池化
            nn.Conv2d(48, 128, kernel_size=3, padding=1),  # 输出:16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出:8x8

            # 第三到五层卷积，使用更小的卷积核
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 输出:8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # 输出:8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 输出:8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出:4x4

            nn.Flatten(),
            # 调整全连接层输入维度为128*4*4
            nn.Linear(128 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        y = self.model(x)
        return y


# 测试模型
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)  # 使用32x32的输入
    model = alex()
    y = model(x)
    print(y.shape)  # 应该输出torch.Size([1, 10])