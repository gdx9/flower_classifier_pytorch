import torch.nn as nn
import torch.nn.functional as F

class Bn(nn.Module):
    def __init__(self, num_in, num_cent, num_out):
        super(Bn, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels=num_in, out_channels=num_cent, kernel_size=3, padding=1, stride=1,bias=False),
            nn.BatchNorm2d(num_cent, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=num_cent, out_channels=num_cent, kernel_size=3, padding=1, stride=1,bias=False),
            nn.BatchNorm2d(num_cent, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=num_cent, out_channels=num_out, kernel_size=3, padding=1, stride=1,bias=False),
            nn.BatchNorm2d(num_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        )

        self.layers2 = nn.Sequential(
            nn.Conv2d(num_in, num_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_out)
        )

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x)

        return F.relu(x1 + x2)

class FlowerModel(nn.Module):
    def __init__(self):
        super(FlowerModel, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=(3, 3), stride=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.l2 = Bn(64,128,64)
        self.l3 = Bn(64,256,64)
        self.pool = nn.AvgPool2d((4,4))
        self.l4 = Bn(64,64,32)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1568, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        x1 = self.l1(x)
        x1 = F.dropout(x1, 0.2)
        x2 = self.l2(x1)
        x3 = x1+x2

        x4 = F.max_pool2d(x3,2)

        x5 = self.l3(x4)
        x5 = F.dropout(x5, 0.25)

        x6 = self.l4(x5)
        x6 = F.dropout(x6, 0.25)

        out = self.pool(x6)
        out = self.classifier(out)

        return out

    def _init_weights(self,layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight)
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, 0)
