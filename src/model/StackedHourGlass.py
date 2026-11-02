import torch
import torch.nn as nn


# Bottle Neck Bit
class BottleNeckBlock(nn.Module):
    def __init__(self, in_plane, out_plane):
        super().__init__()

        self.mid = out_plane // 2
        self.conv_layer_1 = nn.Conv2d(
            in_channels=in_plane,
            out_channels=self.mid,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0,
        )
        self.bn_1 = nn.BatchNorm2d(
            num_features=self.mid,
        )

        self.conv_layer_2 = nn.Conv2d(
            in_channels=self.mid,
            out_channels=self.mid,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(
            num_features=self.mid,
        )

        self.conv_layer_3 = nn.Conv2d(
            in_channels=self.mid,
            out_channels=out_plane,
            kernel_size = 1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(
            num_features=out_plane,
        )


        ## This is for the resedual bit
        if in_plane != out_plane:
            self.skip = nn.Sequential(
                nn.Conv2d(in_plane, out_plane, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_plane)
            )
        else:
            self.skip = nn.Identity()

        ## ReLU bit  --> Not explained why only relu, why not leaky relu etc
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, input):
        res = self.skip(input)

        out = self.conv_layer_1(input)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_layer_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        out = self.conv_layer_3(out)
        out = self.bn_3(out)

        out += res

        return self.relu(out)


# Hour Glass Bit
class HourGlassBlock(nn.Module):
    def __init__(self, depth, nf):
        super().__init__()
        self.depth = depth
        self.nf = nf

        self.b1 = BottleNeckBlock(nf, nf)

        self.pool_layer_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.b2 = BottleNeckBlock(nf, nf)

        if self.depth > 1:
            self.l2 = HourGlassBlock(self.depth - 1, nf)
        else:
            self.l2 = BottleNeckBlock(nf, nf)

        self.l3 = BottleNeckBlock(nf, nf)

    def forward(self, input):
        out0 = self.b1(input)

        out1 = self.pool_layer_1(input)
        out1 = self.b2(out1)
        out2 = self.l2(out1)
        out3 = self.l3(out2)
        
        out4 = nn.functional.interpolate(
            input=out3,
            scale_factor=2,
            mode='nearest'
        )

        return out0 + out4


# Stacked HourGlass Bit
class StackedHourGlassBlock(nn.Module):
    def __init__(self, num_stacks=2, num_joints=21, depth=4, channels=256):
        super().__init__()
        self.num_stacks = num_stacks
        
        self.conv_layer_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )

        self.bn_layer_1 = nn.BatchNorm2d(
            num_features=64
        ) 

        self.res_layer_1 = BottleNeckBlock(64, 128)
        
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.res_layer_2 = BottleNeckBlock(128, 128)
        self.res_layer_3 = BottleNeckBlock(128, channels)

        self.hgs = nn.ModuleList([HourGlassBlock(depth, channels) for _ in range(num_stacks)])

        self.features = nn.ModuleList([
            nn.Sequential(
                BottleNeckBlock(channels, channels),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stacks)
        ])

        self.outs = nn.ModuleList([
            nn.Conv2d(channels, num_joints, kernel_size=1, bias=True)
            for _ in range(num_stacks)
        ])

        self.merge_preds = nn.ModuleList([
            nn.Conv2d(num_joints, channels, kernel_size=1, bias=False)
            for _ in range(num_stacks - 1)
        ])

        self.merge_feats = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for _ in range(num_stacks - 1)
        ])
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.conv_layer_1(input)
        out = self.bn_layer_1(out)
        out = self.relu(out)

        out = self.res_layer_1(out)
        out = self.pool(out)
        out = self.res_layer_2(out)
        out = self.res_layer_3(out)

        combinations = []

        for i in range(self.num_stacks):
            hg_feat = self.hgs[i](out)
            feat = self.features[i](hg_feat)

            out1 = self.outs[i](feat)
            combinations.append(out1)

            if i < self.num_stacks - 1:
                out = out + self.merge_feats[i](feat) + self.merge_preds[i](out1)
            
        return combinations


