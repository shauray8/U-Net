import torch
import torch.nn as nn
import torch.nn.functional as F


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()
        self.model = []
        pass


    def Conv_layer(self, in_channel, out_channel, mid):
        if not mid:
            mid = out_channel

        self.convolution = nn.Sequential(
                nn.Conv2d(in_channel, mid, kernel_size=3, padding=1),
                nn.batchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channel, kernel_size=3, padding=1),
                nn.batchNorm2d(out_channel),
                nn.ReLU(inplace=True))

        return self.convolution

    def Down_Sampling(self, in_channel, out_channel):
        self.Down = nn.Sequential(
                nn.MaxPool2d(2),
                Conv_layer(in_channel, out_channel))

        return self.Down

    



if __name__ == "__main__":
    Unet = U_net()
    print(Unet)

