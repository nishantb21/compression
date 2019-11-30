import torch

def get_convolution(in_channels, out_channels, kernel_size, stride, padding, activation="leaky"):
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    if activation == "leaky":
        act = torch.nn.LeakyReLU(out_channels)

    else:
        act = torch.nn.Identity()

    return torch.nn.Sequential(conv, act)

class Net(torch.nn.Module):
    def __init__(self, height, width, channels, classes):
        super(Net, self).__init__()
        self.conv_1 = get_convolution(channels, 16, 3, 1, 1)
        self.conv_2 = get_convolution(16, 32, 3, 1, 1)
        self.flatten = torch.nn.modules.Flatten()
        self.dense_1 = torch.nn.Linear(height*width*32, 1024)
        self.dense_2 = torch.nn.Linear(1024, classes)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x