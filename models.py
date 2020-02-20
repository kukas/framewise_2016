import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


from math import log2

class HarmonicStacking(nn.Module):
    def __init__(self, octave_bins, undertones, overtones):
        super().__init__()

        self.octave_bins = octave_bins
        self.overtones = overtones
        self.undertones = undertones
        self.shifts = []
        for harmonic in [1/(x+2) for x in range(undertones)]+list(range(1, overtones+1)):
            self.shifts.append(round(octave_bins*log2(harmonic)))
        
        print(self.shifts)

        # for offset in [octave_bins*math.log2(x) for x in range()]:
    def forward(self, x):
        # print(x.shape, "-->")
        channels = []
        for shift in self.shifts:
            if shift == 0:
                padded = x
            if shift > 0:
                shifted = x[:, :, :, shift:]
                padded = F.pad(shifted, (0, shift))
            elif shift < 0:
                shifted = x[:, :, :, :shift]
                padded = F.pad(shifted, (-shift, 0))

            channels.append(padded)
        x = torch.cat(channels, 1)
        # print(x.shape)
        return x

class VGGNet2016(nn.Module):
    def __init__(self, args):
        super().__init__()
        # these parameters are set to what lasagne.layers.BatchNorm implements
        bn_param = dict(
            eps=1e-4,      # just like in lasagne
            momentum=0.1,  # 'alpha' in lasagne
            affine=True,   # we learn a translation, called 'beta' in the paper and lasagne
            track_running_stats=True
        )

        cap = args.capacity
        self.conv = nn.Sequential(
            nn.Conv2d(1, cap, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(cap, **bn_param),
            nn.ReLU(),

            nn.Conv2d(cap, cap, (3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(cap, **bn_param),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),

            nn.Conv2d(cap, cap*2, (3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(cap*2, **bn_param),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),
        )

        self.n_flat = cap*2 * 1 * 55
        self.linear = nn.Sequential(
            nn.Linear(self.n_flat, 512, bias=False),
            nn.BatchNorm1d(512, **bn_param),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 88)
            # the sigmoid nonlinearity is not missing!
            # during training we do not want it to be applied, only during prediction!
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        init.xavier_uniform_(self.linear[-1].weight, init.calculate_gain('sigmoid'))

    # returns logits!
    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, self.n_flat)
        h = self.linear(h)
        return h

    # returns pseudo probabilities
    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class AllConv2016(nn.Module):
    def __init__(self, args):
        super().__init__()

        # these parameters are set to what lasagne.layers.BatchNorm implements
        bn_param = dict(
            eps=1e-4,      # just like in lasagne
            momentum=0.1,  # 'alpha' in lasagne
            affine=True,   # we learn a translation, called 'beta' in the paper and lasagne
            track_running_stats=True
        )
        hcnn_mult = args.hcnn_undertones+args.hcnn_overtones
        conv_in_cap = args.capacity # input capacity for ordinary conv layers
        hcnn_conv_in_cap = conv_in_cap if args.hcnn_onlyinput else conv_in_cap * hcnn_mult # input capacity for conv layers after harmonic stacking
        conv_out_cap = args.capacity
        self.conv = nn.Sequential(
            HarmonicStacking(48, args.hcnn_undertones, args.hcnn_overtones),
            nn.Conv2d(1*hcnn_mult, conv_out_cap, (3, 3), padding=(0, 0), bias=False),
            # the next two layers were not in the paper description,
            # but they should have been! (it does not change very much though)
            nn.BatchNorm2d(conv_out_cap, **bn_param),
            nn.ReLU(),

            HarmonicStacking(48, args.hcnn_undertones, args.hcnn_overtones) if not args.hcnn_onlyinput else nn.Identity(),
            nn.Conv2d(hcnn_conv_in_cap, conv_out_cap, (3, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(conv_out_cap, **bn_param),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(p=0.25),

            HarmonicStacking(24, args.hcnn_undertones, args.hcnn_overtones) if not args.hcnn_onlyinput else nn.Identity(),
            nn.Conv2d(hcnn_conv_in_cap, conv_out_cap, (1, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(conv_out_cap, **bn_param),
            nn.ReLU(),

            HarmonicStacking(24, args.hcnn_undertones, args.hcnn_overtones) if not args.hcnn_onlyinput else nn.Identity(),
            nn.Conv2d(hcnn_conv_in_cap, conv_out_cap, (1, 3), padding=(0, 0), bias=False),
            nn.BatchNorm2d(conv_out_cap, **bn_param),
            nn.ReLU(),

            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.25),

            HarmonicStacking(12, args.hcnn_undertones, args.hcnn_overtones) if not args.hcnn_onlyinput else nn.Identity(),
            nn.Conv2d(hcnn_conv_in_cap, conv_out_cap*2, (1, 25), padding=(0, 0), bias=False),
            nn.BatchNorm2d(conv_out_cap*2, **bn_param),
            nn.ReLU(),

            nn.Conv2d(conv_in_cap*2, conv_out_cap*4, (1, 25), padding=(0, 0), bias=False),
            nn.BatchNorm2d(conv_out_cap*4, **bn_param),
            nn.ReLU(),

            nn.Conv2d(conv_in_cap*4, 88, (1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(88, **bn_param),
            nn.AvgPool2d((1, 6))
            # the sigmoid nonlinearity is not missing!
            # during training we do not want it to be applied, only during prediction!
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # returns logits!
    def forward(self, x):
        h = self.conv(x)
        return h.squeeze(-1).squeeze(-1)

    # returns pseudo probabilities
    def predict(self, x):
        return torch.sigmoid(self.forward(x))


# very hacky thing!
def get_model_classes():
    current_module = __import__(__name__)

    class_names = []
    for name, clazz in current_module.__dict__.items():
        # get all types
        if isinstance(clazz, type):
            derived_from_torch_module = False
            for base in clazz.__bases__:
                # this would be bad juju in all other cases ...
                if type(clazz) == type(nn.Module):
                    derived_from_torch_module = True
                    break

            if derived_from_torch_module:
                class_names.append(name)

    return class_names


def main():
    x = torch.empty(2, 1, 5, 229).uniform_(0, 1)

    print('#' * 30)
    print('testing AllConv2016 shape')
    net = AllConv2016()
    y = net(x)

    print('x.size()', x.size())
    print('y.size()', y.size())

    print('#' * 30)
    print('testing VGGNet2016 shape')
    net = VGGNet2016()
    y = net(x)

    print('x.size()', x.size())
    print('y.size()', y.size())


if __name__ == '__main__':
    main()
