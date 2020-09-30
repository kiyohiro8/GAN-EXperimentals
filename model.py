
import math

import numpy as np
import torch
from torch import nn
#from torch.nn.functional import normalize
import torch.nn.functional as F
from torch.nn import init


def gen_channels(image_size):
    if image_size == 128:
        channels_dict = {
            "in_channels": [16, 16, 8, 4, 2],
            "out_channels": [16, 8, 4, 2, 1]
        }
    elif image_size == 256:
        channels_dict = {
            "in_channels": [8, 8, 8, 8, 4, 2],
            "out_channels": [8, 8, 8, 4, 2, 1]
        }
    return channels_dict


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        latent_dim = params["latent_dim"]
        base_channels = params["base_channels"]
        image_size = params["image_size"] # 2^n
        use_sn = params["use_sn_gen"]

        self.bottom_width = 4
        attention_res = 64
        attention_idx = int(np.log2(attention_res // self.bottom_width)) - 1
        channels = gen_channels(image_size)
        in_channels = [x * base_channels for x in channels["in_channels"]]
        out_channels = [x * base_channels for x in channels["out_channels"]]

        self.init_linear = nn.Linear(latent_dim, in_channels[0] * self.bottom_width * self.bottom_width)
        self.init_linear.apply(init_xavier_uniform)
        if use_sn:
            self.init_linear = SpectralNorm(self.init_linear)

        conv_block_list = []

        for i in range(len(in_channels)):
            conv_block_list.append(GeneratorUpsamplingBlock(in_channels[i], out_channels[i], upsampling_factor=2, use_sn=use_sn))
            if i == attention_idx:
                conv_block_list.append(SelfAttention(out_channels[i], use_sn=use_sn))

        self.upsampling_conv_block = nn.Sequential(*conv_block_list)
        
        self.output_block = nn.Sequential(nn.BatchNorm2d(out_channels[-1]),
                                         nn.ReLU(),
                                         nn.Conv2d(out_channels[-1], 3, 3, 1, 1))
        
        #self.output_block.apply(init_xavier_uniform)
    
        #init.xavier_uniform_(self.init_deconv.weight.data, math.sqrt(2))
        #init.xavier_uniform_(self.final_block.weight.data, math.sqrt(2))

    def forward(self, z):
        x = self.init_linear(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        x = self.upsampling_conv_block(x)
        x = self.output_block(x)
        return torch.tanh(x)


def dis_channels(image_size):
    if image_size == 128:
        channels_dict = {
            "in_channels": [1, 2, 4, 8, 16],
            "out_channels": [2, 4, 8, 16, 16]
        }
    elif image_size == 256:
        channels_dict = {
            "in_channels": [1, 2, 4, 8, 8, 8],
            "out_channels": [2, 4, 8, 8, 8, 8]
        }
    return channels_dict


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        base_channels = params["base_channels"]
        image_size = params["image_size"] #2^n
        attention_res = 64
        attention_idx =  int(np.log(image_size // attention_res)) - 1
        channels = dis_channels(image_size)
        in_channels = [3] + [ch * base_channels for ch in channels["in_channels"]]
        out_channels = [base_channels] + [ch * base_channels for ch in channels["out_channels"]]

        self.downsampling_conv_list = nn.ModuleList([])
        #self.init_conv = nn.Sequential(spectral_norm(nn.Conv2d(3, in_channels, 3, 1, 1)))
        self.unet_dis = params["unet_dis"]

        for i in range(len(in_channels)):
            if i == 0:
                self.downsampling_conv_list.append(DiscriminatorDownsamplingBlock(in_channels[i], out_channels[i], downsampling_factor=2, preactivation=False, unet=self.unet_dis))
            else:
                self.downsampling_conv_list.append(DiscriminatorDownsamplingBlock(in_channels[i], out_channels[i], downsampling_factor=2, preactivation=True, unet=self.unet_dis))
            if i == attention_idx and not self.unet_dis:
                self.downsampling_conv_list.append(SelfAttention(out_channels[i], use_sn=True))

        # self.downsampling_conv = nn.Sequential(*downsampling_conv_list)
        self.output_block = nn.Sequential(SpectralNorm(nn.Linear(out_channels[-1], 1)))

        #self.activation = nn.LeakyReLU(0.2)
        self.activation = nn.ReLU()

        if self.unet_dis:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            dec_channels = gen_channels(image_size)
            dec_in_channels = [ch * base_channels for ch in channels["in_channels"]]
            dec_out_channels = [ch * base_channels for ch in channels["out_channels"]]
            self.upsampling_conv_list = nn.ModuleList([])
            for i in range(len(in_channels) - 1):
                if i == 0:
                    self.upsampling_conv_list.append(DiscriminatorUpsamplingBlock(out_channels[-1] * 2, in_channels[-1], upsampling_factor=1))
                else:
                    self.upsampling_conv_list.append(DiscriminatorUpsamplingBlock(out_channels[-i-1] + in_channels[-i], in_channels[-i-1], upsampling_factor=1))
            self.unet_output_block = nn.Sequential(nn.LeakyReLU(0.2),
                                                   SpectralNorm(nn.Conv2d(in_channels[1], 1, 3, 1, 1)),
                                                   nn.LeakyReLU(0.2),
                                                   SpectralNorm(nn.Conv2d(1, 1, 1, 1, 0)))

        #init.xavier_uniform_(self.init_conv.weight.data, math.sqrt(2))
        #init.xavier_uniform_(self.final_conv.weight.data, math.sqrt(2))

    def forward(self, x):
        sc_list = []
        for block in self.downsampling_conv_list:
            if self.unet_dis:
                x, sc = block(x)
                sc_list.append(sc)
            else:
                x = block(x)
        enc_out = torch.mean(self.activation(x), dim=(2, 3))
        enc_out = self.output_block(enc_out)
        if not self.unet_dis:
            return enc_out
        for i, block in enumerate(self.upsampling_conv_list):
            x = self.upsample(x)
            x = torch.cat([x, sc_list[-i-1]], dim=1)
            x = block(x)
        x = self.upsample(x)
        dec_out = self.unet_output_block(x)
        return enc_out, dec_out


class GeneratorUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, upsampling_factor=1, use_sn=False):
        super(GeneratorUpsamplingBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation=dilation)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, dilation, dilation=dilation)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.conv_1.apply(init_xavier_uniform)
        self.conv_2.apply(init_xavier_uniform)
        self.conv_shortcut.apply(init_xavier_uniform)

        self.use_sn = use_sn

        if self.use_sn:
            self.conv_1 = SpectralNorm(self.conv_1)
            self.conv_2 = SpectralNorm(self.conv_2)
            self.conv_shortcut = SpectralNorm(self.conv_shortcut)

        self.norm_1 = nn.BatchNorm2d(in_channels)
        self.norm_2 = nn.BatchNorm2d(out_channels)

        #self.norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        #self.norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)

        self.activate = nn.ReLU()
        
        self.upsampling_factor = upsampling_factor
        if self.upsampling_factor > 1:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsampling_factor)
    
    def forward(self, input):
        x = self.activate(self.norm_1(input))

        if self.upsampling_factor > 1:
            x = self.upsample(x)
            shortcut = self.upsample(input)
        else:
            shortcut = input
        x = self.conv_1(x)

        x = self.conv_2(self.activate(self.norm_2(x)))

        shortcut = self.conv_shortcut(shortcut)

        return x + shortcut

class DiscriminatorDownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, downsampling_factor=1, preactivation=False, unet=False):
        super(DiscriminatorDownsamplingBlock, self).__init__()
        self.conv_1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation=dilation))
        self.conv_2 = SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, dilation, dilation=dilation))
        self.conv_shortcut = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))

        self.conv_1.apply(init_xavier_uniform)
        self.conv_2.apply(init_xavier_uniform)
        self.conv_shortcut.apply(init_xavier_uniform)

        self.downsampling_factor = downsampling_factor
        if self.downsampling_factor > 1:
            self.downsample = nn.AvgPool2d(kernel_size=self.downsampling_factor)

        #self.activate = nn.LeakyReLU(0.2)
        self.activate = nn.ReLU()
        self.preactivation = preactivation
        self.unet = unet

    def forward(self, input):
        if self.preactivation:
            x = self.activate(input)
        else:
            x = input
        x = self.activate(self.conv_1(x))
        x = self.conv_2(x)

        #shortcut = self.conv_shortcut(input)

        if self.downsampling_factor > 1:
            x_ds = self.downsample(x)
            if self.preactivation:
                shortcut = self.conv_shortcut(input)
                shortcut = self.downsample(shortcut)
            else:
                shortcut = self.downsample(input)
                shortcut = self.conv_shortcut(shortcut)
        if not self.unet:
            return x_ds + shortcut
        else:
            return x_ds + shortcut, x


class DiscriminatorUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, upsampling_factor=1):
        super(DiscriminatorUpsamplingBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation=dilation)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, dilation, dilation=dilation)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.conv_1.apply(init_xavier_uniform)
        self.conv_2.apply(init_xavier_uniform)
        self.conv_shortcut.apply(init_xavier_uniform)

        self.conv_1 = SpectralNorm(self.conv_1)
        self.conv_2 = SpectralNorm(self.conv_2)
        self.conv_shortcut = SpectralNorm(self.conv_shortcut)

        self.activate = nn.ReLU()
        
        self.upsampling_factor = upsampling_factor
        if self.upsampling_factor > 1:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsampling_factor)
    
    def forward(self, input):
        x = self.activate(input)

        if self.upsampling_factor > 1:
            x = self.upsample(x)
            shortcut = self.upsample(input)
        else:
            shortcut = input
        x = self.conv_1(x)

        x = self.conv_2(self.activate(x))

        shortcut = self.conv_shortcut(shortcut)

        return x + shortcut


class SelfAttention(nn.Module):
    def __init__(self, ch, use_sn, name='attention'):
        super(SelfAttention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(ch, ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(ch, ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(ch // 2, ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = SpectralNorm(self.theta)
            self.phi = SpectralNorm(self.phi)
            self.g = SpectralNorm(self.g)
            self.o = SpectralNorm(self.o)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)
