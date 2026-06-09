import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torchvision.models.quantization.utils import _fuse_modules, quantize_model
from halutmatmul.modules import HalutConv2d, HalutLinear
from halutmatmul.modules import LUTMUConv2d, LUTMULinear


from brevitas.nn import QuantReLU
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear

from brevitas.quant import Int32Bias # brevitas 0.12.1
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int8WeightPerChannelFloat

def qat_conv_block(
    in_channels, out_channels, pool=False, 
    halut_active=False, use_torch_conv=False,
    act_bit_width = 8, weight_bit_width = 2, relu = None,
):
    # FINN requires the same sign along residual adds
    if relu is None:
        relu = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
    layers = [
        LUTMUConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            halut_active=halut_active,
            split_factor=4,
            weight_quant=Int8WeightPerChannelFloat,
            weight_bit_width=weight_bit_width,
        )
        if not use_torch_conv
        else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU(inplace=True),
        relu,
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers), relu

halut_active = False
use_torch_conv = False  # use for quantization aware training

# need rewrite init for QAT
def _weights_init(m):
    if isinstance(m, (HalutLinear, HalutConv2d)):
        init.kaiming_normal_(m.weight)
    if isinstance(m, (HalutConv2d)) and m.halut_active:
        init.kaiming_normal_(m.lut)
        init.normal_(m.thresholds)
        
def _qat_weights_init(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    if isinstance(m, (LUTMUConv2d, LUTMULinear)) and m.halut_active:
        init.kaiming_normal_(m.lut)
        init.normal_(m.thresholds)


class QuantResNet9(nn.Module):
    def __init__(self, 
                 in_channels, 
                 num_classes,
                 act_bit_width=4,
                 weight_bit_width=4,):
        super().__init__()

        self.conv1, _ = qat_conv_block(in_channels, 64, 
                                    use_torch_conv=use_torch_conv,
                                    act_bit_width = act_bit_width,
                                    weight_bit_width = weight_bit_width,)
        self.conv2, relu = qat_conv_block(
            64, 128, pool=True, halut_active=halut_active, use_torch_conv=use_torch_conv,
            act_bit_width = act_bit_width,
            weight_bit_width = weight_bit_width,
        )
        self.res1 = nn.Sequential(
            qat_conv_block(
                128, 128, halut_active=halut_active, use_torch_conv=use_torch_conv,
                act_bit_width = act_bit_width,
                weight_bit_width = weight_bit_width,
            )[0],
            qat_conv_block(
                128, 128, halut_active=halut_active, use_torch_conv=use_torch_conv,
                act_bit_width = act_bit_width,
                weight_bit_width = weight_bit_width, relu = relu
            )[0],
        )
        self.conv3, _ = qat_conv_block(
            128,
            256,
            pool=True,
            halut_active=halut_active,
            use_torch_conv=use_torch_conv,
            act_bit_width = act_bit_width,
            weight_bit_width = weight_bit_width,
        )
        self.conv4, relu = qat_conv_block(
            256,
            256,
            pool=True,
            halut_active=halut_active,
            use_torch_conv=use_torch_conv,
            act_bit_width = act_bit_width,
            weight_bit_width = weight_bit_width,
        )
        self.res2 = nn.Sequential(
            qat_conv_block(
                256, 256, halut_active=halut_active, use_torch_conv=use_torch_conv,
                act_bit_width = act_bit_width,
                weight_bit_width = weight_bit_width,
            )[0],
            qat_conv_block(
                256, 256, halut_active=halut_active, use_torch_conv=use_torch_conv,
                act_bit_width = act_bit_width,
                weight_bit_width = weight_bit_width, relu = relu
            )[0],
        )
        self.maxpool = nn.MaxPool2d(4)
        self.classifier = nn.Sequential(
            LUTMULinear(256, num_classes, 
                        weight_quant=Int8WeightPerTensorFloat,
                        bias_quant=Int32Bias,
                        weight_bit_width=weight_bit_width,)
            if not use_torch_conv
            else nn.Linear(256, num_classes)
        )
        self.apply(_qat_weights_init)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.maxpool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

    def fuse_model(self, is_qat=False):
        _fuse_modules(
            self,
            [
                ["conv1.0", "conv1.1", "conv1.2"],
                ["conv2.0", "conv2.1", "conv2.2"],
                ["res1.0.0", "res1.0.1", "res1.0.2"],
                ["res1.1.0", "res1.1.1", "res1.1.2"],
                ["conv3.0", "conv3.1", "conv3.2"],
                ["conv4.0", "conv4.1", "conv4.2"],
                ["res2.0.0", "res2.0.1", "res2.0.2"],
                ["res2.1.0", "res2.1.1", "res2.1.2"],
            ],
            is_qat=is_qat,
            inplace=True,
        )
        return self

    
if __name__ == "__main__":
    model = QuantResNet9(3, 10).to("cpu")
    print(model)

    from torchinfo import summary

    summary(model, input_size=(1, 3, 32, 32))

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(
        model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
