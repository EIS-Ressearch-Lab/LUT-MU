# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Patch added: brevitas-implemented QuantResNet
# Based on https://github.com/Xilinx/brevitas/blob/v0.12.1/src/brevitas_examples/bnn_pynq/models/resnet.py
from typing import List

from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor

from halutmatmul.modules import LUTMUConv2d, LUTMULinear

def make_quant_conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        weight_bit_width=8,
        weight_quant=Int8WeightPerChannelFloat):
    return LUTMUConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        weight_quant=weight_quant,
        weight_bit_width=weight_bit_width)


class QuantBasicBlock(nn.Module):
    """
    Quantized BasicBlock implementation with extra relu activations to respect FINN constraints on the sign of residual
    adds. Ok to train from scratch, but doesn't lend itself to e.g. retrain from torchvision.
    """
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
            bias=False,
            shared_quant_act=None,
            act_bit_width=8,
            weight_bit_width=8,
            weight_quant=Int8WeightPerChannelFloat):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = make_quant_conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = make_quant_conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                make_quant_conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    weight_bit_width=weight_bit_width,
                    weight_quant=weight_quant),
                nn.BatchNorm2d(self.expansion * planes),
                # We add a ReLU activation here because FINN requires the same sign along residual adds
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))
            # Redefine shared_quant_act whenever shortcut is performing downsampling
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        # We add a ReLU activation here because FINN requires the same sign along residual adds
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(return_quant_tensor=True, bit_width=act_bit_width)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if len(self.downsample):
            x = self.downsample(x)
        # Check that the addition is made explicitly among QuantTensor structures
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        out = out + x
        out = self.relu_out(out)
        return out


class QuantBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        bias=False,
        shared_quant_act=None,
        act_bit_width=8,
        weight_bit_width=8,
        weight_quant=Int8WeightPerChannelFloat,
    ) -> None:
        super(QuantBottleneck, self).__init__()
        # width = int(planes * (base_width / 64.0)) * groups
        # width = planes when base_width = 64 (default value)， groups = 1 (default value)
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = make_quant_conv2d(in_planes,
                                       planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=bias,
                                       weight_bit_width=weight_bit_width,
                                       weight_quant=weight_quant)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = make_quant_conv2d(planes, 
                                       planes, 
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=bias,
                                       weight_bit_width=weight_bit_width,
                                       weight_quant=weight_quant)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = make_quant_conv2d(planes, 
                                       planes * self.expansion,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=bias,
                                       weight_bit_width=weight_bit_width,
                                       weight_quant=weight_quant)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.stride = stride
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                make_quant_conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    weight_bit_width=weight_bit_width,
                    weight_quant=weight_quant),
                nn.BatchNorm2d(self.expansion * planes),
                # We add a ReLU activation here because FINN requires the same sign along residual adds
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))
            # Redefine shared_quant_act whenever shortcut is performing downsampling
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        # We add a ReLU activation here because FINN requires the same sign along residual adds
        self.relu3 = shared_quant_act
        self.relu_out = qnn.QuantReLU(return_quant_tensor=True, bit_width=act_bit_width)

    def forward(self, x: Tensor) -> Tensor:

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        
        if len(self.downsample):
            x = self.downsample(x)
        # Check that the addition is made explicitly among QuantTensor structures
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        out += x
        out = self.relu_out(out)

        return out


class QuantResNet(nn.Module):

    def __init__(
            self,
            block_impl,
            num_blocks: List[int],
            is_imagenet=False,
            zero_init_residual=False,
            num_classes=10,
            act_bit_width=8,
            weight_bit_width=8,
            round_average_pool=False,
            last_layer_bias_quant=Int32Bias,
            weight_quant=Int8WeightPerChannelFloat,
            first_layer_weight_quant=Int8WeightPerChannelFloat,
            last_layer_weight_quant=Int8WeightPerTensorFloat):
        super(QuantResNet, self).__init__()
        self.in_planes = 64
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.relu = shared_quant_act
        # MaxPool is typically present for ImageNet but not for CIFAR10
        if is_imagenet:
            self.conv1 = make_quant_conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                weight_bit_width=8,
                weight_quant=first_layer_weight_quant)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = make_quant_conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_bit_width=8,
                weight_quant=first_layer_weight_quant)
            self.maxpool = nn.Identity()

        self.layer1, shared_quant_act = self._make_layer(
            block_impl, 64, num_blocks[0], 1, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)
        self.layer2, shared_quant_act = self._make_layer(
            block_impl, 128, num_blocks[1], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)
        self.layer3, shared_quant_act = self._make_layer(
            block_impl, 256, num_blocks[2], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)
        self.layer4, _ = self._make_layer(
            block_impl, 512, num_blocks[3], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant)

        # Performs truncation to 8b (without rounding), which is supported in FINN
        avgpool_float_to_int_impl_type = 'ROUND' if round_average_pool else 'FLOOR'
        self.final_pool = qnn.TruncAvgPool2d(
            kernel_size=4,
            trunc_quant=TruncTo8bit,
            float_to_int_impl_type=avgpool_float_to_int_impl_type)
        # Keep last layer at 8b
        self.linear = LUTMULinear(
            512 * block_impl.expansion,
            num_classes,
            weight_bit_width=8,
            bias=True,
            bias_quant=last_layer_bias_quant,
            weight_quant=last_layer_weight_quant)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, LUTMUConv2d, LUTMULinear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, (LUTMUConv2d)) and m.halut_active:
                nn.init.kaiming_normal_(m.lut)
                nn.init.normal_(m.thresholds)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block_impl,
            planes,
            num_blocks,
            stride,
            shared_quant_act,
            weight_bit_width,
            act_bit_width,
            weight_quant):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = block_impl(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                bias=False,
                shared_quant_act=shared_quant_act,
                act_bit_width=act_bit_width,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant)
            layers.append(block)
            shared_quant_act = layers[-1].relu_out
            self.in_planes = planes * block_impl.expansion
        return nn.Sequential(*layers), shared_quant_act

    def forward(self, x: Tensor):
        # There is no input quantizer, we assume the input is already 8b RGB
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def quant_resnet18(weight_bit_width, act_bit_width, num_classes, is_imagenet = True) -> QuantResNet:

    model = QuantResNet(
        block_impl=QuantBasicBlock,
        num_blocks=[2, 2, 2, 2],
        is_imagenet=is_imagenet,
        num_classes=num_classes,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width)
    return model


def quant_resnet34(weight_bit_width, act_bit_width, num_classes, is_imagenet = True) -> QuantResNet:

    model = QuantResNet(
        block_impl=QuantBasicBlock,
        num_blocks=[3, 4, 6, 3],
        is_imagenet=is_imagenet,
        num_classes=num_classes,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width)
    return model

def quant_resnet50(weight_bit_width, act_bit_width, num_classes, is_imagenet = True) -> QuantResNet:

    model = QuantResNet(
        block_impl=QuantBottleneck,
        num_blocks=[3, 4, 6, 3],
        is_imagenet=is_imagenet,
        num_classes=num_classes,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width)
    return model