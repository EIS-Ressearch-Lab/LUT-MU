# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from typing import List
import torch
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear

from functools import reduce
from operator import mul

from .SupportLib.common import CommonActQuant
from .SupportLib.common import CommonWeightQuant
from .SupportLib.tensor_norm import TensorNorm

from halutmatmul.modules import LUTMULinear

DROPOUT = 0.2


class QuantFC(Module):

    def __init__(
        self,
        num_classes,
        weight_bit_width,
        act_bit_width,
        in_bit_width,
        out_features,
        in_features=(28, 28)):
        super(QuantFC, self).__init__()

        self.features = ModuleList()
        self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width))
        self.features.append(Dropout(p=DROPOUT))
        in_features = reduce(mul, in_features)
        
        for out_feature in out_features:
            self.features.append(
                LUTMULinear(
                    in_features=in_features,
                    out_features=out_feature,
                    bias=False,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonWeightQuant))
            in_features = out_feature
            
            self.features.append(BatchNorm1d(num_features=in_features))
            self.features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            self.features.append(Dropout(p=DROPOUT))
            
        self.features.append(
            LUTMULinear(
                in_features=in_features,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant))
        self.features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, (QuantLinear, LUTMULinear)):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.features:
            if isinstance(mod, (QuantLinear, LUTMULinear)):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.features:
            x = mod(x)
        return x


def quant_fc(weight_bit_width: int,
             act_bit_width: int,
             in_bit_width: int,
             out_features: List[int],
             num_classes: int) -> QuantFC:

    net = QuantFC(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        out_features=out_features,
        num_classes=num_classes)
    
    return net