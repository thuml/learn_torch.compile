from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    l__mod___inc_double_conv_0 = self.L__mod___inc_double_conv_0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___inc_double_conv_1 = self.L__mod___inc_double_conv_1(l__mod___inc_double_conv_0);  l__mod___inc_double_conv_0 = None
    l__mod___inc_double_conv_2 = self.L__mod___inc_double_conv_2(l__mod___inc_double_conv_1);  l__mod___inc_double_conv_1 = None
    l__mod___inc_double_conv_3 = self.L__mod___inc_double_conv_3(l__mod___inc_double_conv_2);  l__mod___inc_double_conv_2 = None
    l__mod___inc_double_conv_4 = self.L__mod___inc_double_conv_4(l__mod___inc_double_conv_3);  l__mod___inc_double_conv_3 = None
    x1 = self.L__mod___inc_double_conv_5(l__mod___inc_double_conv_4);  l__mod___inc_double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    l__mod___down1_maxpool_conv_0 = self.L__mod___down1_maxpool_conv_0(x1)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    getattr_l__mod___down1_maxpool_conv___1___double_conv_0 = self.getattr_L__mod___down1_maxpool_conv___1___double_conv_0(l__mod___down1_maxpool_conv_0);  l__mod___down1_maxpool_conv_0 = None
    getattr_l__mod___down1_maxpool_conv___1___double_conv_1 = self.getattr_L__mod___down1_maxpool_conv___1___double_conv_1(getattr_l__mod___down1_maxpool_conv___1___double_conv_0);  getattr_l__mod___down1_maxpool_conv___1___double_conv_0 = None
    getattr_l__mod___down1_maxpool_conv___1___double_conv_2 = self.getattr_L__mod___down1_maxpool_conv___1___double_conv_2(getattr_l__mod___down1_maxpool_conv___1___double_conv_1);  getattr_l__mod___down1_maxpool_conv___1___double_conv_1 = None
    getattr_l__mod___down1_maxpool_conv___1___double_conv_3 = self.getattr_L__mod___down1_maxpool_conv___1___double_conv_3(getattr_l__mod___down1_maxpool_conv___1___double_conv_2);  getattr_l__mod___down1_maxpool_conv___1___double_conv_2 = None
    getattr_l__mod___down1_maxpool_conv___1___double_conv_4 = self.getattr_L__mod___down1_maxpool_conv___1___double_conv_4(getattr_l__mod___down1_maxpool_conv___1___double_conv_3);  getattr_l__mod___down1_maxpool_conv___1___double_conv_3 = None
    x2 = self.getattr_L__mod___down1_maxpool_conv___1___double_conv_5(getattr_l__mod___down1_maxpool_conv___1___double_conv_4);  getattr_l__mod___down1_maxpool_conv___1___double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    l__mod___down2_maxpool_conv_0 = self.L__mod___down2_maxpool_conv_0(x2)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    getattr_l__mod___down2_maxpool_conv___1___double_conv_0 = self.getattr_L__mod___down2_maxpool_conv___1___double_conv_0(l__mod___down2_maxpool_conv_0);  l__mod___down2_maxpool_conv_0 = None
    getattr_l__mod___down2_maxpool_conv___1___double_conv_1 = self.getattr_L__mod___down2_maxpool_conv___1___double_conv_1(getattr_l__mod___down2_maxpool_conv___1___double_conv_0);  getattr_l__mod___down2_maxpool_conv___1___double_conv_0 = None
    getattr_l__mod___down2_maxpool_conv___1___double_conv_2 = self.getattr_L__mod___down2_maxpool_conv___1___double_conv_2(getattr_l__mod___down2_maxpool_conv___1___double_conv_1);  getattr_l__mod___down2_maxpool_conv___1___double_conv_1 = None
    getattr_l__mod___down2_maxpool_conv___1___double_conv_3 = self.getattr_L__mod___down2_maxpool_conv___1___double_conv_3(getattr_l__mod___down2_maxpool_conv___1___double_conv_2);  getattr_l__mod___down2_maxpool_conv___1___double_conv_2 = None
    getattr_l__mod___down2_maxpool_conv___1___double_conv_4 = self.getattr_L__mod___down2_maxpool_conv___1___double_conv_4(getattr_l__mod___down2_maxpool_conv___1___double_conv_3);  getattr_l__mod___down2_maxpool_conv___1___double_conv_3 = None
    x3 = self.getattr_L__mod___down2_maxpool_conv___1___double_conv_5(getattr_l__mod___down2_maxpool_conv___1___double_conv_4);  getattr_l__mod___down2_maxpool_conv___1___double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    l__mod___down3_maxpool_conv_0 = self.L__mod___down3_maxpool_conv_0(x3)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    getattr_l__mod___down3_maxpool_conv___1___double_conv_0 = self.getattr_L__mod___down3_maxpool_conv___1___double_conv_0(l__mod___down3_maxpool_conv_0);  l__mod___down3_maxpool_conv_0 = None
    getattr_l__mod___down3_maxpool_conv___1___double_conv_1 = self.getattr_L__mod___down3_maxpool_conv___1___double_conv_1(getattr_l__mod___down3_maxpool_conv___1___double_conv_0);  getattr_l__mod___down3_maxpool_conv___1___double_conv_0 = None
    getattr_l__mod___down3_maxpool_conv___1___double_conv_2 = self.getattr_L__mod___down3_maxpool_conv___1___double_conv_2(getattr_l__mod___down3_maxpool_conv___1___double_conv_1);  getattr_l__mod___down3_maxpool_conv___1___double_conv_1 = None
    getattr_l__mod___down3_maxpool_conv___1___double_conv_3 = self.getattr_L__mod___down3_maxpool_conv___1___double_conv_3(getattr_l__mod___down3_maxpool_conv___1___double_conv_2);  getattr_l__mod___down3_maxpool_conv___1___double_conv_2 = None
    getattr_l__mod___down3_maxpool_conv___1___double_conv_4 = self.getattr_L__mod___down3_maxpool_conv___1___double_conv_4(getattr_l__mod___down3_maxpool_conv___1___double_conv_3);  getattr_l__mod___down3_maxpool_conv___1___double_conv_3 = None
    x4 = self.getattr_L__mod___down3_maxpool_conv___1___double_conv_5(getattr_l__mod___down3_maxpool_conv___1___double_conv_4);  getattr_l__mod___down3_maxpool_conv___1___double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:39, code: return self.maxpool_conv(x)
    l__mod___down4_maxpool_conv_0 = self.L__mod___down4_maxpool_conv_0(x4)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    getattr_l__mod___down4_maxpool_conv___1___double_conv_0 = self.getattr_L__mod___down4_maxpool_conv___1___double_conv_0(l__mod___down4_maxpool_conv_0);  l__mod___down4_maxpool_conv_0 = None
    getattr_l__mod___down4_maxpool_conv___1___double_conv_1 = self.getattr_L__mod___down4_maxpool_conv___1___double_conv_1(getattr_l__mod___down4_maxpool_conv___1___double_conv_0);  getattr_l__mod___down4_maxpool_conv___1___double_conv_0 = None
    getattr_l__mod___down4_maxpool_conv___1___double_conv_2 = self.getattr_L__mod___down4_maxpool_conv___1___double_conv_2(getattr_l__mod___down4_maxpool_conv___1___double_conv_1);  getattr_l__mod___down4_maxpool_conv___1___double_conv_1 = None
    getattr_l__mod___down4_maxpool_conv___1___double_conv_3 = self.getattr_L__mod___down4_maxpool_conv___1___double_conv_3(getattr_l__mod___down4_maxpool_conv___1___double_conv_2);  getattr_l__mod___down4_maxpool_conv___1___double_conv_2 = None
    getattr_l__mod___down4_maxpool_conv___1___double_conv_4 = self.getattr_L__mod___down4_maxpool_conv___1___double_conv_4(getattr_l__mod___down4_maxpool_conv___1___double_conv_3);  getattr_l__mod___down4_maxpool_conv___1___double_conv_3 = None
    x5 = self.getattr_L__mod___down4_maxpool_conv___1___double_conv_5(getattr_l__mod___down4_maxpool_conv___1___double_conv_4);  getattr_l__mod___down4_maxpool_conv___1___double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    x1_1 = self.L__mod___up1_up(x5);  x5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    x1_2 = torch.nn.functional.pad(x1_1, [0, 1, 0, 0]);  x1_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    x = torch.cat([x4, x1_2], dim = 1);  x4 = x1_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    l__mod___up1_conv_double_conv_0 = self.L__mod___up1_conv_double_conv_0(x);  x = None
    l__mod___up1_conv_double_conv_1 = self.L__mod___up1_conv_double_conv_1(l__mod___up1_conv_double_conv_0);  l__mod___up1_conv_double_conv_0 = None
    l__mod___up1_conv_double_conv_2 = self.L__mod___up1_conv_double_conv_2(l__mod___up1_conv_double_conv_1);  l__mod___up1_conv_double_conv_1 = None
    l__mod___up1_conv_double_conv_3 = self.L__mod___up1_conv_double_conv_3(l__mod___up1_conv_double_conv_2);  l__mod___up1_conv_double_conv_2 = None
    l__mod___up1_conv_double_conv_4 = self.L__mod___up1_conv_double_conv_4(l__mod___up1_conv_double_conv_3);  l__mod___up1_conv_double_conv_3 = None
    x_1 = self.L__mod___up1_conv_double_conv_5(l__mod___up1_conv_double_conv_4);  l__mod___up1_conv_double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    x1_3 = self.L__mod___up2_up(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    x1_4 = torch.nn.functional.pad(x1_3, [0, 1, 0, 0]);  x1_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    x_2 = torch.cat([x3, x1_4], dim = 1);  x3 = x1_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    l__mod___up2_conv_double_conv_0 = self.L__mod___up2_conv_double_conv_0(x_2);  x_2 = None
    l__mod___up2_conv_double_conv_1 = self.L__mod___up2_conv_double_conv_1(l__mod___up2_conv_double_conv_0);  l__mod___up2_conv_double_conv_0 = None
    l__mod___up2_conv_double_conv_2 = self.L__mod___up2_conv_double_conv_2(l__mod___up2_conv_double_conv_1);  l__mod___up2_conv_double_conv_1 = None
    l__mod___up2_conv_double_conv_3 = self.L__mod___up2_conv_double_conv_3(l__mod___up2_conv_double_conv_2);  l__mod___up2_conv_double_conv_2 = None
    l__mod___up2_conv_double_conv_4 = self.L__mod___up2_conv_double_conv_4(l__mod___up2_conv_double_conv_3);  l__mod___up2_conv_double_conv_3 = None
    x_3 = self.L__mod___up2_conv_double_conv_5(l__mod___up2_conv_double_conv_4);  l__mod___up2_conv_double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    x1_5 = self.L__mod___up3_up(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    x1_6 = torch.nn.functional.pad(x1_5, [0, 1, 0, 0]);  x1_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    x_4 = torch.cat([x2, x1_6], dim = 1);  x2 = x1_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    l__mod___up3_conv_double_conv_0 = self.L__mod___up3_conv_double_conv_0(x_4);  x_4 = None
    l__mod___up3_conv_double_conv_1 = self.L__mod___up3_conv_double_conv_1(l__mod___up3_conv_double_conv_0);  l__mod___up3_conv_double_conv_0 = None
    l__mod___up3_conv_double_conv_2 = self.L__mod___up3_conv_double_conv_2(l__mod___up3_conv_double_conv_1);  l__mod___up3_conv_double_conv_1 = None
    l__mod___up3_conv_double_conv_3 = self.L__mod___up3_conv_double_conv_3(l__mod___up3_conv_double_conv_2);  l__mod___up3_conv_double_conv_2 = None
    l__mod___up3_conv_double_conv_4 = self.L__mod___up3_conv_double_conv_4(l__mod___up3_conv_double_conv_3);  l__mod___up3_conv_double_conv_3 = None
    x_5 = self.L__mod___up3_conv_double_conv_5(l__mod___up3_conv_double_conv_4);  l__mod___up3_conv_double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:57, code: x1 = self.up(x1)
    x1_7 = self.L__mod___up4_up(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:62, code: x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    x1_8 = torch.nn.functional.pad(x1_7, [0, 1, 0, 0]);  x1_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67, code: x = torch.cat([x2, x1], dim=1)
    x_6 = torch.cat([x1, x1_8], dim = 1);  x1 = x1_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25, code: return self.double_conv(x)
    l__mod___up4_conv_double_conv_0 = self.L__mod___up4_conv_double_conv_0(x_6);  x_6 = None
    l__mod___up4_conv_double_conv_1 = self.L__mod___up4_conv_double_conv_1(l__mod___up4_conv_double_conv_0);  l__mod___up4_conv_double_conv_0 = None
    l__mod___up4_conv_double_conv_2 = self.L__mod___up4_conv_double_conv_2(l__mod___up4_conv_double_conv_1);  l__mod___up4_conv_double_conv_1 = None
    l__mod___up4_conv_double_conv_3 = self.L__mod___up4_conv_double_conv_3(l__mod___up4_conv_double_conv_2);  l__mod___up4_conv_double_conv_2 = None
    l__mod___up4_conv_double_conv_4 = self.L__mod___up4_conv_double_conv_4(l__mod___up4_conv_double_conv_3);  l__mod___up4_conv_double_conv_3 = None
    x_7 = self.L__mod___up4_conv_double_conv_5(l__mod___up4_conv_double_conv_4);  l__mod___up4_conv_double_conv_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:77, code: return self.conv(x)
    logits = self.L__mod___outc_conv(x_7);  x_7 = None
    return (logits,)
    