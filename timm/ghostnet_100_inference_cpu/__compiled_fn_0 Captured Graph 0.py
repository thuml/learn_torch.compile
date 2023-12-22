from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:282, code: x = self.conv_stem(x)
    x = self.L__mod___conv_stem(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:283, code: x = self.bn1(x)
    x_1 = self.L__mod___bn1(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:284, code: x = self.act1(x)
    shortcut = self.L__mod___act1(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___0_____0___ghost1_primary_conv_0(shortcut)
    getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0 = None
    x1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___0_____0___ghost1_cheap_operation_0(x1)
    getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0 = None
    x2 = self.getattr_getattr_L__mod___blocks___0_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out = torch.cat([x1, x2], dim = 1);  x1 = x2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_3 = out[(slice(None, None, None), slice(None, 16, None), slice(None, None, None), slice(None, None, None))];  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___0_____0___ghost2_primary_conv_0(x_3);  x_3 = None
    getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0 = None
    x1_1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___0_____0___ghost2_cheap_operation_0(x1_1)
    getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0 = None
    x2_1 = self.getattr_getattr_L__mod___blocks___0_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_1 = torch.cat([x1_1, x2_1], dim = 1);  x1_1 = x2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_4 = out_1[(slice(None, None, None), slice(None, 16, None), slice(None, None, None), slice(None, None, None))];  out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_4 += shortcut;  shortcut_1 = x_4;  x_4 = shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___1_____0___ghost1_primary_conv_0(shortcut_1)
    getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___1_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0 = None
    x1_2 = self.getattr_getattr_L__mod___blocks___1_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___1_____0___ghost1_cheap_operation_0(x1_2)
    getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___1_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0 = None
    x2_2 = self.getattr_getattr_L__mod___blocks___1_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_2 = torch.cat([x1_2, x2_2], dim = 1);  x1_2 = x2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_6 = out_2[(slice(None, None, None), slice(None, 48, None), slice(None, None, None), slice(None, None, None))];  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    x_7 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    x_8 = self.getattr_getattr_L__mod___blocks___1_____0___bn_dw(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___1_____0___ghost2_primary_conv_0(x_8);  x_8 = None
    getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___1_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0 = None
    x1_3 = self.getattr_getattr_L__mod___blocks___1_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___1_____0___ghost2_cheap_operation_0(x1_3)
    getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___1_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0 = None
    x2_3 = self.getattr_getattr_L__mod___blocks___1_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_3 = torch.cat([x1_3, x2_3], dim = 1);  x1_3 = x2_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_9 = out_3[(slice(None, None, None), slice(None, 24, None), slice(None, None, None), slice(None, None, None))];  out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    getattr_getattr_l__mod___blocks___1_____0___shortcut_0 = self.getattr_getattr_L__mod___blocks___1_____0___shortcut_0(shortcut_1);  shortcut_1 = None
    getattr_getattr_l__mod___blocks___1_____0___shortcut_1 = self.getattr_getattr_L__mod___blocks___1_____0___shortcut_1(getattr_getattr_l__mod___blocks___1_____0___shortcut_0);  getattr_getattr_l__mod___blocks___1_____0___shortcut_0 = None
    getattr_getattr_l__mod___blocks___1_____0___shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___shortcut_2(getattr_getattr_l__mod___blocks___1_____0___shortcut_1);  getattr_getattr_l__mod___blocks___1_____0___shortcut_1 = None
    getattr_getattr_l__mod___blocks___1_____0___shortcut_3 = self.getattr_getattr_L__mod___blocks___1_____0___shortcut_3(getattr_getattr_l__mod___blocks___1_____0___shortcut_2);  getattr_getattr_l__mod___blocks___1_____0___shortcut_2 = None
    x_9 += getattr_getattr_l__mod___blocks___1_____0___shortcut_3;  shortcut_2 = x_9;  x_9 = getattr_getattr_l__mod___blocks___1_____0___shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___2_____0___ghost1_primary_conv_0(shortcut_2)
    getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___2_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0 = None
    x1_4 = self.getattr_getattr_L__mod___blocks___2_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___2_____0___ghost1_cheap_operation_0(x1_4)
    getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___2_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0 = None
    x2_4 = self.getattr_getattr_L__mod___blocks___2_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_4 = torch.cat([x1_4, x2_4], dim = 1);  x1_4 = x2_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_11 = out_4[(slice(None, None, None), slice(None, 72, None), slice(None, None, None), slice(None, None, None))];  out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___2_____0___ghost2_primary_conv_0(x_11);  x_11 = None
    getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___2_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0 = None
    x1_5 = self.getattr_getattr_L__mod___blocks___2_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___2_____0___ghost2_cheap_operation_0(x1_5)
    getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___2_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0 = None
    x2_5 = self.getattr_getattr_L__mod___blocks___2_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_5 = torch.cat([x1_5, x2_5], dim = 1);  x1_5 = x2_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_12 = out_5[(slice(None, None, None), slice(None, 24, None), slice(None, None, None), slice(None, None, None))];  out_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_12 += shortcut_2;  shortcut_3 = x_12;  x_12 = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___3_____0___ghost1_primary_conv_0(shortcut_3)
    getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___3_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0 = None
    x1_6 = self.getattr_getattr_L__mod___blocks___3_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___3_____0___ghost1_cheap_operation_0(x1_6)
    getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___3_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0 = None
    x2_6 = self.getattr_getattr_L__mod___blocks___3_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_6 = torch.cat([x1_6, x2_6], dim = 1);  x1_6 = x2_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_14 = out_6[(slice(None, None, None), slice(None, 72, None), slice(None, None, None), slice(None, None, None))];  out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    x_15 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    x_16 = self.getattr_getattr_L__mod___blocks___3_____0___bn_dw(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_16.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_1 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_reduce(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_2 = self.getattr_getattr_L__mod___blocks___3_____0___se_act1(x_se_1);  x_se_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_3 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_expand(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____0___se_gate = self.getattr_getattr_L__mod___blocks___3_____0___se_gate(x_se_3);  x_se_3 = None
    x_17 = x_16 * getattr_getattr_l__mod___blocks___3_____0___se_gate;  x_16 = getattr_getattr_l__mod___blocks___3_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___3_____0___ghost2_primary_conv_0(x_17);  x_17 = None
    getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___3_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0 = None
    x1_7 = self.getattr_getattr_L__mod___blocks___3_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___3_____0___ghost2_cheap_operation_0(x1_7)
    getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___3_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0 = None
    x2_7 = self.getattr_getattr_L__mod___blocks___3_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_7 = torch.cat([x1_7, x2_7], dim = 1);  x1_7 = x2_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_18 = out_7[(slice(None, None, None), slice(None, 40, None), slice(None, None, None), slice(None, None, None))];  out_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    getattr_getattr_l__mod___blocks___3_____0___shortcut_0 = self.getattr_getattr_L__mod___blocks___3_____0___shortcut_0(shortcut_3);  shortcut_3 = None
    getattr_getattr_l__mod___blocks___3_____0___shortcut_1 = self.getattr_getattr_L__mod___blocks___3_____0___shortcut_1(getattr_getattr_l__mod___blocks___3_____0___shortcut_0);  getattr_getattr_l__mod___blocks___3_____0___shortcut_0 = None
    getattr_getattr_l__mod___blocks___3_____0___shortcut_2 = self.getattr_getattr_L__mod___blocks___3_____0___shortcut_2(getattr_getattr_l__mod___blocks___3_____0___shortcut_1);  getattr_getattr_l__mod___blocks___3_____0___shortcut_1 = None
    getattr_getattr_l__mod___blocks___3_____0___shortcut_3 = self.getattr_getattr_L__mod___blocks___3_____0___shortcut_3(getattr_getattr_l__mod___blocks___3_____0___shortcut_2);  getattr_getattr_l__mod___blocks___3_____0___shortcut_2 = None
    x_18 += getattr_getattr_l__mod___blocks___3_____0___shortcut_3;  shortcut_4 = x_18;  x_18 = getattr_getattr_l__mod___blocks___3_____0___shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___4_____0___ghost1_primary_conv_0(shortcut_4)
    getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___4_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0 = None
    x1_8 = self.getattr_getattr_L__mod___blocks___4_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___4_____0___ghost1_cheap_operation_0(x1_8)
    getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___4_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0 = None
    x2_8 = self.getattr_getattr_L__mod___blocks___4_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_8 = torch.cat([x1_8, x2_8], dim = 1);  x1_8 = x2_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_20 = out_8[(slice(None, None, None), slice(None, 120, None), slice(None, None, None), slice(None, None, None))];  out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_20.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_5 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_reduce(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_6 = self.getattr_getattr_L__mod___blocks___4_____0___se_act1(x_se_5);  x_se_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_7 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_expand(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____0___se_gate = self.getattr_getattr_L__mod___blocks___4_____0___se_gate(x_se_7);  x_se_7 = None
    x_21 = x_20 * getattr_getattr_l__mod___blocks___4_____0___se_gate;  x_20 = getattr_getattr_l__mod___blocks___4_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___4_____0___ghost2_primary_conv_0(x_21);  x_21 = None
    getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___4_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0 = None
    x1_9 = self.getattr_getattr_L__mod___blocks___4_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___4_____0___ghost2_cheap_operation_0(x1_9)
    getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___4_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0 = None
    x2_9 = self.getattr_getattr_L__mod___blocks___4_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_9 = torch.cat([x1_9, x2_9], dim = 1);  x1_9 = x2_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_22 = out_9[(slice(None, None, None), slice(None, 40, None), slice(None, None, None), slice(None, None, None))];  out_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_22 += shortcut_4;  shortcut_5 = x_22;  x_22 = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___5_____0___ghost1_primary_conv_0(shortcut_5)
    getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___5_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0 = None
    x1_10 = self.getattr_getattr_L__mod___blocks___5_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___5_____0___ghost1_cheap_operation_0(x1_10)
    getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___5_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0 = None
    x2_10 = self.getattr_getattr_L__mod___blocks___5_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_10 = torch.cat([x1_10, x2_10], dim = 1);  x1_10 = x2_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_24 = out_10[(slice(None, None, None), slice(None, 240, None), slice(None, None, None), slice(None, None, None))];  out_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    x_25 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    x_26 = self.getattr_getattr_L__mod___blocks___5_____0___bn_dw(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___5_____0___ghost2_primary_conv_0(x_26);  x_26 = None
    getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___5_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0 = None
    x1_11 = self.getattr_getattr_L__mod___blocks___5_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___5_____0___ghost2_cheap_operation_0(x1_11)
    getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___5_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0 = None
    x2_11 = self.getattr_getattr_L__mod___blocks___5_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_11 = torch.cat([x1_11, x2_11], dim = 1);  x1_11 = x2_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_27 = out_11[(slice(None, None, None), slice(None, 80, None), slice(None, None, None), slice(None, None, None))];  out_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    getattr_getattr_l__mod___blocks___5_____0___shortcut_0 = self.getattr_getattr_L__mod___blocks___5_____0___shortcut_0(shortcut_5);  shortcut_5 = None
    getattr_getattr_l__mod___blocks___5_____0___shortcut_1 = self.getattr_getattr_L__mod___blocks___5_____0___shortcut_1(getattr_getattr_l__mod___blocks___5_____0___shortcut_0);  getattr_getattr_l__mod___blocks___5_____0___shortcut_0 = None
    getattr_getattr_l__mod___blocks___5_____0___shortcut_2 = self.getattr_getattr_L__mod___blocks___5_____0___shortcut_2(getattr_getattr_l__mod___blocks___5_____0___shortcut_1);  getattr_getattr_l__mod___blocks___5_____0___shortcut_1 = None
    getattr_getattr_l__mod___blocks___5_____0___shortcut_3 = self.getattr_getattr_L__mod___blocks___5_____0___shortcut_3(getattr_getattr_l__mod___blocks___5_____0___shortcut_2);  getattr_getattr_l__mod___blocks___5_____0___shortcut_2 = None
    x_27 += getattr_getattr_l__mod___blocks___5_____0___shortcut_3;  shortcut_6 = x_27;  x_27 = getattr_getattr_l__mod___blocks___5_____0___shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____0___ghost1_primary_conv_0(shortcut_6)
    getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0 = None
    x1_12 = self.getattr_getattr_L__mod___blocks___6_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____0___ghost1_cheap_operation_0(x1_12)
    getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0 = None
    x2_12 = self.getattr_getattr_L__mod___blocks___6_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_12 = torch.cat([x1_12, x2_12], dim = 1);  x1_12 = x2_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_29 = out_12[(slice(None, None, None), slice(None, 200, None), slice(None, None, None), slice(None, None, None))];  out_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____0___ghost2_primary_conv_0(x_29);  x_29 = None
    getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0 = None
    x1_13 = self.getattr_getattr_L__mod___blocks___6_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____0___ghost2_cheap_operation_0(x1_13)
    getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0 = None
    x2_13 = self.getattr_getattr_L__mod___blocks___6_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_13 = torch.cat([x1_13, x2_13], dim = 1);  x1_13 = x2_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_30 = out_13[(slice(None, None, None), slice(None, 80, None), slice(None, None, None), slice(None, None, None))];  out_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_30 += shortcut_6;  shortcut_7 = x_30;  x_30 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____1___ghost1_primary_conv_0(shortcut_7)
    getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____1___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0 = None
    x1_14 = self.getattr_getattr_L__mod___blocks___6_____1___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____1___ghost1_cheap_operation_0(x1_14)
    getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____1___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0 = None
    x2_14 = self.getattr_getattr_L__mod___blocks___6_____1___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_14 = torch.cat([x1_14, x2_14], dim = 1);  x1_14 = x2_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_32 = out_14[(slice(None, None, None), slice(None, 184, None), slice(None, None, None), slice(None, None, None))];  out_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____1___ghost2_primary_conv_0(x_32);  x_32 = None
    getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____1___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0 = None
    x1_15 = self.getattr_getattr_L__mod___blocks___6_____1___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____1___ghost2_cheap_operation_0(x1_15)
    getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____1___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0 = None
    x2_15 = self.getattr_getattr_L__mod___blocks___6_____1___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_15 = torch.cat([x1_15, x2_15], dim = 1);  x1_15 = x2_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_33 = out_15[(slice(None, None, None), slice(None, 80, None), slice(None, None, None), slice(None, None, None))];  out_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_33 += shortcut_7;  shortcut_8 = x_33;  x_33 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____2___ghost1_primary_conv_0(shortcut_8)
    getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____2___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0 = None
    x1_16 = self.getattr_getattr_L__mod___blocks___6_____2___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____2___ghost1_cheap_operation_0(x1_16)
    getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____2___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0 = None
    x2_16 = self.getattr_getattr_L__mod___blocks___6_____2___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_16 = torch.cat([x1_16, x2_16], dim = 1);  x1_16 = x2_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_35 = out_16[(slice(None, None, None), slice(None, 184, None), slice(None, None, None), slice(None, None, None))];  out_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____2___ghost2_primary_conv_0(x_35);  x_35 = None
    getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____2___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0 = None
    x1_17 = self.getattr_getattr_L__mod___blocks___6_____2___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____2___ghost2_cheap_operation_0(x1_17)
    getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____2___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0 = None
    x2_17 = self.getattr_getattr_L__mod___blocks___6_____2___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_17 = torch.cat([x1_17, x2_17], dim = 1);  x1_17 = x2_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_36 = out_17[(slice(None, None, None), slice(None, 80, None), slice(None, None, None), slice(None, None, None))];  out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_36 += shortcut_8;  shortcut_9 = x_36;  x_36 = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____3___ghost1_primary_conv_0(shortcut_9)
    getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____3___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0 = None
    x1_18 = self.getattr_getattr_L__mod___blocks___6_____3___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____3___ghost1_cheap_operation_0(x1_18)
    getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____3___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0 = None
    x2_18 = self.getattr_getattr_L__mod___blocks___6_____3___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_18 = torch.cat([x1_18, x2_18], dim = 1);  x1_18 = x2_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_38 = out_18[(slice(None, None, None), slice(None, 480, None), slice(None, None, None), slice(None, None, None))];  out_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_38.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_9 = self.getattr_getattr_L__mod___blocks___6_____3___se_conv_reduce(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_10 = self.getattr_getattr_L__mod___blocks___6_____3___se_act1(x_se_9);  x_se_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_11 = self.getattr_getattr_L__mod___blocks___6_____3___se_conv_expand(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___6_____3___se_gate = self.getattr_getattr_L__mod___blocks___6_____3___se_gate(x_se_11);  x_se_11 = None
    x_39 = x_38 * getattr_getattr_l__mod___blocks___6_____3___se_gate;  x_38 = getattr_getattr_l__mod___blocks___6_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____3___ghost2_primary_conv_0(x_39);  x_39 = None
    getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____3___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0 = None
    x1_19 = self.getattr_getattr_L__mod___blocks___6_____3___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____3___ghost2_cheap_operation_0(x1_19)
    getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____3___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0 = None
    x2_19 = self.getattr_getattr_L__mod___blocks___6_____3___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_19 = torch.cat([x1_19, x2_19], dim = 1);  x1_19 = x2_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_40 = out_19[(slice(None, None, None), slice(None, 112, None), slice(None, None, None), slice(None, None, None))];  out_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    getattr_getattr_l__mod___blocks___6_____3___shortcut_0 = self.getattr_getattr_L__mod___blocks___6_____3___shortcut_0(shortcut_9);  shortcut_9 = None
    getattr_getattr_l__mod___blocks___6_____3___shortcut_1 = self.getattr_getattr_L__mod___blocks___6_____3___shortcut_1(getattr_getattr_l__mod___blocks___6_____3___shortcut_0);  getattr_getattr_l__mod___blocks___6_____3___shortcut_0 = None
    getattr_getattr_l__mod___blocks___6_____3___shortcut_2 = self.getattr_getattr_L__mod___blocks___6_____3___shortcut_2(getattr_getattr_l__mod___blocks___6_____3___shortcut_1);  getattr_getattr_l__mod___blocks___6_____3___shortcut_1 = None
    getattr_getattr_l__mod___blocks___6_____3___shortcut_3 = self.getattr_getattr_L__mod___blocks___6_____3___shortcut_3(getattr_getattr_l__mod___blocks___6_____3___shortcut_2);  getattr_getattr_l__mod___blocks___6_____3___shortcut_2 = None
    x_40 += getattr_getattr_l__mod___blocks___6_____3___shortcut_3;  shortcut_10 = x_40;  x_40 = getattr_getattr_l__mod___blocks___6_____3___shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____4___ghost1_primary_conv_0(shortcut_10)
    getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____4___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0 = None
    x1_20 = self.getattr_getattr_L__mod___blocks___6_____4___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____4___ghost1_cheap_operation_0(x1_20)
    getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____4___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0 = None
    x2_20 = self.getattr_getattr_L__mod___blocks___6_____4___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_20 = torch.cat([x1_20, x2_20], dim = 1);  x1_20 = x2_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_42 = out_20[(slice(None, None, None), slice(None, 672, None), slice(None, None, None), slice(None, None, None))];  out_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_42.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_13 = self.getattr_getattr_L__mod___blocks___6_____4___se_conv_reduce(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_14 = self.getattr_getattr_L__mod___blocks___6_____4___se_act1(x_se_13);  x_se_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_15 = self.getattr_getattr_L__mod___blocks___6_____4___se_conv_expand(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___6_____4___se_gate = self.getattr_getattr_L__mod___blocks___6_____4___se_gate(x_se_15);  x_se_15 = None
    x_43 = x_42 * getattr_getattr_l__mod___blocks___6_____4___se_gate;  x_42 = getattr_getattr_l__mod___blocks___6_____4___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___6_____4___ghost2_primary_conv_0(x_43);  x_43 = None
    getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___6_____4___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0 = None
    x1_21 = self.getattr_getattr_L__mod___blocks___6_____4___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___6_____4___ghost2_cheap_operation_0(x1_21)
    getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___6_____4___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0 = None
    x2_21 = self.getattr_getattr_L__mod___blocks___6_____4___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_21 = torch.cat([x1_21, x2_21], dim = 1);  x1_21 = x2_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_44 = out_21[(slice(None, None, None), slice(None, 112, None), slice(None, None, None), slice(None, None, None))];  out_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_44 += shortcut_10;  shortcut_11 = x_44;  x_44 = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___7_____0___ghost1_primary_conv_0(shortcut_11)
    getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___7_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0 = None
    x1_22 = self.getattr_getattr_L__mod___blocks___7_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___7_____0___ghost1_cheap_operation_0(x1_22)
    getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___7_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0 = None
    x2_22 = self.getattr_getattr_L__mod___blocks___7_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_22 = torch.cat([x1_22, x2_22], dim = 1);  x1_22 = x2_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_46 = out_22[(slice(None, None, None), slice(None, 672, None), slice(None, None, None), slice(None, None, None))];  out_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    x_47 = self.getattr_getattr_L__mod___blocks___7_____0___conv_dw(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    x_48 = self.getattr_getattr_L__mod___blocks___7_____0___bn_dw(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_48.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_17 = self.getattr_getattr_L__mod___blocks___7_____0___se_conv_reduce(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_18 = self.getattr_getattr_L__mod___blocks___7_____0___se_act1(x_se_17);  x_se_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_19 = self.getattr_getattr_L__mod___blocks___7_____0___se_conv_expand(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___7_____0___se_gate = self.getattr_getattr_L__mod___blocks___7_____0___se_gate(x_se_19);  x_se_19 = None
    x_49 = x_48 * getattr_getattr_l__mod___blocks___7_____0___se_gate;  x_48 = getattr_getattr_l__mod___blocks___7_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___7_____0___ghost2_primary_conv_0(x_49);  x_49 = None
    getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___7_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0 = None
    x1_23 = self.getattr_getattr_L__mod___blocks___7_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___7_____0___ghost2_cheap_operation_0(x1_23)
    getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___7_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0 = None
    x2_23 = self.getattr_getattr_L__mod___blocks___7_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_23 = torch.cat([x1_23, x2_23], dim = 1);  x1_23 = x2_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_50 = out_23[(slice(None, None, None), slice(None, 160, None), slice(None, None, None), slice(None, None, None))];  out_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    getattr_getattr_l__mod___blocks___7_____0___shortcut_0 = self.getattr_getattr_L__mod___blocks___7_____0___shortcut_0(shortcut_11);  shortcut_11 = None
    getattr_getattr_l__mod___blocks___7_____0___shortcut_1 = self.getattr_getattr_L__mod___blocks___7_____0___shortcut_1(getattr_getattr_l__mod___blocks___7_____0___shortcut_0);  getattr_getattr_l__mod___blocks___7_____0___shortcut_0 = None
    getattr_getattr_l__mod___blocks___7_____0___shortcut_2 = self.getattr_getattr_L__mod___blocks___7_____0___shortcut_2(getattr_getattr_l__mod___blocks___7_____0___shortcut_1);  getattr_getattr_l__mod___blocks___7_____0___shortcut_1 = None
    getattr_getattr_l__mod___blocks___7_____0___shortcut_3 = self.getattr_getattr_L__mod___blocks___7_____0___shortcut_3(getattr_getattr_l__mod___blocks___7_____0___shortcut_2);  getattr_getattr_l__mod___blocks___7_____0___shortcut_2 = None
    x_50 += getattr_getattr_l__mod___blocks___7_____0___shortcut_3;  shortcut_12 = x_50;  x_50 = getattr_getattr_l__mod___blocks___7_____0___shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____0___ghost1_primary_conv_0(shortcut_12)
    getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____0___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0 = None
    x1_24 = self.getattr_getattr_L__mod___blocks___8_____0___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____0___ghost1_cheap_operation_0(x1_24)
    getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____0___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0 = None
    x2_24 = self.getattr_getattr_L__mod___blocks___8_____0___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_24 = torch.cat([x1_24, x2_24], dim = 1);  x1_24 = x2_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_52 = out_24[(slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None))];  out_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____0___ghost2_primary_conv_0(x_52);  x_52 = None
    getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____0___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0 = None
    x1_25 = self.getattr_getattr_L__mod___blocks___8_____0___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____0___ghost2_cheap_operation_0(x1_25)
    getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____0___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0 = None
    x2_25 = self.getattr_getattr_L__mod___blocks___8_____0___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_25 = torch.cat([x1_25, x2_25], dim = 1);  x1_25 = x2_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_53 = out_25[(slice(None, None, None), slice(None, 160, None), slice(None, None, None), slice(None, None, None))];  out_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_53 += shortcut_12;  shortcut_13 = x_53;  x_53 = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____1___ghost1_primary_conv_0(shortcut_13)
    getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____1___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0 = None
    x1_26 = self.getattr_getattr_L__mod___blocks___8_____1___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____1___ghost1_cheap_operation_0(x1_26)
    getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____1___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0 = None
    x2_26 = self.getattr_getattr_L__mod___blocks___8_____1___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_26 = torch.cat([x1_26, x2_26], dim = 1);  x1_26 = x2_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_55 = out_26[(slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None))];  out_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_55.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_21 = self.getattr_getattr_L__mod___blocks___8_____1___se_conv_reduce(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_22 = self.getattr_getattr_L__mod___blocks___8_____1___se_act1(x_se_21);  x_se_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_23 = self.getattr_getattr_L__mod___blocks___8_____1___se_conv_expand(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___8_____1___se_gate = self.getattr_getattr_L__mod___blocks___8_____1___se_gate(x_se_23);  x_se_23 = None
    x_56 = x_55 * getattr_getattr_l__mod___blocks___8_____1___se_gate;  x_55 = getattr_getattr_l__mod___blocks___8_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____1___ghost2_primary_conv_0(x_56);  x_56 = None
    getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____1___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0 = None
    x1_27 = self.getattr_getattr_L__mod___blocks___8_____1___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____1___ghost2_cheap_operation_0(x1_27)
    getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____1___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0 = None
    x2_27 = self.getattr_getattr_L__mod___blocks___8_____1___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_27 = torch.cat([x1_27, x2_27], dim = 1);  x1_27 = x2_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_57 = out_27[(slice(None, None, None), slice(None, 160, None), slice(None, None, None), slice(None, None, None))];  out_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_57 += shortcut_13;  shortcut_14 = x_57;  x_57 = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____2___ghost1_primary_conv_0(shortcut_14)
    getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____2___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0 = None
    x1_28 = self.getattr_getattr_L__mod___blocks___8_____2___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____2___ghost1_cheap_operation_0(x1_28)
    getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____2___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0 = None
    x2_28 = self.getattr_getattr_L__mod___blocks___8_____2___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_28 = torch.cat([x1_28, x2_28], dim = 1);  x1_28 = x2_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_59 = out_28[(slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None))];  out_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____2___ghost2_primary_conv_0(x_59);  x_59 = None
    getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____2___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0 = None
    x1_29 = self.getattr_getattr_L__mod___blocks___8_____2___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____2___ghost2_cheap_operation_0(x1_29)
    getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____2___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0 = None
    x2_29 = self.getattr_getattr_L__mod___blocks___8_____2___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_29 = torch.cat([x1_29, x2_29], dim = 1);  x1_29 = x2_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_60 = out_29[(slice(None, None, None), slice(None, 160, None), slice(None, None, None), slice(None, None, None))];  out_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_60 += shortcut_14;  shortcut_15 = x_60;  x_60 = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____3___ghost1_primary_conv_0(shortcut_15)
    getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____3___ghost1_primary_conv_1(getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0 = None
    x1_30 = self.getattr_getattr_L__mod___blocks___8_____3___ghost1_primary_conv_2(getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____3___ghost1_cheap_operation_0(x1_30)
    getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____3___ghost1_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0 = None
    x2_30 = self.getattr_getattr_L__mod___blocks___8_____3___ghost1_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_30 = torch.cat([x1_30, x2_30], dim = 1);  x1_30 = x2_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_62 = out_30[(slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None))];  out_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_62.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_25 = self.getattr_getattr_L__mod___blocks___8_____3___se_conv_reduce(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_26 = self.getattr_getattr_L__mod___blocks___8_____3___se_act1(x_se_25);  x_se_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_27 = self.getattr_getattr_L__mod___blocks___8_____3___se_conv_expand(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___8_____3___se_gate = self.getattr_getattr_L__mod___blocks___8_____3___se_gate(x_se_27);  x_se_27 = None
    x_63 = x_62 * getattr_getattr_l__mod___blocks___8_____3___se_gate;  x_62 = getattr_getattr_l__mod___blocks___8_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0 = self.getattr_getattr_L__mod___blocks___8_____3___ghost2_primary_conv_0(x_63);  x_63 = None
    getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1 = self.getattr_getattr_L__mod___blocks___8_____3___ghost2_primary_conv_1(getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0);  getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0 = None
    x1_31 = self.getattr_getattr_L__mod___blocks___8_____3___ghost2_primary_conv_2(getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1);  getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0 = self.getattr_getattr_L__mod___blocks___8_____3___ghost2_cheap_operation_0(x1_31)
    getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_1 = self.getattr_getattr_L__mod___blocks___8_____3___ghost2_cheap_operation_1(getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0);  getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0 = None
    x2_31 = self.getattr_getattr_L__mod___blocks___8_____3___ghost2_cheap_operation_2(getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_1);  getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    out_31 = torch.cat([x1_31, x2_31], dim = 1);  x1_31 = x2_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    x_64 = out_31[(slice(None, None, None), slice(None, 160, None), slice(None, None, None), slice(None, None, None))];  out_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    x_64 += shortcut_15;  shortcut_16 = x_64;  x_64 = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    x_66 = self.getattr_getattr_L__mod___blocks___9_____0___conv(shortcut_16);  shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___9_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___9_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___9_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___9_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___9_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___9_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___9_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___9_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_67 = torch.nn.functional.batch_norm(x_66, getattr_getattr_l__mod___blocks___9_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___9_____0___bn1_running_var, getattr_getattr_l__mod___blocks___9_____0___bn1_weight, getattr_getattr_l__mod___blocks___9_____0___bn1_bias, False, 0.1, 1e-05);  x_66 = getattr_getattr_l__mod___blocks___9_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___9_____0___bn1_running_var = getattr_getattr_l__mod___blocks___9_____0___bn1_weight = getattr_getattr_l__mod___blocks___9_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_68 = self.getattr_getattr_L__mod___blocks___9_____0___bn1_drop(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_72 = self.getattr_getattr_L__mod___blocks___9_____0___bn1_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_73 = self.L__mod___global_pool_pool(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_75 = self.L__mod___global_pool_flatten(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:293, code: x = self.conv_head(x)
    x_76 = self.L__mod___conv_head(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:294, code: x = self.act2(x)
    x_77 = self.L__mod___act2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:295, code: x = self.flatten(x)
    x_78 = self.L__mod___flatten(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    l__mod___classifier_weight = self.L__mod___classifier_weight
    l__mod___classifier_bias = self.L__mod___classifier_bias
    x_80 = torch._C._nn.linear(x_78, l__mod___classifier_weight, l__mod___classifier_bias);  x_78 = l__mod___classifier_weight = l__mod___classifier_bias = None
    return (x_80,)
    