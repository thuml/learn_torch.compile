from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    l__mod___conv1_0 = self.L__mod___conv1_0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___conv1_1 = self.L__mod___conv1_1(l__mod___conv1_0);  l__mod___conv1_0 = None
    l__mod___conv1_2 = self.L__mod___conv1_2(l__mod___conv1_1);  l__mod___conv1_1 = None
    l__mod___conv1_3 = self.L__mod___conv1_3(l__mod___conv1_2);  l__mod___conv1_2 = None
    l__mod___conv1_4 = self.L__mod___conv1_4(l__mod___conv1_3);  l__mod___conv1_3 = None
    l__mod___conv1_5 = self.L__mod___conv1_5(l__mod___conv1_4);  l__mod___conv1_4 = None
    x = self.L__mod___conv1_6(l__mod___conv1_5);  l__mod___conv1_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    x_1 = self.L__mod___bn1(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    x_2 = self.L__mod___act1(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    shortcut = self.L__mod___maxpool(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out = self.getattr_L__mod___layer1___0___conv1(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_1 = self.getattr_L__mod___layer1___0___bn1(out);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_2 = self.getattr_L__mod___layer1___0___act1(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_4 = self.getattr_L__mod___layer1___0___conv2_conv(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_5 = self.getattr_L__mod___layer1___0___conv2_bn0(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_6 = self.getattr_L__mod___layer1___0___conv2_drop(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_7 = self.getattr_L__mod___layer1___0___conv2_act0(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_8 = x_7.reshape((8, 2, 64, 64, 64));  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap = x_8.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_1 = x_gap.mean((2, 3), keepdim = True);  x_gap = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_2 = self.getattr_L__mod___layer1___0___conv2_fc1(x_gap_1);  x_gap_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_3 = self.getattr_L__mod___layer1___0___conv2_bn1(x_gap_2);  x_gap_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_4 = self.getattr_L__mod___layer1___0___conv2_act1(x_gap_3);  x_gap_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn = self.getattr_L__mod___layer1___0___conv2_fc2(x_gap_4);  x_gap_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view = x_attn.view(8, 1, 2, -1);  x_attn = None
    x_9 = view.transpose(1, 2);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_10 = torch.nn.functional.softmax(x_9, dim = 1);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_11 = x_10.reshape(8, -1);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_1 = x_11.view(8, -1, 1, 1);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_2 = x_attn_1.reshape((8, 2, 64, 1, 1));  x_attn_1 = None
    mul = x_8 * reshape_2;  x_8 = reshape_2 = None
    out_3 = mul.sum(dim = 1);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_4 = out_3.contiguous();  out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_5 = self.getattr_L__mod___layer1___0___bn2(out_4);  out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_6 = self.getattr_L__mod___layer1___0___drop_block(out_5);  out_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_7 = self.getattr_L__mod___layer1___0___act2(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_8 = self.getattr_L__mod___layer1___0___conv3(out_7);  out_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_9 = self.getattr_L__mod___layer1___0___bn3(out_8);  out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer1___0___downsample_0 = self.getattr_L__mod___layer1___0___downsample_0(shortcut);  shortcut = None
    getattr_l__mod___layer1___0___downsample_1 = self.getattr_L__mod___layer1___0___downsample_1(getattr_l__mod___layer1___0___downsample_0);  getattr_l__mod___layer1___0___downsample_0 = None
    shortcut_1 = self.getattr_L__mod___layer1___0___downsample_2(getattr_l__mod___layer1___0___downsample_1);  getattr_l__mod___layer1___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_9 += shortcut_1;  out_10 = out_9;  out_9 = shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_2 = self.getattr_L__mod___layer1___0___act3(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_12 = self.getattr_L__mod___layer1___1___conv1(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_13 = self.getattr_L__mod___layer1___1___bn1(out_12);  out_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_14 = self.getattr_L__mod___layer1___1___act1(out_13);  out_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_12 = self.getattr_L__mod___layer1___1___conv2_conv(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_13 = self.getattr_L__mod___layer1___1___conv2_bn0(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_14 = self.getattr_L__mod___layer1___1___conv2_drop(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_15 = self.getattr_L__mod___layer1___1___conv2_act0(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_16 = x_15.reshape((8, 2, 64, 64, 64));  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_5 = x_16.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_6 = x_gap_5.mean((2, 3), keepdim = True);  x_gap_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_7 = self.getattr_L__mod___layer1___1___conv2_fc1(x_gap_6);  x_gap_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_8 = self.getattr_L__mod___layer1___1___conv2_bn1(x_gap_7);  x_gap_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_9 = self.getattr_L__mod___layer1___1___conv2_act1(x_gap_8);  x_gap_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_2 = self.getattr_L__mod___layer1___1___conv2_fc2(x_gap_9);  x_gap_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_2 = x_attn_2.view(8, 1, 2, -1);  x_attn_2 = None
    x_17 = view_2.transpose(1, 2);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_18 = torch.nn.functional.softmax(x_17, dim = 1);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_19 = x_18.reshape(8, -1);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_3 = x_19.view(8, -1, 1, 1);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_5 = x_attn_3.reshape((8, 2, 64, 1, 1));  x_attn_3 = None
    mul_1 = x_16 * reshape_5;  x_16 = reshape_5 = None
    out_15 = mul_1.sum(dim = 1);  mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_16 = out_15.contiguous();  out_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_17 = self.getattr_L__mod___layer1___1___bn2(out_16);  out_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_18 = self.getattr_L__mod___layer1___1___drop_block(out_17);  out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_19 = self.getattr_L__mod___layer1___1___act2(out_18);  out_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_20 = self.getattr_L__mod___layer1___1___conv3(out_19);  out_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_21 = self.getattr_L__mod___layer1___1___bn3(out_20);  out_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_21 += shortcut_2;  out_22 = out_21;  out_21 = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_3 = self.getattr_L__mod___layer1___1___act3(out_22);  out_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_24 = self.getattr_L__mod___layer1___2___conv1(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_25 = self.getattr_L__mod___layer1___2___bn1(out_24);  out_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_26 = self.getattr_L__mod___layer1___2___act1(out_25);  out_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_20 = self.getattr_L__mod___layer1___2___conv2_conv(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_21 = self.getattr_L__mod___layer1___2___conv2_bn0(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_22 = self.getattr_L__mod___layer1___2___conv2_drop(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_23 = self.getattr_L__mod___layer1___2___conv2_act0(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_24 = x_23.reshape((8, 2, 64, 64, 64));  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_10 = x_24.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_11 = x_gap_10.mean((2, 3), keepdim = True);  x_gap_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_12 = self.getattr_L__mod___layer1___2___conv2_fc1(x_gap_11);  x_gap_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_13 = self.getattr_L__mod___layer1___2___conv2_bn1(x_gap_12);  x_gap_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_14 = self.getattr_L__mod___layer1___2___conv2_act1(x_gap_13);  x_gap_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_4 = self.getattr_L__mod___layer1___2___conv2_fc2(x_gap_14);  x_gap_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_4 = x_attn_4.view(8, 1, 2, -1);  x_attn_4 = None
    x_25 = view_4.transpose(1, 2);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_26 = torch.nn.functional.softmax(x_25, dim = 1);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_27 = x_26.reshape(8, -1);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_5 = x_27.view(8, -1, 1, 1);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_8 = x_attn_5.reshape((8, 2, 64, 1, 1));  x_attn_5 = None
    mul_2 = x_24 * reshape_8;  x_24 = reshape_8 = None
    out_27 = mul_2.sum(dim = 1);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_28 = out_27.contiguous();  out_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_29 = self.getattr_L__mod___layer1___2___bn2(out_28);  out_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_30 = self.getattr_L__mod___layer1___2___drop_block(out_29);  out_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_31 = self.getattr_L__mod___layer1___2___act2(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_32 = self.getattr_L__mod___layer1___2___conv3(out_31);  out_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_33 = self.getattr_L__mod___layer1___2___bn3(out_32);  out_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_33 += shortcut_3;  out_34 = out_33;  out_33 = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_4 = self.getattr_L__mod___layer1___2___act3(out_34);  out_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_36 = self.getattr_L__mod___layer2___0___conv1(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_37 = self.getattr_L__mod___layer2___0___bn1(out_36);  out_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_38 = self.getattr_L__mod___layer2___0___act1(out_37);  out_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_29 = self.getattr_L__mod___layer2___0___conv2_conv(out_38);  out_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_30 = self.getattr_L__mod___layer2___0___conv2_bn0(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_31 = self.getattr_L__mod___layer2___0___conv2_drop(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_32 = self.getattr_L__mod___layer2___0___conv2_act0(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_33 = x_32.reshape((8, 2, 128, 64, 64));  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_15 = x_33.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_16 = x_gap_15.mean((2, 3), keepdim = True);  x_gap_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_17 = self.getattr_L__mod___layer2___0___conv2_fc1(x_gap_16);  x_gap_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_18 = self.getattr_L__mod___layer2___0___conv2_bn1(x_gap_17);  x_gap_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_19 = self.getattr_L__mod___layer2___0___conv2_act1(x_gap_18);  x_gap_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_6 = self.getattr_L__mod___layer2___0___conv2_fc2(x_gap_19);  x_gap_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_6 = x_attn_6.view(8, 1, 2, -1);  x_attn_6 = None
    x_34 = view_6.transpose(1, 2);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_35 = torch.nn.functional.softmax(x_34, dim = 1);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_36 = x_35.reshape(8, -1);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_7 = x_36.view(8, -1, 1, 1);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_11 = x_attn_7.reshape((8, 2, 128, 1, 1));  x_attn_7 = None
    mul_3 = x_33 * reshape_11;  x_33 = reshape_11 = None
    out_39 = mul_3.sum(dim = 1);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_40 = out_39.contiguous();  out_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_41 = self.getattr_L__mod___layer2___0___bn2(out_40);  out_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_42 = self.getattr_L__mod___layer2___0___drop_block(out_41);  out_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_43 = self.getattr_L__mod___layer2___0___act2(out_42);  out_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    out_44 = self.getattr_L__mod___layer2___0___avd_last(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_45 = self.getattr_L__mod___layer2___0___conv3(out_44);  out_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_46 = self.getattr_L__mod___layer2___0___bn3(out_45);  out_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer2___0___downsample_0 = self.getattr_L__mod___layer2___0___downsample_0(shortcut_4);  shortcut_4 = None
    getattr_l__mod___layer2___0___downsample_1 = self.getattr_L__mod___layer2___0___downsample_1(getattr_l__mod___layer2___0___downsample_0);  getattr_l__mod___layer2___0___downsample_0 = None
    shortcut_5 = self.getattr_L__mod___layer2___0___downsample_2(getattr_l__mod___layer2___0___downsample_1);  getattr_l__mod___layer2___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_46 += shortcut_5;  out_47 = out_46;  out_46 = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_6 = self.getattr_L__mod___layer2___0___act3(out_47);  out_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_49 = self.getattr_L__mod___layer2___1___conv1(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_50 = self.getattr_L__mod___layer2___1___bn1(out_49);  out_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_51 = self.getattr_L__mod___layer2___1___act1(out_50);  out_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_37 = self.getattr_L__mod___layer2___1___conv2_conv(out_51);  out_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_38 = self.getattr_L__mod___layer2___1___conv2_bn0(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_39 = self.getattr_L__mod___layer2___1___conv2_drop(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_40 = self.getattr_L__mod___layer2___1___conv2_act0(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_41 = x_40.reshape((8, 2, 128, 32, 32));  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_20 = x_41.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_21 = x_gap_20.mean((2, 3), keepdim = True);  x_gap_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_22 = self.getattr_L__mod___layer2___1___conv2_fc1(x_gap_21);  x_gap_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_23 = self.getattr_L__mod___layer2___1___conv2_bn1(x_gap_22);  x_gap_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_24 = self.getattr_L__mod___layer2___1___conv2_act1(x_gap_23);  x_gap_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_8 = self.getattr_L__mod___layer2___1___conv2_fc2(x_gap_24);  x_gap_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_8 = x_attn_8.view(8, 1, 2, -1);  x_attn_8 = None
    x_42 = view_8.transpose(1, 2);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_43 = torch.nn.functional.softmax(x_42, dim = 1);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_44 = x_43.reshape(8, -1);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_9 = x_44.view(8, -1, 1, 1);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_14 = x_attn_9.reshape((8, 2, 128, 1, 1));  x_attn_9 = None
    mul_4 = x_41 * reshape_14;  x_41 = reshape_14 = None
    out_52 = mul_4.sum(dim = 1);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_53 = out_52.contiguous();  out_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_54 = self.getattr_L__mod___layer2___1___bn2(out_53);  out_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_55 = self.getattr_L__mod___layer2___1___drop_block(out_54);  out_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_56 = self.getattr_L__mod___layer2___1___act2(out_55);  out_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_57 = self.getattr_L__mod___layer2___1___conv3(out_56);  out_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_58 = self.getattr_L__mod___layer2___1___bn3(out_57);  out_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_58 += shortcut_6;  out_59 = out_58;  out_58 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_7 = self.getattr_L__mod___layer2___1___act3(out_59);  out_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_61 = self.getattr_L__mod___layer2___2___conv1(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_62 = self.getattr_L__mod___layer2___2___bn1(out_61);  out_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_63 = self.getattr_L__mod___layer2___2___act1(out_62);  out_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_45 = self.getattr_L__mod___layer2___2___conv2_conv(out_63);  out_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_46 = self.getattr_L__mod___layer2___2___conv2_bn0(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_47 = self.getattr_L__mod___layer2___2___conv2_drop(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_48 = self.getattr_L__mod___layer2___2___conv2_act0(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_49 = x_48.reshape((8, 2, 128, 32, 32));  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_25 = x_49.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_26 = x_gap_25.mean((2, 3), keepdim = True);  x_gap_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_27 = self.getattr_L__mod___layer2___2___conv2_fc1(x_gap_26);  x_gap_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_28 = self.getattr_L__mod___layer2___2___conv2_bn1(x_gap_27);  x_gap_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_29 = self.getattr_L__mod___layer2___2___conv2_act1(x_gap_28);  x_gap_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_10 = self.getattr_L__mod___layer2___2___conv2_fc2(x_gap_29);  x_gap_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_10 = x_attn_10.view(8, 1, 2, -1);  x_attn_10 = None
    x_50 = view_10.transpose(1, 2);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_51 = torch.nn.functional.softmax(x_50, dim = 1);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_52 = x_51.reshape(8, -1);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_11 = x_52.view(8, -1, 1, 1);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_17 = x_attn_11.reshape((8, 2, 128, 1, 1));  x_attn_11 = None
    mul_5 = x_49 * reshape_17;  x_49 = reshape_17 = None
    out_64 = mul_5.sum(dim = 1);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_65 = out_64.contiguous();  out_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_66 = self.getattr_L__mod___layer2___2___bn2(out_65);  out_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_67 = self.getattr_L__mod___layer2___2___drop_block(out_66);  out_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_68 = self.getattr_L__mod___layer2___2___act2(out_67);  out_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_69 = self.getattr_L__mod___layer2___2___conv3(out_68);  out_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_70 = self.getattr_L__mod___layer2___2___bn3(out_69);  out_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_70 += shortcut_7;  out_71 = out_70;  out_70 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_8 = self.getattr_L__mod___layer2___2___act3(out_71);  out_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_73 = self.getattr_L__mod___layer2___3___conv1(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_74 = self.getattr_L__mod___layer2___3___bn1(out_73);  out_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_75 = self.getattr_L__mod___layer2___3___act1(out_74);  out_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_53 = self.getattr_L__mod___layer2___3___conv2_conv(out_75);  out_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_54 = self.getattr_L__mod___layer2___3___conv2_bn0(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_55 = self.getattr_L__mod___layer2___3___conv2_drop(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_56 = self.getattr_L__mod___layer2___3___conv2_act0(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_57 = x_56.reshape((8, 2, 128, 32, 32));  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_30 = x_57.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_31 = x_gap_30.mean((2, 3), keepdim = True);  x_gap_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_32 = self.getattr_L__mod___layer2___3___conv2_fc1(x_gap_31);  x_gap_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_33 = self.getattr_L__mod___layer2___3___conv2_bn1(x_gap_32);  x_gap_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_34 = self.getattr_L__mod___layer2___3___conv2_act1(x_gap_33);  x_gap_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_12 = self.getattr_L__mod___layer2___3___conv2_fc2(x_gap_34);  x_gap_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_12 = x_attn_12.view(8, 1, 2, -1);  x_attn_12 = None
    x_58 = view_12.transpose(1, 2);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_59 = torch.nn.functional.softmax(x_58, dim = 1);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_60 = x_59.reshape(8, -1);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_13 = x_60.view(8, -1, 1, 1);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_20 = x_attn_13.reshape((8, 2, 128, 1, 1));  x_attn_13 = None
    mul_6 = x_57 * reshape_20;  x_57 = reshape_20 = None
    out_76 = mul_6.sum(dim = 1);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_77 = out_76.contiguous();  out_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_78 = self.getattr_L__mod___layer2___3___bn2(out_77);  out_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_79 = self.getattr_L__mod___layer2___3___drop_block(out_78);  out_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_80 = self.getattr_L__mod___layer2___3___act2(out_79);  out_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_81 = self.getattr_L__mod___layer2___3___conv3(out_80);  out_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_82 = self.getattr_L__mod___layer2___3___bn3(out_81);  out_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_82 += shortcut_8;  out_83 = out_82;  out_82 = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_9 = self.getattr_L__mod___layer2___3___act3(out_83);  out_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_85 = self.getattr_L__mod___layer3___0___conv1(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_86 = self.getattr_L__mod___layer3___0___bn1(out_85);  out_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_87 = self.getattr_L__mod___layer3___0___act1(out_86);  out_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_62 = self.getattr_L__mod___layer3___0___conv2_conv(out_87);  out_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_63 = self.getattr_L__mod___layer3___0___conv2_bn0(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_64 = self.getattr_L__mod___layer3___0___conv2_drop(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_65 = self.getattr_L__mod___layer3___0___conv2_act0(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_66 = x_65.reshape((8, 2, 256, 32, 32));  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_35 = x_66.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_36 = x_gap_35.mean((2, 3), keepdim = True);  x_gap_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_37 = self.getattr_L__mod___layer3___0___conv2_fc1(x_gap_36);  x_gap_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_38 = self.getattr_L__mod___layer3___0___conv2_bn1(x_gap_37);  x_gap_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_39 = self.getattr_L__mod___layer3___0___conv2_act1(x_gap_38);  x_gap_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_14 = self.getattr_L__mod___layer3___0___conv2_fc2(x_gap_39);  x_gap_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_14 = x_attn_14.view(8, 1, 2, -1);  x_attn_14 = None
    x_67 = view_14.transpose(1, 2);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_68 = torch.nn.functional.softmax(x_67, dim = 1);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_69 = x_68.reshape(8, -1);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_15 = x_69.view(8, -1, 1, 1);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_23 = x_attn_15.reshape((8, 2, 256, 1, 1));  x_attn_15 = None
    mul_7 = x_66 * reshape_23;  x_66 = reshape_23 = None
    out_88 = mul_7.sum(dim = 1);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_89 = out_88.contiguous();  out_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_90 = self.getattr_L__mod___layer3___0___bn2(out_89);  out_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_91 = self.getattr_L__mod___layer3___0___drop_block(out_90);  out_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_92 = self.getattr_L__mod___layer3___0___act2(out_91);  out_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    out_93 = self.getattr_L__mod___layer3___0___avd_last(out_92);  out_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_94 = self.getattr_L__mod___layer3___0___conv3(out_93);  out_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_95 = self.getattr_L__mod___layer3___0___bn3(out_94);  out_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer3___0___downsample_0 = self.getattr_L__mod___layer3___0___downsample_0(shortcut_9);  shortcut_9 = None
    getattr_l__mod___layer3___0___downsample_1 = self.getattr_L__mod___layer3___0___downsample_1(getattr_l__mod___layer3___0___downsample_0);  getattr_l__mod___layer3___0___downsample_0 = None
    shortcut_10 = self.getattr_L__mod___layer3___0___downsample_2(getattr_l__mod___layer3___0___downsample_1);  getattr_l__mod___layer3___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_95 += shortcut_10;  out_96 = out_95;  out_95 = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_11 = self.getattr_L__mod___layer3___0___act3(out_96);  out_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_98 = self.getattr_L__mod___layer3___1___conv1(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_99 = self.getattr_L__mod___layer3___1___bn1(out_98);  out_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_100 = self.getattr_L__mod___layer3___1___act1(out_99);  out_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_70 = self.getattr_L__mod___layer3___1___conv2_conv(out_100);  out_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_71 = self.getattr_L__mod___layer3___1___conv2_bn0(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_72 = self.getattr_L__mod___layer3___1___conv2_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_73 = self.getattr_L__mod___layer3___1___conv2_act0(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_74 = x_73.reshape((8, 2, 256, 16, 16));  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_40 = x_74.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_41 = x_gap_40.mean((2, 3), keepdim = True);  x_gap_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_42 = self.getattr_L__mod___layer3___1___conv2_fc1(x_gap_41);  x_gap_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_43 = self.getattr_L__mod___layer3___1___conv2_bn1(x_gap_42);  x_gap_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_44 = self.getattr_L__mod___layer3___1___conv2_act1(x_gap_43);  x_gap_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_16 = self.getattr_L__mod___layer3___1___conv2_fc2(x_gap_44);  x_gap_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_16 = x_attn_16.view(8, 1, 2, -1);  x_attn_16 = None
    x_75 = view_16.transpose(1, 2);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_76 = torch.nn.functional.softmax(x_75, dim = 1);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_77 = x_76.reshape(8, -1);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_17 = x_77.view(8, -1, 1, 1);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_26 = x_attn_17.reshape((8, 2, 256, 1, 1));  x_attn_17 = None
    mul_8 = x_74 * reshape_26;  x_74 = reshape_26 = None
    out_101 = mul_8.sum(dim = 1);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_102 = out_101.contiguous();  out_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_103 = self.getattr_L__mod___layer3___1___bn2(out_102);  out_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_104 = self.getattr_L__mod___layer3___1___drop_block(out_103);  out_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_105 = self.getattr_L__mod___layer3___1___act2(out_104);  out_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_106 = self.getattr_L__mod___layer3___1___conv3(out_105);  out_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_107 = self.getattr_L__mod___layer3___1___bn3(out_106);  out_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_107 += shortcut_11;  out_108 = out_107;  out_107 = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_12 = self.getattr_L__mod___layer3___1___act3(out_108);  out_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_110 = self.getattr_L__mod___layer3___2___conv1(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_111 = self.getattr_L__mod___layer3___2___bn1(out_110);  out_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_112 = self.getattr_L__mod___layer3___2___act1(out_111);  out_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_78 = self.getattr_L__mod___layer3___2___conv2_conv(out_112);  out_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_79 = self.getattr_L__mod___layer3___2___conv2_bn0(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_80 = self.getattr_L__mod___layer3___2___conv2_drop(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_81 = self.getattr_L__mod___layer3___2___conv2_act0(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_82 = x_81.reshape((8, 2, 256, 16, 16));  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_45 = x_82.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_46 = x_gap_45.mean((2, 3), keepdim = True);  x_gap_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_47 = self.getattr_L__mod___layer3___2___conv2_fc1(x_gap_46);  x_gap_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_48 = self.getattr_L__mod___layer3___2___conv2_bn1(x_gap_47);  x_gap_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_49 = self.getattr_L__mod___layer3___2___conv2_act1(x_gap_48);  x_gap_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_18 = self.getattr_L__mod___layer3___2___conv2_fc2(x_gap_49);  x_gap_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_18 = x_attn_18.view(8, 1, 2, -1);  x_attn_18 = None
    x_83 = view_18.transpose(1, 2);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_84 = torch.nn.functional.softmax(x_83, dim = 1);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_85 = x_84.reshape(8, -1);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_19 = x_85.view(8, -1, 1, 1);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_29 = x_attn_19.reshape((8, 2, 256, 1, 1));  x_attn_19 = None
    mul_9 = x_82 * reshape_29;  x_82 = reshape_29 = None
    out_113 = mul_9.sum(dim = 1);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_114 = out_113.contiguous();  out_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_115 = self.getattr_L__mod___layer3___2___bn2(out_114);  out_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_116 = self.getattr_L__mod___layer3___2___drop_block(out_115);  out_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_117 = self.getattr_L__mod___layer3___2___act2(out_116);  out_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_118 = self.getattr_L__mod___layer3___2___conv3(out_117);  out_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_119 = self.getattr_L__mod___layer3___2___bn3(out_118);  out_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_119 += shortcut_12;  out_120 = out_119;  out_119 = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_13 = self.getattr_L__mod___layer3___2___act3(out_120);  out_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_122 = self.getattr_L__mod___layer3___3___conv1(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_123 = self.getattr_L__mod___layer3___3___bn1(out_122);  out_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_124 = self.getattr_L__mod___layer3___3___act1(out_123);  out_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_86 = self.getattr_L__mod___layer3___3___conv2_conv(out_124);  out_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_87 = self.getattr_L__mod___layer3___3___conv2_bn0(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_88 = self.getattr_L__mod___layer3___3___conv2_drop(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_89 = self.getattr_L__mod___layer3___3___conv2_act0(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_90 = x_89.reshape((8, 2, 256, 16, 16));  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_50 = x_90.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_51 = x_gap_50.mean((2, 3), keepdim = True);  x_gap_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_52 = self.getattr_L__mod___layer3___3___conv2_fc1(x_gap_51);  x_gap_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_53 = self.getattr_L__mod___layer3___3___conv2_bn1(x_gap_52);  x_gap_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_54 = self.getattr_L__mod___layer3___3___conv2_act1(x_gap_53);  x_gap_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_20 = self.getattr_L__mod___layer3___3___conv2_fc2(x_gap_54);  x_gap_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_20 = x_attn_20.view(8, 1, 2, -1);  x_attn_20 = None
    x_91 = view_20.transpose(1, 2);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_92 = torch.nn.functional.softmax(x_91, dim = 1);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_93 = x_92.reshape(8, -1);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_21 = x_93.view(8, -1, 1, 1);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_32 = x_attn_21.reshape((8, 2, 256, 1, 1));  x_attn_21 = None
    mul_10 = x_90 * reshape_32;  x_90 = reshape_32 = None
    out_125 = mul_10.sum(dim = 1);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_126 = out_125.contiguous();  out_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_127 = self.getattr_L__mod___layer3___3___bn2(out_126);  out_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_128 = self.getattr_L__mod___layer3___3___drop_block(out_127);  out_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_129 = self.getattr_L__mod___layer3___3___act2(out_128);  out_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_130 = self.getattr_L__mod___layer3___3___conv3(out_129);  out_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_131 = self.getattr_L__mod___layer3___3___bn3(out_130);  out_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_131 += shortcut_13;  out_132 = out_131;  out_131 = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_14 = self.getattr_L__mod___layer3___3___act3(out_132);  out_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_134 = self.getattr_L__mod___layer3___4___conv1(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_135 = self.getattr_L__mod___layer3___4___bn1(out_134);  out_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_136 = self.getattr_L__mod___layer3___4___act1(out_135);  out_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_94 = self.getattr_L__mod___layer3___4___conv2_conv(out_136);  out_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_95 = self.getattr_L__mod___layer3___4___conv2_bn0(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_96 = self.getattr_L__mod___layer3___4___conv2_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_97 = self.getattr_L__mod___layer3___4___conv2_act0(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_98 = x_97.reshape((8, 2, 256, 16, 16));  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_55 = x_98.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_56 = x_gap_55.mean((2, 3), keepdim = True);  x_gap_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_57 = self.getattr_L__mod___layer3___4___conv2_fc1(x_gap_56);  x_gap_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_58 = self.getattr_L__mod___layer3___4___conv2_bn1(x_gap_57);  x_gap_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_59 = self.getattr_L__mod___layer3___4___conv2_act1(x_gap_58);  x_gap_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_22 = self.getattr_L__mod___layer3___4___conv2_fc2(x_gap_59);  x_gap_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_22 = x_attn_22.view(8, 1, 2, -1);  x_attn_22 = None
    x_99 = view_22.transpose(1, 2);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_100 = torch.nn.functional.softmax(x_99, dim = 1);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_101 = x_100.reshape(8, -1);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_23 = x_101.view(8, -1, 1, 1);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_35 = x_attn_23.reshape((8, 2, 256, 1, 1));  x_attn_23 = None
    mul_11 = x_98 * reshape_35;  x_98 = reshape_35 = None
    out_137 = mul_11.sum(dim = 1);  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_138 = out_137.contiguous();  out_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_139 = self.getattr_L__mod___layer3___4___bn2(out_138);  out_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_140 = self.getattr_L__mod___layer3___4___drop_block(out_139);  out_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_141 = self.getattr_L__mod___layer3___4___act2(out_140);  out_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_142 = self.getattr_L__mod___layer3___4___conv3(out_141);  out_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_143 = self.getattr_L__mod___layer3___4___bn3(out_142);  out_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_143 += shortcut_14;  out_144 = out_143;  out_143 = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_15 = self.getattr_L__mod___layer3___4___act3(out_144);  out_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_146 = self.getattr_L__mod___layer3___5___conv1(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_147 = self.getattr_L__mod___layer3___5___bn1(out_146);  out_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_148 = self.getattr_L__mod___layer3___5___act1(out_147);  out_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_102 = self.getattr_L__mod___layer3___5___conv2_conv(out_148);  out_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_103 = self.getattr_L__mod___layer3___5___conv2_bn0(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_104 = self.getattr_L__mod___layer3___5___conv2_drop(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_105 = self.getattr_L__mod___layer3___5___conv2_act0(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_106 = x_105.reshape((8, 2, 256, 16, 16));  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_60 = x_106.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_61 = x_gap_60.mean((2, 3), keepdim = True);  x_gap_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_62 = self.getattr_L__mod___layer3___5___conv2_fc1(x_gap_61);  x_gap_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_63 = self.getattr_L__mod___layer3___5___conv2_bn1(x_gap_62);  x_gap_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_64 = self.getattr_L__mod___layer3___5___conv2_act1(x_gap_63);  x_gap_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_24 = self.getattr_L__mod___layer3___5___conv2_fc2(x_gap_64);  x_gap_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_24 = x_attn_24.view(8, 1, 2, -1);  x_attn_24 = None
    x_107 = view_24.transpose(1, 2);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_108 = torch.nn.functional.softmax(x_107, dim = 1);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_109 = x_108.reshape(8, -1);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_25 = x_109.view(8, -1, 1, 1);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_38 = x_attn_25.reshape((8, 2, 256, 1, 1));  x_attn_25 = None
    mul_12 = x_106 * reshape_38;  x_106 = reshape_38 = None
    out_149 = mul_12.sum(dim = 1);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_150 = out_149.contiguous();  out_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_151 = self.getattr_L__mod___layer3___5___bn2(out_150);  out_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_152 = self.getattr_L__mod___layer3___5___drop_block(out_151);  out_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_153 = self.getattr_L__mod___layer3___5___act2(out_152);  out_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_154 = self.getattr_L__mod___layer3___5___conv3(out_153);  out_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_155 = self.getattr_L__mod___layer3___5___bn3(out_154);  out_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_155 += shortcut_15;  out_156 = out_155;  out_155 = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_16 = self.getattr_L__mod___layer3___5___act3(out_156);  out_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_158 = self.getattr_L__mod___layer3___6___conv1(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_159 = self.getattr_L__mod___layer3___6___bn1(out_158);  out_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_160 = self.getattr_L__mod___layer3___6___act1(out_159);  out_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_110 = self.getattr_L__mod___layer3___6___conv2_conv(out_160);  out_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_111 = self.getattr_L__mod___layer3___6___conv2_bn0(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_112 = self.getattr_L__mod___layer3___6___conv2_drop(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_113 = self.getattr_L__mod___layer3___6___conv2_act0(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_114 = x_113.reshape((8, 2, 256, 16, 16));  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_65 = x_114.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_66 = x_gap_65.mean((2, 3), keepdim = True);  x_gap_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_67 = self.getattr_L__mod___layer3___6___conv2_fc1(x_gap_66);  x_gap_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_68 = self.getattr_L__mod___layer3___6___conv2_bn1(x_gap_67);  x_gap_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_69 = self.getattr_L__mod___layer3___6___conv2_act1(x_gap_68);  x_gap_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_26 = self.getattr_L__mod___layer3___6___conv2_fc2(x_gap_69);  x_gap_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_26 = x_attn_26.view(8, 1, 2, -1);  x_attn_26 = None
    x_115 = view_26.transpose(1, 2);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_116 = torch.nn.functional.softmax(x_115, dim = 1);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_117 = x_116.reshape(8, -1);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_27 = x_117.view(8, -1, 1, 1);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_41 = x_attn_27.reshape((8, 2, 256, 1, 1));  x_attn_27 = None
    mul_13 = x_114 * reshape_41;  x_114 = reshape_41 = None
    out_161 = mul_13.sum(dim = 1);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_162 = out_161.contiguous();  out_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_163 = self.getattr_L__mod___layer3___6___bn2(out_162);  out_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_164 = self.getattr_L__mod___layer3___6___drop_block(out_163);  out_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_165 = self.getattr_L__mod___layer3___6___act2(out_164);  out_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_166 = self.getattr_L__mod___layer3___6___conv3(out_165);  out_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_167 = self.getattr_L__mod___layer3___6___bn3(out_166);  out_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_167 += shortcut_16;  out_168 = out_167;  out_167 = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_17 = self.getattr_L__mod___layer3___6___act3(out_168);  out_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_170 = self.getattr_L__mod___layer3___7___conv1(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_171 = self.getattr_L__mod___layer3___7___bn1(out_170);  out_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_172 = self.getattr_L__mod___layer3___7___act1(out_171);  out_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_118 = self.getattr_L__mod___layer3___7___conv2_conv(out_172);  out_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_119 = self.getattr_L__mod___layer3___7___conv2_bn0(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_120 = self.getattr_L__mod___layer3___7___conv2_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_121 = self.getattr_L__mod___layer3___7___conv2_act0(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_122 = x_121.reshape((8, 2, 256, 16, 16));  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_70 = x_122.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_71 = x_gap_70.mean((2, 3), keepdim = True);  x_gap_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_72 = self.getattr_L__mod___layer3___7___conv2_fc1(x_gap_71);  x_gap_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_73 = self.getattr_L__mod___layer3___7___conv2_bn1(x_gap_72);  x_gap_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_74 = self.getattr_L__mod___layer3___7___conv2_act1(x_gap_73);  x_gap_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_28 = self.getattr_L__mod___layer3___7___conv2_fc2(x_gap_74);  x_gap_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_28 = x_attn_28.view(8, 1, 2, -1);  x_attn_28 = None
    x_123 = view_28.transpose(1, 2);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_124 = torch.nn.functional.softmax(x_123, dim = 1);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_125 = x_124.reshape(8, -1);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_29 = x_125.view(8, -1, 1, 1);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_44 = x_attn_29.reshape((8, 2, 256, 1, 1));  x_attn_29 = None
    mul_14 = x_122 * reshape_44;  x_122 = reshape_44 = None
    out_173 = mul_14.sum(dim = 1);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_174 = out_173.contiguous();  out_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_175 = self.getattr_L__mod___layer3___7___bn2(out_174);  out_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_176 = self.getattr_L__mod___layer3___7___drop_block(out_175);  out_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_177 = self.getattr_L__mod___layer3___7___act2(out_176);  out_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_178 = self.getattr_L__mod___layer3___7___conv3(out_177);  out_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_179 = self.getattr_L__mod___layer3___7___bn3(out_178);  out_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_179 += shortcut_17;  out_180 = out_179;  out_179 = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_18 = self.getattr_L__mod___layer3___7___act3(out_180);  out_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_182 = self.getattr_L__mod___layer3___8___conv1(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_183 = self.getattr_L__mod___layer3___8___bn1(out_182);  out_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_184 = self.getattr_L__mod___layer3___8___act1(out_183);  out_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_126 = self.getattr_L__mod___layer3___8___conv2_conv(out_184);  out_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_127 = self.getattr_L__mod___layer3___8___conv2_bn0(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_128 = self.getattr_L__mod___layer3___8___conv2_drop(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_129 = self.getattr_L__mod___layer3___8___conv2_act0(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_130 = x_129.reshape((8, 2, 256, 16, 16));  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_75 = x_130.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_76 = x_gap_75.mean((2, 3), keepdim = True);  x_gap_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_77 = self.getattr_L__mod___layer3___8___conv2_fc1(x_gap_76);  x_gap_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_78 = self.getattr_L__mod___layer3___8___conv2_bn1(x_gap_77);  x_gap_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_79 = self.getattr_L__mod___layer3___8___conv2_act1(x_gap_78);  x_gap_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_30 = self.getattr_L__mod___layer3___8___conv2_fc2(x_gap_79);  x_gap_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_30 = x_attn_30.view(8, 1, 2, -1);  x_attn_30 = None
    x_131 = view_30.transpose(1, 2);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_132 = torch.nn.functional.softmax(x_131, dim = 1);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_133 = x_132.reshape(8, -1);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_31 = x_133.view(8, -1, 1, 1);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_47 = x_attn_31.reshape((8, 2, 256, 1, 1));  x_attn_31 = None
    mul_15 = x_130 * reshape_47;  x_130 = reshape_47 = None
    out_185 = mul_15.sum(dim = 1);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_186 = out_185.contiguous();  out_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_187 = self.getattr_L__mod___layer3___8___bn2(out_186);  out_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_188 = self.getattr_L__mod___layer3___8___drop_block(out_187);  out_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_189 = self.getattr_L__mod___layer3___8___act2(out_188);  out_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_190 = self.getattr_L__mod___layer3___8___conv3(out_189);  out_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_191 = self.getattr_L__mod___layer3___8___bn3(out_190);  out_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_191 += shortcut_18;  out_192 = out_191;  out_191 = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_19 = self.getattr_L__mod___layer3___8___act3(out_192);  out_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_194 = self.getattr_L__mod___layer3___9___conv1(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_195 = self.getattr_L__mod___layer3___9___bn1(out_194);  out_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_196 = self.getattr_L__mod___layer3___9___act1(out_195);  out_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_134 = self.getattr_L__mod___layer3___9___conv2_conv(out_196);  out_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_135 = self.getattr_L__mod___layer3___9___conv2_bn0(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_136 = self.getattr_L__mod___layer3___9___conv2_drop(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_137 = self.getattr_L__mod___layer3___9___conv2_act0(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_138 = x_137.reshape((8, 2, 256, 16, 16));  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_80 = x_138.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_81 = x_gap_80.mean((2, 3), keepdim = True);  x_gap_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_82 = self.getattr_L__mod___layer3___9___conv2_fc1(x_gap_81);  x_gap_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_83 = self.getattr_L__mod___layer3___9___conv2_bn1(x_gap_82);  x_gap_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_84 = self.getattr_L__mod___layer3___9___conv2_act1(x_gap_83);  x_gap_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_32 = self.getattr_L__mod___layer3___9___conv2_fc2(x_gap_84);  x_gap_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_32 = x_attn_32.view(8, 1, 2, -1);  x_attn_32 = None
    x_139 = view_32.transpose(1, 2);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_140 = torch.nn.functional.softmax(x_139, dim = 1);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_141 = x_140.reshape(8, -1);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_33 = x_141.view(8, -1, 1, 1);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_50 = x_attn_33.reshape((8, 2, 256, 1, 1));  x_attn_33 = None
    mul_16 = x_138 * reshape_50;  x_138 = reshape_50 = None
    out_197 = mul_16.sum(dim = 1);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_198 = out_197.contiguous();  out_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_199 = self.getattr_L__mod___layer3___9___bn2(out_198);  out_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_200 = self.getattr_L__mod___layer3___9___drop_block(out_199);  out_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_201 = self.getattr_L__mod___layer3___9___act2(out_200);  out_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_202 = self.getattr_L__mod___layer3___9___conv3(out_201);  out_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_203 = self.getattr_L__mod___layer3___9___bn3(out_202);  out_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_203 += shortcut_19;  out_204 = out_203;  out_203 = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_20 = self.getattr_L__mod___layer3___9___act3(out_204);  out_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_206 = self.getattr_L__mod___layer3___10___conv1(shortcut_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_207 = self.getattr_L__mod___layer3___10___bn1(out_206);  out_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_208 = self.getattr_L__mod___layer3___10___act1(out_207);  out_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_142 = self.getattr_L__mod___layer3___10___conv2_conv(out_208);  out_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_143 = self.getattr_L__mod___layer3___10___conv2_bn0(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_144 = self.getattr_L__mod___layer3___10___conv2_drop(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_145 = self.getattr_L__mod___layer3___10___conv2_act0(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_146 = x_145.reshape((8, 2, 256, 16, 16));  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_85 = x_146.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_86 = x_gap_85.mean((2, 3), keepdim = True);  x_gap_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_87 = self.getattr_L__mod___layer3___10___conv2_fc1(x_gap_86);  x_gap_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_88 = self.getattr_L__mod___layer3___10___conv2_bn1(x_gap_87);  x_gap_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_89 = self.getattr_L__mod___layer3___10___conv2_act1(x_gap_88);  x_gap_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_34 = self.getattr_L__mod___layer3___10___conv2_fc2(x_gap_89);  x_gap_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_34 = x_attn_34.view(8, 1, 2, -1);  x_attn_34 = None
    x_147 = view_34.transpose(1, 2);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_148 = torch.nn.functional.softmax(x_147, dim = 1);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_149 = x_148.reshape(8, -1);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_35 = x_149.view(8, -1, 1, 1);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_53 = x_attn_35.reshape((8, 2, 256, 1, 1));  x_attn_35 = None
    mul_17 = x_146 * reshape_53;  x_146 = reshape_53 = None
    out_209 = mul_17.sum(dim = 1);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_210 = out_209.contiguous();  out_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_211 = self.getattr_L__mod___layer3___10___bn2(out_210);  out_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_212 = self.getattr_L__mod___layer3___10___drop_block(out_211);  out_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_213 = self.getattr_L__mod___layer3___10___act2(out_212);  out_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_214 = self.getattr_L__mod___layer3___10___conv3(out_213);  out_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_215 = self.getattr_L__mod___layer3___10___bn3(out_214);  out_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_215 += shortcut_20;  out_216 = out_215;  out_215 = shortcut_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_21 = self.getattr_L__mod___layer3___10___act3(out_216);  out_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_218 = self.getattr_L__mod___layer3___11___conv1(shortcut_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_219 = self.getattr_L__mod___layer3___11___bn1(out_218);  out_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_220 = self.getattr_L__mod___layer3___11___act1(out_219);  out_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_150 = self.getattr_L__mod___layer3___11___conv2_conv(out_220);  out_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_151 = self.getattr_L__mod___layer3___11___conv2_bn0(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_152 = self.getattr_L__mod___layer3___11___conv2_drop(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_153 = self.getattr_L__mod___layer3___11___conv2_act0(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_154 = x_153.reshape((8, 2, 256, 16, 16));  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_90 = x_154.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_91 = x_gap_90.mean((2, 3), keepdim = True);  x_gap_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_92 = self.getattr_L__mod___layer3___11___conv2_fc1(x_gap_91);  x_gap_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_93 = self.getattr_L__mod___layer3___11___conv2_bn1(x_gap_92);  x_gap_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_94 = self.getattr_L__mod___layer3___11___conv2_act1(x_gap_93);  x_gap_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_36 = self.getattr_L__mod___layer3___11___conv2_fc2(x_gap_94);  x_gap_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_36 = x_attn_36.view(8, 1, 2, -1);  x_attn_36 = None
    x_155 = view_36.transpose(1, 2);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_156 = torch.nn.functional.softmax(x_155, dim = 1);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_157 = x_156.reshape(8, -1);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_37 = x_157.view(8, -1, 1, 1);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_56 = x_attn_37.reshape((8, 2, 256, 1, 1));  x_attn_37 = None
    mul_18 = x_154 * reshape_56;  x_154 = reshape_56 = None
    out_221 = mul_18.sum(dim = 1);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_222 = out_221.contiguous();  out_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_223 = self.getattr_L__mod___layer3___11___bn2(out_222);  out_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_224 = self.getattr_L__mod___layer3___11___drop_block(out_223);  out_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_225 = self.getattr_L__mod___layer3___11___act2(out_224);  out_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_226 = self.getattr_L__mod___layer3___11___conv3(out_225);  out_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_227 = self.getattr_L__mod___layer3___11___bn3(out_226);  out_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_227 += shortcut_21;  out_228 = out_227;  out_227 = shortcut_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_22 = self.getattr_L__mod___layer3___11___act3(out_228);  out_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_230 = self.getattr_L__mod___layer3___12___conv1(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_231 = self.getattr_L__mod___layer3___12___bn1(out_230);  out_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_232 = self.getattr_L__mod___layer3___12___act1(out_231);  out_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_158 = self.getattr_L__mod___layer3___12___conv2_conv(out_232);  out_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_159 = self.getattr_L__mod___layer3___12___conv2_bn0(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_160 = self.getattr_L__mod___layer3___12___conv2_drop(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_161 = self.getattr_L__mod___layer3___12___conv2_act0(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_162 = x_161.reshape((8, 2, 256, 16, 16));  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_95 = x_162.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_96 = x_gap_95.mean((2, 3), keepdim = True);  x_gap_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_97 = self.getattr_L__mod___layer3___12___conv2_fc1(x_gap_96);  x_gap_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_98 = self.getattr_L__mod___layer3___12___conv2_bn1(x_gap_97);  x_gap_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_99 = self.getattr_L__mod___layer3___12___conv2_act1(x_gap_98);  x_gap_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_38 = self.getattr_L__mod___layer3___12___conv2_fc2(x_gap_99);  x_gap_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_38 = x_attn_38.view(8, 1, 2, -1);  x_attn_38 = None
    x_163 = view_38.transpose(1, 2);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_164 = torch.nn.functional.softmax(x_163, dim = 1);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_165 = x_164.reshape(8, -1);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_39 = x_165.view(8, -1, 1, 1);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_59 = x_attn_39.reshape((8, 2, 256, 1, 1));  x_attn_39 = None
    mul_19 = x_162 * reshape_59;  x_162 = reshape_59 = None
    out_233 = mul_19.sum(dim = 1);  mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_234 = out_233.contiguous();  out_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_235 = self.getattr_L__mod___layer3___12___bn2(out_234);  out_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_236 = self.getattr_L__mod___layer3___12___drop_block(out_235);  out_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_237 = self.getattr_L__mod___layer3___12___act2(out_236);  out_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_238 = self.getattr_L__mod___layer3___12___conv3(out_237);  out_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_239 = self.getattr_L__mod___layer3___12___bn3(out_238);  out_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_239 += shortcut_22;  out_240 = out_239;  out_239 = shortcut_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_23 = self.getattr_L__mod___layer3___12___act3(out_240);  out_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_242 = self.getattr_L__mod___layer3___13___conv1(shortcut_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_243 = self.getattr_L__mod___layer3___13___bn1(out_242);  out_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_244 = self.getattr_L__mod___layer3___13___act1(out_243);  out_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_166 = self.getattr_L__mod___layer3___13___conv2_conv(out_244);  out_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_167 = self.getattr_L__mod___layer3___13___conv2_bn0(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_168 = self.getattr_L__mod___layer3___13___conv2_drop(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_169 = self.getattr_L__mod___layer3___13___conv2_act0(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_170 = x_169.reshape((8, 2, 256, 16, 16));  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_100 = x_170.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_101 = x_gap_100.mean((2, 3), keepdim = True);  x_gap_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_102 = self.getattr_L__mod___layer3___13___conv2_fc1(x_gap_101);  x_gap_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_103 = self.getattr_L__mod___layer3___13___conv2_bn1(x_gap_102);  x_gap_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_104 = self.getattr_L__mod___layer3___13___conv2_act1(x_gap_103);  x_gap_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_40 = self.getattr_L__mod___layer3___13___conv2_fc2(x_gap_104);  x_gap_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_40 = x_attn_40.view(8, 1, 2, -1);  x_attn_40 = None
    x_171 = view_40.transpose(1, 2);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_172 = torch.nn.functional.softmax(x_171, dim = 1);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_173 = x_172.reshape(8, -1);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_41 = x_173.view(8, -1, 1, 1);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_62 = x_attn_41.reshape((8, 2, 256, 1, 1));  x_attn_41 = None
    mul_20 = x_170 * reshape_62;  x_170 = reshape_62 = None
    out_245 = mul_20.sum(dim = 1);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_246 = out_245.contiguous();  out_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_247 = self.getattr_L__mod___layer3___13___bn2(out_246);  out_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_248 = self.getattr_L__mod___layer3___13___drop_block(out_247);  out_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_249 = self.getattr_L__mod___layer3___13___act2(out_248);  out_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_250 = self.getattr_L__mod___layer3___13___conv3(out_249);  out_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_251 = self.getattr_L__mod___layer3___13___bn3(out_250);  out_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_251 += shortcut_23;  out_252 = out_251;  out_251 = shortcut_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_24 = self.getattr_L__mod___layer3___13___act3(out_252);  out_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_254 = self.getattr_L__mod___layer3___14___conv1(shortcut_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_255 = self.getattr_L__mod___layer3___14___bn1(out_254);  out_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_256 = self.getattr_L__mod___layer3___14___act1(out_255);  out_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_174 = self.getattr_L__mod___layer3___14___conv2_conv(out_256);  out_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_175 = self.getattr_L__mod___layer3___14___conv2_bn0(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_176 = self.getattr_L__mod___layer3___14___conv2_drop(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_177 = self.getattr_L__mod___layer3___14___conv2_act0(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_178 = x_177.reshape((8, 2, 256, 16, 16));  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_105 = x_178.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_106 = x_gap_105.mean((2, 3), keepdim = True);  x_gap_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_107 = self.getattr_L__mod___layer3___14___conv2_fc1(x_gap_106);  x_gap_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_108 = self.getattr_L__mod___layer3___14___conv2_bn1(x_gap_107);  x_gap_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_109 = self.getattr_L__mod___layer3___14___conv2_act1(x_gap_108);  x_gap_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_42 = self.getattr_L__mod___layer3___14___conv2_fc2(x_gap_109);  x_gap_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_42 = x_attn_42.view(8, 1, 2, -1);  x_attn_42 = None
    x_179 = view_42.transpose(1, 2);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_180 = torch.nn.functional.softmax(x_179, dim = 1);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_181 = x_180.reshape(8, -1);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_43 = x_181.view(8, -1, 1, 1);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_65 = x_attn_43.reshape((8, 2, 256, 1, 1));  x_attn_43 = None
    mul_21 = x_178 * reshape_65;  x_178 = reshape_65 = None
    out_257 = mul_21.sum(dim = 1);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_258 = out_257.contiguous();  out_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_259 = self.getattr_L__mod___layer3___14___bn2(out_258);  out_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_260 = self.getattr_L__mod___layer3___14___drop_block(out_259);  out_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_261 = self.getattr_L__mod___layer3___14___act2(out_260);  out_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_262 = self.getattr_L__mod___layer3___14___conv3(out_261);  out_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_263 = self.getattr_L__mod___layer3___14___bn3(out_262);  out_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_263 += shortcut_24;  out_264 = out_263;  out_263 = shortcut_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_25 = self.getattr_L__mod___layer3___14___act3(out_264);  out_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_266 = self.getattr_L__mod___layer3___15___conv1(shortcut_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_267 = self.getattr_L__mod___layer3___15___bn1(out_266);  out_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_268 = self.getattr_L__mod___layer3___15___act1(out_267);  out_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_182 = self.getattr_L__mod___layer3___15___conv2_conv(out_268);  out_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_183 = self.getattr_L__mod___layer3___15___conv2_bn0(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_184 = self.getattr_L__mod___layer3___15___conv2_drop(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_185 = self.getattr_L__mod___layer3___15___conv2_act0(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_186 = x_185.reshape((8, 2, 256, 16, 16));  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_110 = x_186.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_111 = x_gap_110.mean((2, 3), keepdim = True);  x_gap_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_112 = self.getattr_L__mod___layer3___15___conv2_fc1(x_gap_111);  x_gap_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_113 = self.getattr_L__mod___layer3___15___conv2_bn1(x_gap_112);  x_gap_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_114 = self.getattr_L__mod___layer3___15___conv2_act1(x_gap_113);  x_gap_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_44 = self.getattr_L__mod___layer3___15___conv2_fc2(x_gap_114);  x_gap_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_44 = x_attn_44.view(8, 1, 2, -1);  x_attn_44 = None
    x_187 = view_44.transpose(1, 2);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_188 = torch.nn.functional.softmax(x_187, dim = 1);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_189 = x_188.reshape(8, -1);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_45 = x_189.view(8, -1, 1, 1);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_68 = x_attn_45.reshape((8, 2, 256, 1, 1));  x_attn_45 = None
    mul_22 = x_186 * reshape_68;  x_186 = reshape_68 = None
    out_269 = mul_22.sum(dim = 1);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_270 = out_269.contiguous();  out_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_271 = self.getattr_L__mod___layer3___15___bn2(out_270);  out_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_272 = self.getattr_L__mod___layer3___15___drop_block(out_271);  out_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_273 = self.getattr_L__mod___layer3___15___act2(out_272);  out_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_274 = self.getattr_L__mod___layer3___15___conv3(out_273);  out_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_275 = self.getattr_L__mod___layer3___15___bn3(out_274);  out_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_275 += shortcut_25;  out_276 = out_275;  out_275 = shortcut_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_26 = self.getattr_L__mod___layer3___15___act3(out_276);  out_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_278 = self.getattr_L__mod___layer3___16___conv1(shortcut_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_279 = self.getattr_L__mod___layer3___16___bn1(out_278);  out_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_280 = self.getattr_L__mod___layer3___16___act1(out_279);  out_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_190 = self.getattr_L__mod___layer3___16___conv2_conv(out_280);  out_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_191 = self.getattr_L__mod___layer3___16___conv2_bn0(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_192 = self.getattr_L__mod___layer3___16___conv2_drop(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_193 = self.getattr_L__mod___layer3___16___conv2_act0(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_194 = x_193.reshape((8, 2, 256, 16, 16));  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_115 = x_194.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_116 = x_gap_115.mean((2, 3), keepdim = True);  x_gap_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_117 = self.getattr_L__mod___layer3___16___conv2_fc1(x_gap_116);  x_gap_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_118 = self.getattr_L__mod___layer3___16___conv2_bn1(x_gap_117);  x_gap_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_119 = self.getattr_L__mod___layer3___16___conv2_act1(x_gap_118);  x_gap_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_46 = self.getattr_L__mod___layer3___16___conv2_fc2(x_gap_119);  x_gap_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_46 = x_attn_46.view(8, 1, 2, -1);  x_attn_46 = None
    x_195 = view_46.transpose(1, 2);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_196 = torch.nn.functional.softmax(x_195, dim = 1);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_197 = x_196.reshape(8, -1);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_47 = x_197.view(8, -1, 1, 1);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_71 = x_attn_47.reshape((8, 2, 256, 1, 1));  x_attn_47 = None
    mul_23 = x_194 * reshape_71;  x_194 = reshape_71 = None
    out_281 = mul_23.sum(dim = 1);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_282 = out_281.contiguous();  out_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_283 = self.getattr_L__mod___layer3___16___bn2(out_282);  out_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_284 = self.getattr_L__mod___layer3___16___drop_block(out_283);  out_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_285 = self.getattr_L__mod___layer3___16___act2(out_284);  out_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_286 = self.getattr_L__mod___layer3___16___conv3(out_285);  out_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_287 = self.getattr_L__mod___layer3___16___bn3(out_286);  out_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_287 += shortcut_26;  out_288 = out_287;  out_287 = shortcut_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_27 = self.getattr_L__mod___layer3___16___act3(out_288);  out_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_290 = self.getattr_L__mod___layer3___17___conv1(shortcut_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_291 = self.getattr_L__mod___layer3___17___bn1(out_290);  out_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_292 = self.getattr_L__mod___layer3___17___act1(out_291);  out_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_198 = self.getattr_L__mod___layer3___17___conv2_conv(out_292);  out_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_199 = self.getattr_L__mod___layer3___17___conv2_bn0(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_200 = self.getattr_L__mod___layer3___17___conv2_drop(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_201 = self.getattr_L__mod___layer3___17___conv2_act0(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_202 = x_201.reshape((8, 2, 256, 16, 16));  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_120 = x_202.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_121 = x_gap_120.mean((2, 3), keepdim = True);  x_gap_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_122 = self.getattr_L__mod___layer3___17___conv2_fc1(x_gap_121);  x_gap_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_123 = self.getattr_L__mod___layer3___17___conv2_bn1(x_gap_122);  x_gap_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_124 = self.getattr_L__mod___layer3___17___conv2_act1(x_gap_123);  x_gap_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_48 = self.getattr_L__mod___layer3___17___conv2_fc2(x_gap_124);  x_gap_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_48 = x_attn_48.view(8, 1, 2, -1);  x_attn_48 = None
    x_203 = view_48.transpose(1, 2);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_204 = torch.nn.functional.softmax(x_203, dim = 1);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_205 = x_204.reshape(8, -1);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_49 = x_205.view(8, -1, 1, 1);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_74 = x_attn_49.reshape((8, 2, 256, 1, 1));  x_attn_49 = None
    mul_24 = x_202 * reshape_74;  x_202 = reshape_74 = None
    out_293 = mul_24.sum(dim = 1);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_294 = out_293.contiguous();  out_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_295 = self.getattr_L__mod___layer3___17___bn2(out_294);  out_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_296 = self.getattr_L__mod___layer3___17___drop_block(out_295);  out_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_297 = self.getattr_L__mod___layer3___17___act2(out_296);  out_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_298 = self.getattr_L__mod___layer3___17___conv3(out_297);  out_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_299 = self.getattr_L__mod___layer3___17___bn3(out_298);  out_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_299 += shortcut_27;  out_300 = out_299;  out_299 = shortcut_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_28 = self.getattr_L__mod___layer3___17___act3(out_300);  out_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_302 = self.getattr_L__mod___layer3___18___conv1(shortcut_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_303 = self.getattr_L__mod___layer3___18___bn1(out_302);  out_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_304 = self.getattr_L__mod___layer3___18___act1(out_303);  out_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_206 = self.getattr_L__mod___layer3___18___conv2_conv(out_304);  out_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_207 = self.getattr_L__mod___layer3___18___conv2_bn0(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_208 = self.getattr_L__mod___layer3___18___conv2_drop(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_209 = self.getattr_L__mod___layer3___18___conv2_act0(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_210 = x_209.reshape((8, 2, 256, 16, 16));  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_125 = x_210.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_126 = x_gap_125.mean((2, 3), keepdim = True);  x_gap_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_127 = self.getattr_L__mod___layer3___18___conv2_fc1(x_gap_126);  x_gap_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_128 = self.getattr_L__mod___layer3___18___conv2_bn1(x_gap_127);  x_gap_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_129 = self.getattr_L__mod___layer3___18___conv2_act1(x_gap_128);  x_gap_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_50 = self.getattr_L__mod___layer3___18___conv2_fc2(x_gap_129);  x_gap_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_50 = x_attn_50.view(8, 1, 2, -1);  x_attn_50 = None
    x_211 = view_50.transpose(1, 2);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_212 = torch.nn.functional.softmax(x_211, dim = 1);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_213 = x_212.reshape(8, -1);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_51 = x_213.view(8, -1, 1, 1);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_77 = x_attn_51.reshape((8, 2, 256, 1, 1));  x_attn_51 = None
    mul_25 = x_210 * reshape_77;  x_210 = reshape_77 = None
    out_305 = mul_25.sum(dim = 1);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_306 = out_305.contiguous();  out_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_307 = self.getattr_L__mod___layer3___18___bn2(out_306);  out_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_308 = self.getattr_L__mod___layer3___18___drop_block(out_307);  out_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_309 = self.getattr_L__mod___layer3___18___act2(out_308);  out_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_310 = self.getattr_L__mod___layer3___18___conv3(out_309);  out_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_311 = self.getattr_L__mod___layer3___18___bn3(out_310);  out_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_311 += shortcut_28;  out_312 = out_311;  out_311 = shortcut_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_29 = self.getattr_L__mod___layer3___18___act3(out_312);  out_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_314 = self.getattr_L__mod___layer3___19___conv1(shortcut_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_315 = self.getattr_L__mod___layer3___19___bn1(out_314);  out_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_316 = self.getattr_L__mod___layer3___19___act1(out_315);  out_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_214 = self.getattr_L__mod___layer3___19___conv2_conv(out_316);  out_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_215 = self.getattr_L__mod___layer3___19___conv2_bn0(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_216 = self.getattr_L__mod___layer3___19___conv2_drop(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_217 = self.getattr_L__mod___layer3___19___conv2_act0(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_218 = x_217.reshape((8, 2, 256, 16, 16));  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_130 = x_218.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_131 = x_gap_130.mean((2, 3), keepdim = True);  x_gap_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_132 = self.getattr_L__mod___layer3___19___conv2_fc1(x_gap_131);  x_gap_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_133 = self.getattr_L__mod___layer3___19___conv2_bn1(x_gap_132);  x_gap_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_134 = self.getattr_L__mod___layer3___19___conv2_act1(x_gap_133);  x_gap_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_52 = self.getattr_L__mod___layer3___19___conv2_fc2(x_gap_134);  x_gap_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_52 = x_attn_52.view(8, 1, 2, -1);  x_attn_52 = None
    x_219 = view_52.transpose(1, 2);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_220 = torch.nn.functional.softmax(x_219, dim = 1);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_221 = x_220.reshape(8, -1);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_53 = x_221.view(8, -1, 1, 1);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_80 = x_attn_53.reshape((8, 2, 256, 1, 1));  x_attn_53 = None
    mul_26 = x_218 * reshape_80;  x_218 = reshape_80 = None
    out_317 = mul_26.sum(dim = 1);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_318 = out_317.contiguous();  out_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_319 = self.getattr_L__mod___layer3___19___bn2(out_318);  out_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_320 = self.getattr_L__mod___layer3___19___drop_block(out_319);  out_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_321 = self.getattr_L__mod___layer3___19___act2(out_320);  out_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_322 = self.getattr_L__mod___layer3___19___conv3(out_321);  out_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_323 = self.getattr_L__mod___layer3___19___bn3(out_322);  out_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_323 += shortcut_29;  out_324 = out_323;  out_323 = shortcut_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_30 = self.getattr_L__mod___layer3___19___act3(out_324);  out_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_326 = self.getattr_L__mod___layer3___20___conv1(shortcut_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_327 = self.getattr_L__mod___layer3___20___bn1(out_326);  out_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_328 = self.getattr_L__mod___layer3___20___act1(out_327);  out_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_222 = self.getattr_L__mod___layer3___20___conv2_conv(out_328);  out_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_223 = self.getattr_L__mod___layer3___20___conv2_bn0(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_224 = self.getattr_L__mod___layer3___20___conv2_drop(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_225 = self.getattr_L__mod___layer3___20___conv2_act0(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_226 = x_225.reshape((8, 2, 256, 16, 16));  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_135 = x_226.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_136 = x_gap_135.mean((2, 3), keepdim = True);  x_gap_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_137 = self.getattr_L__mod___layer3___20___conv2_fc1(x_gap_136);  x_gap_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_138 = self.getattr_L__mod___layer3___20___conv2_bn1(x_gap_137);  x_gap_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_139 = self.getattr_L__mod___layer3___20___conv2_act1(x_gap_138);  x_gap_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_54 = self.getattr_L__mod___layer3___20___conv2_fc2(x_gap_139);  x_gap_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_54 = x_attn_54.view(8, 1, 2, -1);  x_attn_54 = None
    x_227 = view_54.transpose(1, 2);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_228 = torch.nn.functional.softmax(x_227, dim = 1);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_229 = x_228.reshape(8, -1);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_55 = x_229.view(8, -1, 1, 1);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_83 = x_attn_55.reshape((8, 2, 256, 1, 1));  x_attn_55 = None
    mul_27 = x_226 * reshape_83;  x_226 = reshape_83 = None
    out_329 = mul_27.sum(dim = 1);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_330 = out_329.contiguous();  out_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_331 = self.getattr_L__mod___layer3___20___bn2(out_330);  out_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_332 = self.getattr_L__mod___layer3___20___drop_block(out_331);  out_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_333 = self.getattr_L__mod___layer3___20___act2(out_332);  out_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_334 = self.getattr_L__mod___layer3___20___conv3(out_333);  out_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_335 = self.getattr_L__mod___layer3___20___bn3(out_334);  out_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_335 += shortcut_30;  out_336 = out_335;  out_335 = shortcut_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_31 = self.getattr_L__mod___layer3___20___act3(out_336);  out_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_338 = self.getattr_L__mod___layer3___21___conv1(shortcut_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_339 = self.getattr_L__mod___layer3___21___bn1(out_338);  out_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_340 = self.getattr_L__mod___layer3___21___act1(out_339);  out_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_230 = self.getattr_L__mod___layer3___21___conv2_conv(out_340);  out_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_231 = self.getattr_L__mod___layer3___21___conv2_bn0(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_232 = self.getattr_L__mod___layer3___21___conv2_drop(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_233 = self.getattr_L__mod___layer3___21___conv2_act0(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_234 = x_233.reshape((8, 2, 256, 16, 16));  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_140 = x_234.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_141 = x_gap_140.mean((2, 3), keepdim = True);  x_gap_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_142 = self.getattr_L__mod___layer3___21___conv2_fc1(x_gap_141);  x_gap_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_143 = self.getattr_L__mod___layer3___21___conv2_bn1(x_gap_142);  x_gap_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_144 = self.getattr_L__mod___layer3___21___conv2_act1(x_gap_143);  x_gap_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_56 = self.getattr_L__mod___layer3___21___conv2_fc2(x_gap_144);  x_gap_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_56 = x_attn_56.view(8, 1, 2, -1);  x_attn_56 = None
    x_235 = view_56.transpose(1, 2);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_236 = torch.nn.functional.softmax(x_235, dim = 1);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_237 = x_236.reshape(8, -1);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_57 = x_237.view(8, -1, 1, 1);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_86 = x_attn_57.reshape((8, 2, 256, 1, 1));  x_attn_57 = None
    mul_28 = x_234 * reshape_86;  x_234 = reshape_86 = None
    out_341 = mul_28.sum(dim = 1);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_342 = out_341.contiguous();  out_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_343 = self.getattr_L__mod___layer3___21___bn2(out_342);  out_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_344 = self.getattr_L__mod___layer3___21___drop_block(out_343);  out_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_345 = self.getattr_L__mod___layer3___21___act2(out_344);  out_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_346 = self.getattr_L__mod___layer3___21___conv3(out_345);  out_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_347 = self.getattr_L__mod___layer3___21___bn3(out_346);  out_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_347 += shortcut_31;  out_348 = out_347;  out_347 = shortcut_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_32 = self.getattr_L__mod___layer3___21___act3(out_348);  out_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_350 = self.getattr_L__mod___layer3___22___conv1(shortcut_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_351 = self.getattr_L__mod___layer3___22___bn1(out_350);  out_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_352 = self.getattr_L__mod___layer3___22___act1(out_351);  out_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_238 = self.getattr_L__mod___layer3___22___conv2_conv(out_352);  out_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_239 = self.getattr_L__mod___layer3___22___conv2_bn0(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_240 = self.getattr_L__mod___layer3___22___conv2_drop(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_241 = self.getattr_L__mod___layer3___22___conv2_act0(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_242 = x_241.reshape((8, 2, 256, 16, 16));  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_145 = x_242.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_146 = x_gap_145.mean((2, 3), keepdim = True);  x_gap_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_147 = self.getattr_L__mod___layer3___22___conv2_fc1(x_gap_146);  x_gap_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_148 = self.getattr_L__mod___layer3___22___conv2_bn1(x_gap_147);  x_gap_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_149 = self.getattr_L__mod___layer3___22___conv2_act1(x_gap_148);  x_gap_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_58 = self.getattr_L__mod___layer3___22___conv2_fc2(x_gap_149);  x_gap_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_58 = x_attn_58.view(8, 1, 2, -1);  x_attn_58 = None
    x_243 = view_58.transpose(1, 2);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_244 = torch.nn.functional.softmax(x_243, dim = 1);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_245 = x_244.reshape(8, -1);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_59 = x_245.view(8, -1, 1, 1);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_89 = x_attn_59.reshape((8, 2, 256, 1, 1));  x_attn_59 = None
    mul_29 = x_242 * reshape_89;  x_242 = reshape_89 = None
    out_353 = mul_29.sum(dim = 1);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_354 = out_353.contiguous();  out_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_355 = self.getattr_L__mod___layer3___22___bn2(out_354);  out_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_356 = self.getattr_L__mod___layer3___22___drop_block(out_355);  out_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_357 = self.getattr_L__mod___layer3___22___act2(out_356);  out_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_358 = self.getattr_L__mod___layer3___22___conv3(out_357);  out_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_359 = self.getattr_L__mod___layer3___22___bn3(out_358);  out_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_359 += shortcut_32;  out_360 = out_359;  out_359 = shortcut_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_33 = self.getattr_L__mod___layer3___22___act3(out_360);  out_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_362 = self.getattr_L__mod___layer4___0___conv1(shortcut_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_363 = self.getattr_L__mod___layer4___0___bn1(out_362);  out_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_364 = self.getattr_L__mod___layer4___0___act1(out_363);  out_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_247 = self.getattr_L__mod___layer4___0___conv2_conv(out_364);  out_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_248 = self.getattr_L__mod___layer4___0___conv2_bn0(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_249 = self.getattr_L__mod___layer4___0___conv2_drop(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_250 = self.getattr_L__mod___layer4___0___conv2_act0(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_251 = x_250.reshape((8, 2, 512, 16, 16));  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_150 = x_251.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_151 = x_gap_150.mean((2, 3), keepdim = True);  x_gap_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_152 = self.getattr_L__mod___layer4___0___conv2_fc1(x_gap_151);  x_gap_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_153 = self.getattr_L__mod___layer4___0___conv2_bn1(x_gap_152);  x_gap_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_154 = self.getattr_L__mod___layer4___0___conv2_act1(x_gap_153);  x_gap_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_60 = self.getattr_L__mod___layer4___0___conv2_fc2(x_gap_154);  x_gap_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_60 = x_attn_60.view(8, 1, 2, -1);  x_attn_60 = None
    x_252 = view_60.transpose(1, 2);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_253 = torch.nn.functional.softmax(x_252, dim = 1);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_254 = x_253.reshape(8, -1);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_61 = x_254.view(8, -1, 1, 1);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_92 = x_attn_61.reshape((8, 2, 512, 1, 1));  x_attn_61 = None
    mul_30 = x_251 * reshape_92;  x_251 = reshape_92 = None
    out_365 = mul_30.sum(dim = 1);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_366 = out_365.contiguous();  out_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_367 = self.getattr_L__mod___layer4___0___bn2(out_366);  out_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_368 = self.getattr_L__mod___layer4___0___drop_block(out_367);  out_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_369 = self.getattr_L__mod___layer4___0___act2(out_368);  out_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    out_370 = self.getattr_L__mod___layer4___0___avd_last(out_369);  out_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_371 = self.getattr_L__mod___layer4___0___conv3(out_370);  out_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_372 = self.getattr_L__mod___layer4___0___bn3(out_371);  out_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(shortcut_33);  shortcut_33 = None
    getattr_l__mod___layer4___0___downsample_1 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    shortcut_34 = self.getattr_L__mod___layer4___0___downsample_2(getattr_l__mod___layer4___0___downsample_1);  getattr_l__mod___layer4___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_372 += shortcut_34;  out_373 = out_372;  out_372 = shortcut_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_35 = self.getattr_L__mod___layer4___0___act3(out_373);  out_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_375 = self.getattr_L__mod___layer4___1___conv1(shortcut_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_376 = self.getattr_L__mod___layer4___1___bn1(out_375);  out_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_377 = self.getattr_L__mod___layer4___1___act1(out_376);  out_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_255 = self.getattr_L__mod___layer4___1___conv2_conv(out_377);  out_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_256 = self.getattr_L__mod___layer4___1___conv2_bn0(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_257 = self.getattr_L__mod___layer4___1___conv2_drop(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_258 = self.getattr_L__mod___layer4___1___conv2_act0(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_259 = x_258.reshape((8, 2, 512, 8, 8));  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_155 = x_259.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_156 = x_gap_155.mean((2, 3), keepdim = True);  x_gap_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_157 = self.getattr_L__mod___layer4___1___conv2_fc1(x_gap_156);  x_gap_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_158 = self.getattr_L__mod___layer4___1___conv2_bn1(x_gap_157);  x_gap_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_159 = self.getattr_L__mod___layer4___1___conv2_act1(x_gap_158);  x_gap_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_62 = self.getattr_L__mod___layer4___1___conv2_fc2(x_gap_159);  x_gap_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_62 = x_attn_62.view(8, 1, 2, -1);  x_attn_62 = None
    x_260 = view_62.transpose(1, 2);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_261 = torch.nn.functional.softmax(x_260, dim = 1);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_262 = x_261.reshape(8, -1);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_63 = x_262.view(8, -1, 1, 1);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_95 = x_attn_63.reshape((8, 2, 512, 1, 1));  x_attn_63 = None
    mul_31 = x_259 * reshape_95;  x_259 = reshape_95 = None
    out_378 = mul_31.sum(dim = 1);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_379 = out_378.contiguous();  out_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_380 = self.getattr_L__mod___layer4___1___bn2(out_379);  out_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_381 = self.getattr_L__mod___layer4___1___drop_block(out_380);  out_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_382 = self.getattr_L__mod___layer4___1___act2(out_381);  out_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_383 = self.getattr_L__mod___layer4___1___conv3(out_382);  out_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_384 = self.getattr_L__mod___layer4___1___bn3(out_383);  out_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_384 += shortcut_35;  out_385 = out_384;  out_384 = shortcut_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_36 = self.getattr_L__mod___layer4___1___act3(out_385);  out_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_387 = self.getattr_L__mod___layer4___2___conv1(shortcut_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_388 = self.getattr_L__mod___layer4___2___bn1(out_387);  out_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_389 = self.getattr_L__mod___layer4___2___act1(out_388);  out_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_263 = self.getattr_L__mod___layer4___2___conv2_conv(out_389);  out_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_264 = self.getattr_L__mod___layer4___2___conv2_bn0(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_265 = self.getattr_L__mod___layer4___2___conv2_drop(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_266 = self.getattr_L__mod___layer4___2___conv2_act0(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_267 = x_266.reshape((8, 2, 512, 8, 8));  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_160 = x_267.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_161 = x_gap_160.mean((2, 3), keepdim = True);  x_gap_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_162 = self.getattr_L__mod___layer4___2___conv2_fc1(x_gap_161);  x_gap_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_163 = self.getattr_L__mod___layer4___2___conv2_bn1(x_gap_162);  x_gap_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_164 = self.getattr_L__mod___layer4___2___conv2_act1(x_gap_163);  x_gap_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_64 = self.getattr_L__mod___layer4___2___conv2_fc2(x_gap_164);  x_gap_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_64 = x_attn_64.view(8, 1, 2, -1);  x_attn_64 = None
    x_268 = view_64.transpose(1, 2);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_269 = torch.nn.functional.softmax(x_268, dim = 1);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_270 = x_269.reshape(8, -1);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_65 = x_270.view(8, -1, 1, 1);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_98 = x_attn_65.reshape((8, 2, 512, 1, 1));  x_attn_65 = None
    mul_32 = x_267 * reshape_98;  x_267 = reshape_98 = None
    out_390 = mul_32.sum(dim = 1);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_391 = out_390.contiguous();  out_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_392 = self.getattr_L__mod___layer4___2___bn2(out_391);  out_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_393 = self.getattr_L__mod___layer4___2___drop_block(out_392);  out_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_394 = self.getattr_L__mod___layer4___2___act2(out_393);  out_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_395 = self.getattr_L__mod___layer4___2___conv3(out_394);  out_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_396 = self.getattr_L__mod___layer4___2___bn3(out_395);  out_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_396 += shortcut_36;  out_397 = out_396;  out_396 = shortcut_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    x_272 = self.getattr_L__mod___layer4___2___act3(out_397);  out_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_273 = self.L__mod___global_pool_pool(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_275 = self.L__mod___global_pool_flatten(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    x_276 = self.L__mod___fc(x_275);  x_275 = None
    return (x_276,)
    