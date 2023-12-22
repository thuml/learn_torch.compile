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
    x_8 = x_7.reshape((4, 2, 64, 56, 56));  x_7 = None
    
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
    view = x_attn.view(4, 1, 2, -1);  x_attn = None
    x_9 = view.transpose(1, 2);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_10 = torch.nn.functional.softmax(x_9, dim = 1);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_11 = x_10.reshape(4, -1);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_1 = x_11.view(4, -1, 1, 1);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_2 = x_attn_1.reshape((4, 2, 64, 1, 1));  x_attn_1 = None
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
    out_12 = self.getattr_L__mod___layer2___0___conv1(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_13 = self.getattr_L__mod___layer2___0___bn1(out_12);  out_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_14 = self.getattr_L__mod___layer2___0___act1(out_13);  out_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_13 = self.getattr_L__mod___layer2___0___conv2_conv(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_14 = self.getattr_L__mod___layer2___0___conv2_bn0(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_15 = self.getattr_L__mod___layer2___0___conv2_drop(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_16 = self.getattr_L__mod___layer2___0___conv2_act0(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_17 = x_16.reshape((4, 2, 128, 56, 56));  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_5 = x_17.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_6 = x_gap_5.mean((2, 3), keepdim = True);  x_gap_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_7 = self.getattr_L__mod___layer2___0___conv2_fc1(x_gap_6);  x_gap_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_8 = self.getattr_L__mod___layer2___0___conv2_bn1(x_gap_7);  x_gap_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_9 = self.getattr_L__mod___layer2___0___conv2_act1(x_gap_8);  x_gap_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_2 = self.getattr_L__mod___layer2___0___conv2_fc2(x_gap_9);  x_gap_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_2 = x_attn_2.view(4, 1, 2, -1);  x_attn_2 = None
    x_18 = view_2.transpose(1, 2);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_19 = torch.nn.functional.softmax(x_18, dim = 1);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_20 = x_19.reshape(4, -1);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_3 = x_20.view(4, -1, 1, 1);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_5 = x_attn_3.reshape((4, 2, 128, 1, 1));  x_attn_3 = None
    mul_1 = x_17 * reshape_5;  x_17 = reshape_5 = None
    out_15 = mul_1.sum(dim = 1);  mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_16 = out_15.contiguous();  out_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_17 = self.getattr_L__mod___layer2___0___bn2(out_16);  out_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_18 = self.getattr_L__mod___layer2___0___drop_block(out_17);  out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_19 = self.getattr_L__mod___layer2___0___act2(out_18);  out_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    out_20 = self.getattr_L__mod___layer2___0___avd_last(out_19);  out_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_21 = self.getattr_L__mod___layer2___0___conv3(out_20);  out_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_22 = self.getattr_L__mod___layer2___0___bn3(out_21);  out_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer2___0___downsample_0 = self.getattr_L__mod___layer2___0___downsample_0(shortcut_2);  shortcut_2 = None
    getattr_l__mod___layer2___0___downsample_1 = self.getattr_L__mod___layer2___0___downsample_1(getattr_l__mod___layer2___0___downsample_0);  getattr_l__mod___layer2___0___downsample_0 = None
    shortcut_3 = self.getattr_L__mod___layer2___0___downsample_2(getattr_l__mod___layer2___0___downsample_1);  getattr_l__mod___layer2___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_22 += shortcut_3;  out_23 = out_22;  out_22 = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_4 = self.getattr_L__mod___layer2___0___act3(out_23);  out_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_25 = self.getattr_L__mod___layer3___0___conv1(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_26 = self.getattr_L__mod___layer3___0___bn1(out_25);  out_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_27 = self.getattr_L__mod___layer3___0___act1(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_22 = self.getattr_L__mod___layer3___0___conv2_conv(out_27);  out_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_23 = self.getattr_L__mod___layer3___0___conv2_bn0(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_24 = self.getattr_L__mod___layer3___0___conv2_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_25 = self.getattr_L__mod___layer3___0___conv2_act0(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_26 = x_25.reshape((4, 2, 256, 28, 28));  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_10 = x_26.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_11 = x_gap_10.mean((2, 3), keepdim = True);  x_gap_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_12 = self.getattr_L__mod___layer3___0___conv2_fc1(x_gap_11);  x_gap_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_13 = self.getattr_L__mod___layer3___0___conv2_bn1(x_gap_12);  x_gap_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_14 = self.getattr_L__mod___layer3___0___conv2_act1(x_gap_13);  x_gap_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_4 = self.getattr_L__mod___layer3___0___conv2_fc2(x_gap_14);  x_gap_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_4 = x_attn_4.view(4, 1, 2, -1);  x_attn_4 = None
    x_27 = view_4.transpose(1, 2);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_28 = torch.nn.functional.softmax(x_27, dim = 1);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_29 = x_28.reshape(4, -1);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_5 = x_29.view(4, -1, 1, 1);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_8 = x_attn_5.reshape((4, 2, 256, 1, 1));  x_attn_5 = None
    mul_2 = x_26 * reshape_8;  x_26 = reshape_8 = None
    out_28 = mul_2.sum(dim = 1);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_29 = out_28.contiguous();  out_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_30 = self.getattr_L__mod___layer3___0___bn2(out_29);  out_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_31 = self.getattr_L__mod___layer3___0___drop_block(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_32 = self.getattr_L__mod___layer3___0___act2(out_31);  out_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    out_33 = self.getattr_L__mod___layer3___0___avd_last(out_32);  out_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_34 = self.getattr_L__mod___layer3___0___conv3(out_33);  out_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_35 = self.getattr_L__mod___layer3___0___bn3(out_34);  out_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer3___0___downsample_0 = self.getattr_L__mod___layer3___0___downsample_0(shortcut_4);  shortcut_4 = None
    getattr_l__mod___layer3___0___downsample_1 = self.getattr_L__mod___layer3___0___downsample_1(getattr_l__mod___layer3___0___downsample_0);  getattr_l__mod___layer3___0___downsample_0 = None
    shortcut_5 = self.getattr_L__mod___layer3___0___downsample_2(getattr_l__mod___layer3___0___downsample_1);  getattr_l__mod___layer3___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_35 += shortcut_5;  out_36 = out_35;  out_35 = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    shortcut_6 = self.getattr_L__mod___layer3___0___act3(out_36);  out_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    out_38 = self.getattr_L__mod___layer4___0___conv1(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    out_39 = self.getattr_L__mod___layer4___0___bn1(out_38);  out_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    out_40 = self.getattr_L__mod___layer4___0___act1(out_39);  out_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    x_31 = self.getattr_L__mod___layer4___0___conv2_conv(out_40);  out_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    x_32 = self.getattr_L__mod___layer4___0___conv2_bn0(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:64, code: x = self.drop(x)
    x_33 = self.getattr_L__mod___layer4___0___conv2_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    x_34 = self.getattr_L__mod___layer4___0___conv2_act0(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:69, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
    x_35 = x_34.reshape((4, 2, 512, 14, 14));  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    x_gap_15 = x_35.sum(dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    x_gap_16 = x_gap_15.mean((2, 3), keepdim = True);  x_gap_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    x_gap_17 = self.getattr_L__mod___layer4___0___conv2_fc1(x_gap_16);  x_gap_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    x_gap_18 = self.getattr_L__mod___layer4___0___conv2_bn1(x_gap_17);  x_gap_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    x_gap_19 = self.getattr_L__mod___layer4___0___conv2_act1(x_gap_18);  x_gap_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    x_attn_6 = self.getattr_L__mod___layer4___0___conv2_fc2(x_gap_19);  x_gap_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_6 = x_attn_6.view(4, 1, 2, -1);  x_attn_6 = None
    x_36 = view_6.transpose(1, 2);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    x_37 = torch.nn.functional.softmax(x_36, dim = 1);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    x_38 = x_37.reshape(4, -1);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    x_attn_7 = x_38.view(4, -1, 1, 1);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    reshape_11 = x_attn_7.reshape((4, 2, 512, 1, 1));  x_attn_7 = None
    mul_3 = x_35 * reshape_11;  x_35 = reshape_11 = None
    out_41 = mul_3.sum(dim = 1);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:84, code: return out.contiguous()
    out_42 = out_41.contiguous();  out_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:102, code: out = self.bn2(out)
    out_43 = self.getattr_L__mod___layer4___0___bn2(out_42);  out_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:103, code: out = self.drop_block(out)
    out_44 = self.getattr_L__mod___layer4___0___drop_block(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:104, code: out = self.act2(out)
    out_45 = self.getattr_L__mod___layer4___0___act2(out_44);  out_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    out_46 = self.getattr_L__mod___layer4___0___avd_last(out_45);  out_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    out_47 = self.getattr_L__mod___layer4___0___conv3(out_46);  out_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    out_48 = self.getattr_L__mod___layer4___0___bn3(out_47);  out_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(shortcut_6);  shortcut_6 = None
    getattr_l__mod___layer4___0___downsample_1 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    shortcut_7 = self.getattr_L__mod___layer4___0___downsample_2(getattr_l__mod___layer4___0___downsample_1);  getattr_l__mod___layer4___0___downsample_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    out_48 += shortcut_7;  out_49 = out_48;  out_48 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    x_40 = self.getattr_L__mod___layer4___0___act3(out_49);  out_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_41 = self.L__mod___global_pool_pool(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_43 = self.L__mod___global_pool_flatten(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    x_44 = self.L__mod___fc(x_43);  x_43 = None
    return (x_44,)
    