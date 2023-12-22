from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    x = self.L__mod___conv1(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    x_1 = self.L__mod___bn1(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    x_2 = self.L__mod___act1(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    shortcut = self.L__mod___maxpool(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out = self.getattr_L__mod___layer1___0___conv1(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_1 = self.getattr_L__mod___layer1___0___bn1(out);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_2 = self.getattr_L__mod___layer1___0___relu(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split = torch.functional.split(out_2, 26, 1);  out_2 = None
    sp = split[0]
    sp_4 = split[1]
    sp_8 = split[2]
    getitem_3 = split[3];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_1 = self.getattr_L__mod___layer1___0___convs_0(sp);  sp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_2 = self.getattr_L__mod___layer1___0___bns_0(sp_1);  sp_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_3 = self.getattr_L__mod___layer1___0___relu(sp_2);  sp_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_5 = self.getattr_L__mod___layer1___0___convs_1(sp_4);  sp_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_6 = self.getattr_L__mod___layer1___0___bns_1(sp_5);  sp_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_7 = self.getattr_L__mod___layer1___0___relu(sp_6);  sp_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_9 = self.getattr_L__mod___layer1___0___convs_2(sp_8);  sp_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_10 = self.getattr_L__mod___layer1___0___bns_2(sp_9);  sp_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_11 = self.getattr_L__mod___layer1___0___relu(sp_10);  sp_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer1___0___pool = self.getattr_L__mod___layer1___0___pool(getitem_3);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_3 = torch.cat([sp_3, sp_7, sp_11, getattr_l__mod___layer1___0___pool], 1);  sp_3 = sp_7 = sp_11 = getattr_l__mod___layer1___0___pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_4 = self.getattr_L__mod___layer1___0___conv3(out_3);  out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_5 = self.getattr_L__mod___layer1___0___bn3(out_4);  out_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    getattr_l__mod___layer1___0___downsample_0 = self.getattr_L__mod___layer1___0___downsample_0(shortcut);  shortcut = None
    shortcut_1 = self.getattr_L__mod___layer1___0___downsample_1(getattr_l__mod___layer1___0___downsample_0);  getattr_l__mod___layer1___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_5 += shortcut_1;  out_6 = out_5;  out_5 = shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_2 = self.getattr_L__mod___layer1___0___relu(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_8 = self.getattr_L__mod___layer1___1___conv1(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_9 = self.getattr_L__mod___layer1___1___bn1(out_8);  out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_10 = self.getattr_L__mod___layer1___1___relu(out_9);  out_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_1 = torch.functional.split(out_10, 26, 1);  out_10 = None
    sp_13 = split_1[0]
    getitem_5 = split_1[1]
    getitem_6 = split_1[2]
    getitem_7 = split_1[3];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_14 = self.getattr_L__mod___layer1___1___convs_0(sp_13);  sp_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_15 = self.getattr_L__mod___layer1___1___bns_0(sp_14);  sp_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_16 = self.getattr_L__mod___layer1___1___relu(sp_15);  sp_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_17 = sp_16 + getitem_5;  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_18 = self.getattr_L__mod___layer1___1___convs_1(sp_17);  sp_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_19 = self.getattr_L__mod___layer1___1___bns_1(sp_18);  sp_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_20 = self.getattr_L__mod___layer1___1___relu(sp_19);  sp_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_21 = sp_20 + getitem_6;  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_22 = self.getattr_L__mod___layer1___1___convs_2(sp_21);  sp_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_23 = self.getattr_L__mod___layer1___1___bns_2(sp_22);  sp_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_24 = self.getattr_L__mod___layer1___1___relu(sp_23);  sp_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_11 = torch.cat([sp_16, sp_20, sp_24, getitem_7], 1);  sp_16 = sp_20 = sp_24 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_12 = self.getattr_L__mod___layer1___1___conv3(out_11);  out_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_13 = self.getattr_L__mod___layer1___1___bn3(out_12);  out_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_13 += shortcut_2;  out_14 = out_13;  out_13 = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_3 = self.getattr_L__mod___layer1___1___relu(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_16 = self.getattr_L__mod___layer1___2___conv1(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_17 = self.getattr_L__mod___layer1___2___bn1(out_16);  out_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_18 = self.getattr_L__mod___layer1___2___relu(out_17);  out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_2 = torch.functional.split(out_18, 26, 1);  out_18 = None
    sp_26 = split_2[0]
    getitem_9 = split_2[1]
    getitem_10 = split_2[2]
    getitem_11 = split_2[3];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_27 = self.getattr_L__mod___layer1___2___convs_0(sp_26);  sp_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_28 = self.getattr_L__mod___layer1___2___bns_0(sp_27);  sp_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_29 = self.getattr_L__mod___layer1___2___relu(sp_28);  sp_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_30 = sp_29 + getitem_9;  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_31 = self.getattr_L__mod___layer1___2___convs_1(sp_30);  sp_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_32 = self.getattr_L__mod___layer1___2___bns_1(sp_31);  sp_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_33 = self.getattr_L__mod___layer1___2___relu(sp_32);  sp_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_34 = sp_33 + getitem_10;  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_35 = self.getattr_L__mod___layer1___2___convs_2(sp_34);  sp_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_36 = self.getattr_L__mod___layer1___2___bns_2(sp_35);  sp_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_37 = self.getattr_L__mod___layer1___2___relu(sp_36);  sp_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_19 = torch.cat([sp_29, sp_33, sp_37, getitem_11], 1);  sp_29 = sp_33 = sp_37 = getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_20 = self.getattr_L__mod___layer1___2___conv3(out_19);  out_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_21 = self.getattr_L__mod___layer1___2___bn3(out_20);  out_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_21 += shortcut_3;  out_22 = out_21;  out_21 = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_4 = self.getattr_L__mod___layer1___2___relu(out_22);  out_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_24 = self.getattr_L__mod___layer2___0___conv1(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_25 = self.getattr_L__mod___layer2___0___bn1(out_24);  out_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_26 = self.getattr_L__mod___layer2___0___relu(out_25);  out_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_3 = torch.functional.split(out_26, 52, 1);  out_26 = None
    sp_39 = split_3[0]
    sp_43 = split_3[1]
    sp_47 = split_3[2]
    getitem_15 = split_3[3];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_40 = self.getattr_L__mod___layer2___0___convs_0(sp_39);  sp_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_41 = self.getattr_L__mod___layer2___0___bns_0(sp_40);  sp_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_42 = self.getattr_L__mod___layer2___0___relu(sp_41);  sp_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_44 = self.getattr_L__mod___layer2___0___convs_1(sp_43);  sp_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_45 = self.getattr_L__mod___layer2___0___bns_1(sp_44);  sp_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_46 = self.getattr_L__mod___layer2___0___relu(sp_45);  sp_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_48 = self.getattr_L__mod___layer2___0___convs_2(sp_47);  sp_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_49 = self.getattr_L__mod___layer2___0___bns_2(sp_48);  sp_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_50 = self.getattr_L__mod___layer2___0___relu(sp_49);  sp_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer2___0___pool = self.getattr_L__mod___layer2___0___pool(getitem_15);  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_27 = torch.cat([sp_42, sp_46, sp_50, getattr_l__mod___layer2___0___pool], 1);  sp_42 = sp_46 = sp_50 = getattr_l__mod___layer2___0___pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_28 = self.getattr_L__mod___layer2___0___conv3(out_27);  out_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_29 = self.getattr_L__mod___layer2___0___bn3(out_28);  out_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    getattr_l__mod___layer2___0___downsample_0 = self.getattr_L__mod___layer2___0___downsample_0(shortcut_4);  shortcut_4 = None
    shortcut_5 = self.getattr_L__mod___layer2___0___downsample_1(getattr_l__mod___layer2___0___downsample_0);  getattr_l__mod___layer2___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_29 += shortcut_5;  out_30 = out_29;  out_29 = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_6 = self.getattr_L__mod___layer2___0___relu(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_32 = self.getattr_L__mod___layer2___1___conv1(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_33 = self.getattr_L__mod___layer2___1___bn1(out_32);  out_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_34 = self.getattr_L__mod___layer2___1___relu(out_33);  out_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_4 = torch.functional.split(out_34, 52, 1);  out_34 = None
    sp_52 = split_4[0]
    getitem_17 = split_4[1]
    getitem_18 = split_4[2]
    getitem_19 = split_4[3];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_53 = self.getattr_L__mod___layer2___1___convs_0(sp_52);  sp_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_54 = self.getattr_L__mod___layer2___1___bns_0(sp_53);  sp_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_55 = self.getattr_L__mod___layer2___1___relu(sp_54);  sp_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_56 = sp_55 + getitem_17;  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_57 = self.getattr_L__mod___layer2___1___convs_1(sp_56);  sp_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_58 = self.getattr_L__mod___layer2___1___bns_1(sp_57);  sp_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_59 = self.getattr_L__mod___layer2___1___relu(sp_58);  sp_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_60 = sp_59 + getitem_18;  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_61 = self.getattr_L__mod___layer2___1___convs_2(sp_60);  sp_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_62 = self.getattr_L__mod___layer2___1___bns_2(sp_61);  sp_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_63 = self.getattr_L__mod___layer2___1___relu(sp_62);  sp_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_35 = torch.cat([sp_55, sp_59, sp_63, getitem_19], 1);  sp_55 = sp_59 = sp_63 = getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_36 = self.getattr_L__mod___layer2___1___conv3(out_35);  out_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_37 = self.getattr_L__mod___layer2___1___bn3(out_36);  out_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_37 += shortcut_6;  out_38 = out_37;  out_37 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_7 = self.getattr_L__mod___layer2___1___relu(out_38);  out_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_40 = self.getattr_L__mod___layer2___2___conv1(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_41 = self.getattr_L__mod___layer2___2___bn1(out_40);  out_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_42 = self.getattr_L__mod___layer2___2___relu(out_41);  out_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_5 = torch.functional.split(out_42, 52, 1);  out_42 = None
    sp_65 = split_5[0]
    getitem_21 = split_5[1]
    getitem_22 = split_5[2]
    getitem_23 = split_5[3];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_66 = self.getattr_L__mod___layer2___2___convs_0(sp_65);  sp_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_67 = self.getattr_L__mod___layer2___2___bns_0(sp_66);  sp_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_68 = self.getattr_L__mod___layer2___2___relu(sp_67);  sp_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_69 = sp_68 + getitem_21;  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_70 = self.getattr_L__mod___layer2___2___convs_1(sp_69);  sp_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_71 = self.getattr_L__mod___layer2___2___bns_1(sp_70);  sp_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_72 = self.getattr_L__mod___layer2___2___relu(sp_71);  sp_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_73 = sp_72 + getitem_22;  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_74 = self.getattr_L__mod___layer2___2___convs_2(sp_73);  sp_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_75 = self.getattr_L__mod___layer2___2___bns_2(sp_74);  sp_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_76 = self.getattr_L__mod___layer2___2___relu(sp_75);  sp_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_43 = torch.cat([sp_68, sp_72, sp_76, getitem_23], 1);  sp_68 = sp_72 = sp_76 = getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_44 = self.getattr_L__mod___layer2___2___conv3(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_45 = self.getattr_L__mod___layer2___2___bn3(out_44);  out_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_45 += shortcut_7;  out_46 = out_45;  out_45 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_8 = self.getattr_L__mod___layer2___2___relu(out_46);  out_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_48 = self.getattr_L__mod___layer2___3___conv1(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_49 = self.getattr_L__mod___layer2___3___bn1(out_48);  out_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_50 = self.getattr_L__mod___layer2___3___relu(out_49);  out_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_6 = torch.functional.split(out_50, 52, 1);  out_50 = None
    sp_78 = split_6[0]
    getitem_25 = split_6[1]
    getitem_26 = split_6[2]
    getitem_27 = split_6[3];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_79 = self.getattr_L__mod___layer2___3___convs_0(sp_78);  sp_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_80 = self.getattr_L__mod___layer2___3___bns_0(sp_79);  sp_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_81 = self.getattr_L__mod___layer2___3___relu(sp_80);  sp_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_82 = sp_81 + getitem_25;  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_83 = self.getattr_L__mod___layer2___3___convs_1(sp_82);  sp_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_84 = self.getattr_L__mod___layer2___3___bns_1(sp_83);  sp_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_85 = self.getattr_L__mod___layer2___3___relu(sp_84);  sp_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_86 = sp_85 + getitem_26;  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_87 = self.getattr_L__mod___layer2___3___convs_2(sp_86);  sp_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_88 = self.getattr_L__mod___layer2___3___bns_2(sp_87);  sp_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_89 = self.getattr_L__mod___layer2___3___relu(sp_88);  sp_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_51 = torch.cat([sp_81, sp_85, sp_89, getitem_27], 1);  sp_81 = sp_85 = sp_89 = getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_52 = self.getattr_L__mod___layer2___3___conv3(out_51);  out_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_53 = self.getattr_L__mod___layer2___3___bn3(out_52);  out_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_53 += shortcut_8;  out_54 = out_53;  out_53 = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_9 = self.getattr_L__mod___layer2___3___relu(out_54);  out_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_56 = self.getattr_L__mod___layer3___0___conv1(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_57 = self.getattr_L__mod___layer3___0___bn1(out_56);  out_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_58 = self.getattr_L__mod___layer3___0___relu(out_57);  out_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_7 = torch.functional.split(out_58, 104, 1);  out_58 = None
    sp_91 = split_7[0]
    sp_95 = split_7[1]
    sp_99 = split_7[2]
    getitem_31 = split_7[3];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_92 = self.getattr_L__mod___layer3___0___convs_0(sp_91);  sp_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_93 = self.getattr_L__mod___layer3___0___bns_0(sp_92);  sp_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_94 = self.getattr_L__mod___layer3___0___relu(sp_93);  sp_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_96 = self.getattr_L__mod___layer3___0___convs_1(sp_95);  sp_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_97 = self.getattr_L__mod___layer3___0___bns_1(sp_96);  sp_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_98 = self.getattr_L__mod___layer3___0___relu(sp_97);  sp_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_100 = self.getattr_L__mod___layer3___0___convs_2(sp_99);  sp_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_101 = self.getattr_L__mod___layer3___0___bns_2(sp_100);  sp_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_102 = self.getattr_L__mod___layer3___0___relu(sp_101);  sp_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer3___0___pool = self.getattr_L__mod___layer3___0___pool(getitem_31);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_59 = torch.cat([sp_94, sp_98, sp_102, getattr_l__mod___layer3___0___pool], 1);  sp_94 = sp_98 = sp_102 = getattr_l__mod___layer3___0___pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_60 = self.getattr_L__mod___layer3___0___conv3(out_59);  out_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_61 = self.getattr_L__mod___layer3___0___bn3(out_60);  out_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    getattr_l__mod___layer3___0___downsample_0 = self.getattr_L__mod___layer3___0___downsample_0(shortcut_9);  shortcut_9 = None
    shortcut_10 = self.getattr_L__mod___layer3___0___downsample_1(getattr_l__mod___layer3___0___downsample_0);  getattr_l__mod___layer3___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_61 += shortcut_10;  out_62 = out_61;  out_61 = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_11 = self.getattr_L__mod___layer3___0___relu(out_62);  out_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_64 = self.getattr_L__mod___layer3___1___conv1(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_65 = self.getattr_L__mod___layer3___1___bn1(out_64);  out_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_66 = self.getattr_L__mod___layer3___1___relu(out_65);  out_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_8 = torch.functional.split(out_66, 104, 1);  out_66 = None
    sp_104 = split_8[0]
    getitem_33 = split_8[1]
    getitem_34 = split_8[2]
    getitem_35 = split_8[3];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_105 = self.getattr_L__mod___layer3___1___convs_0(sp_104);  sp_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_106 = self.getattr_L__mod___layer3___1___bns_0(sp_105);  sp_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_107 = self.getattr_L__mod___layer3___1___relu(sp_106);  sp_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_108 = sp_107 + getitem_33;  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_109 = self.getattr_L__mod___layer3___1___convs_1(sp_108);  sp_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_110 = self.getattr_L__mod___layer3___1___bns_1(sp_109);  sp_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_111 = self.getattr_L__mod___layer3___1___relu(sp_110);  sp_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_112 = sp_111 + getitem_34;  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_113 = self.getattr_L__mod___layer3___1___convs_2(sp_112);  sp_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_114 = self.getattr_L__mod___layer3___1___bns_2(sp_113);  sp_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_115 = self.getattr_L__mod___layer3___1___relu(sp_114);  sp_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_67 = torch.cat([sp_107, sp_111, sp_115, getitem_35], 1);  sp_107 = sp_111 = sp_115 = getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_68 = self.getattr_L__mod___layer3___1___conv3(out_67);  out_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_69 = self.getattr_L__mod___layer3___1___bn3(out_68);  out_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_69 += shortcut_11;  out_70 = out_69;  out_69 = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_12 = self.getattr_L__mod___layer3___1___relu(out_70);  out_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_72 = self.getattr_L__mod___layer3___2___conv1(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_73 = self.getattr_L__mod___layer3___2___bn1(out_72);  out_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_74 = self.getattr_L__mod___layer3___2___relu(out_73);  out_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_9 = torch.functional.split(out_74, 104, 1);  out_74 = None
    sp_117 = split_9[0]
    getitem_37 = split_9[1]
    getitem_38 = split_9[2]
    getitem_39 = split_9[3];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_118 = self.getattr_L__mod___layer3___2___convs_0(sp_117);  sp_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_119 = self.getattr_L__mod___layer3___2___bns_0(sp_118);  sp_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_120 = self.getattr_L__mod___layer3___2___relu(sp_119);  sp_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_121 = sp_120 + getitem_37;  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_122 = self.getattr_L__mod___layer3___2___convs_1(sp_121);  sp_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_123 = self.getattr_L__mod___layer3___2___bns_1(sp_122);  sp_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_124 = self.getattr_L__mod___layer3___2___relu(sp_123);  sp_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_125 = sp_124 + getitem_38;  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_126 = self.getattr_L__mod___layer3___2___convs_2(sp_125);  sp_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_127 = self.getattr_L__mod___layer3___2___bns_2(sp_126);  sp_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_128 = self.getattr_L__mod___layer3___2___relu(sp_127);  sp_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_75 = torch.cat([sp_120, sp_124, sp_128, getitem_39], 1);  sp_120 = sp_124 = sp_128 = getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_76 = self.getattr_L__mod___layer3___2___conv3(out_75);  out_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_77 = self.getattr_L__mod___layer3___2___bn3(out_76);  out_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_77 += shortcut_12;  out_78 = out_77;  out_77 = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_13 = self.getattr_L__mod___layer3___2___relu(out_78);  out_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_80 = self.getattr_L__mod___layer3___3___conv1(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_81 = self.getattr_L__mod___layer3___3___bn1(out_80);  out_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_82 = self.getattr_L__mod___layer3___3___relu(out_81);  out_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_10 = torch.functional.split(out_82, 104, 1);  out_82 = None
    sp_130 = split_10[0]
    getitem_41 = split_10[1]
    getitem_42 = split_10[2]
    getitem_43 = split_10[3];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_131 = self.getattr_L__mod___layer3___3___convs_0(sp_130);  sp_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_132 = self.getattr_L__mod___layer3___3___bns_0(sp_131);  sp_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_133 = self.getattr_L__mod___layer3___3___relu(sp_132);  sp_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_134 = sp_133 + getitem_41;  getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_135 = self.getattr_L__mod___layer3___3___convs_1(sp_134);  sp_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_136 = self.getattr_L__mod___layer3___3___bns_1(sp_135);  sp_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_137 = self.getattr_L__mod___layer3___3___relu(sp_136);  sp_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_138 = sp_137 + getitem_42;  getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_139 = self.getattr_L__mod___layer3___3___convs_2(sp_138);  sp_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_140 = self.getattr_L__mod___layer3___3___bns_2(sp_139);  sp_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_141 = self.getattr_L__mod___layer3___3___relu(sp_140);  sp_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_83 = torch.cat([sp_133, sp_137, sp_141, getitem_43], 1);  sp_133 = sp_137 = sp_141 = getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_84 = self.getattr_L__mod___layer3___3___conv3(out_83);  out_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_85 = self.getattr_L__mod___layer3___3___bn3(out_84);  out_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_85 += shortcut_13;  out_86 = out_85;  out_85 = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_14 = self.getattr_L__mod___layer3___3___relu(out_86);  out_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_88 = self.getattr_L__mod___layer3___4___conv1(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_89 = self.getattr_L__mod___layer3___4___bn1(out_88);  out_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_90 = self.getattr_L__mod___layer3___4___relu(out_89);  out_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_11 = torch.functional.split(out_90, 104, 1);  out_90 = None
    sp_143 = split_11[0]
    getitem_45 = split_11[1]
    getitem_46 = split_11[2]
    getitem_47 = split_11[3];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_144 = self.getattr_L__mod___layer3___4___convs_0(sp_143);  sp_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_145 = self.getattr_L__mod___layer3___4___bns_0(sp_144);  sp_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_146 = self.getattr_L__mod___layer3___4___relu(sp_145);  sp_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_147 = sp_146 + getitem_45;  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_148 = self.getattr_L__mod___layer3___4___convs_1(sp_147);  sp_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_149 = self.getattr_L__mod___layer3___4___bns_1(sp_148);  sp_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_150 = self.getattr_L__mod___layer3___4___relu(sp_149);  sp_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_151 = sp_150 + getitem_46;  getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_152 = self.getattr_L__mod___layer3___4___convs_2(sp_151);  sp_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_153 = self.getattr_L__mod___layer3___4___bns_2(sp_152);  sp_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_154 = self.getattr_L__mod___layer3___4___relu(sp_153);  sp_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_91 = torch.cat([sp_146, sp_150, sp_154, getitem_47], 1);  sp_146 = sp_150 = sp_154 = getitem_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_92 = self.getattr_L__mod___layer3___4___conv3(out_91);  out_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_93 = self.getattr_L__mod___layer3___4___bn3(out_92);  out_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_93 += shortcut_14;  out_94 = out_93;  out_93 = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_15 = self.getattr_L__mod___layer3___4___relu(out_94);  out_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_96 = self.getattr_L__mod___layer3___5___conv1(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_97 = self.getattr_L__mod___layer3___5___bn1(out_96);  out_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_98 = self.getattr_L__mod___layer3___5___relu(out_97);  out_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_12 = torch.functional.split(out_98, 104, 1);  out_98 = None
    sp_156 = split_12[0]
    getitem_49 = split_12[1]
    getitem_50 = split_12[2]
    getitem_51 = split_12[3];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_157 = self.getattr_L__mod___layer3___5___convs_0(sp_156);  sp_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_158 = self.getattr_L__mod___layer3___5___bns_0(sp_157);  sp_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_159 = self.getattr_L__mod___layer3___5___relu(sp_158);  sp_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_160 = sp_159 + getitem_49;  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_161 = self.getattr_L__mod___layer3___5___convs_1(sp_160);  sp_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_162 = self.getattr_L__mod___layer3___5___bns_1(sp_161);  sp_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_163 = self.getattr_L__mod___layer3___5___relu(sp_162);  sp_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_164 = sp_163 + getitem_50;  getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_165 = self.getattr_L__mod___layer3___5___convs_2(sp_164);  sp_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_166 = self.getattr_L__mod___layer3___5___bns_2(sp_165);  sp_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_167 = self.getattr_L__mod___layer3___5___relu(sp_166);  sp_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_99 = torch.cat([sp_159, sp_163, sp_167, getitem_51], 1);  sp_159 = sp_163 = sp_167 = getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_100 = self.getattr_L__mod___layer3___5___conv3(out_99);  out_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_101 = self.getattr_L__mod___layer3___5___bn3(out_100);  out_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_101 += shortcut_15;  out_102 = out_101;  out_101 = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_16 = self.getattr_L__mod___layer3___5___relu(out_102);  out_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_104 = self.getattr_L__mod___layer3___6___conv1(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_105 = self.getattr_L__mod___layer3___6___bn1(out_104);  out_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_106 = self.getattr_L__mod___layer3___6___relu(out_105);  out_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_13 = torch.functional.split(out_106, 104, 1);  out_106 = None
    sp_169 = split_13[0]
    getitem_53 = split_13[1]
    getitem_54 = split_13[2]
    getitem_55 = split_13[3];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_170 = self.getattr_L__mod___layer3___6___convs_0(sp_169);  sp_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_171 = self.getattr_L__mod___layer3___6___bns_0(sp_170);  sp_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_172 = self.getattr_L__mod___layer3___6___relu(sp_171);  sp_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_173 = sp_172 + getitem_53;  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_174 = self.getattr_L__mod___layer3___6___convs_1(sp_173);  sp_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_175 = self.getattr_L__mod___layer3___6___bns_1(sp_174);  sp_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_176 = self.getattr_L__mod___layer3___6___relu(sp_175);  sp_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_177 = sp_176 + getitem_54;  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_178 = self.getattr_L__mod___layer3___6___convs_2(sp_177);  sp_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_179 = self.getattr_L__mod___layer3___6___bns_2(sp_178);  sp_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_180 = self.getattr_L__mod___layer3___6___relu(sp_179);  sp_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_107 = torch.cat([sp_172, sp_176, sp_180, getitem_55], 1);  sp_172 = sp_176 = sp_180 = getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_108 = self.getattr_L__mod___layer3___6___conv3(out_107);  out_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_109 = self.getattr_L__mod___layer3___6___bn3(out_108);  out_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_109 += shortcut_16;  out_110 = out_109;  out_109 = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_17 = self.getattr_L__mod___layer3___6___relu(out_110);  out_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_112 = self.getattr_L__mod___layer3___7___conv1(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_113 = self.getattr_L__mod___layer3___7___bn1(out_112);  out_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_114 = self.getattr_L__mod___layer3___7___relu(out_113);  out_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_14 = torch.functional.split(out_114, 104, 1);  out_114 = None
    sp_182 = split_14[0]
    getitem_57 = split_14[1]
    getitem_58 = split_14[2]
    getitem_59 = split_14[3];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_183 = self.getattr_L__mod___layer3___7___convs_0(sp_182);  sp_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_184 = self.getattr_L__mod___layer3___7___bns_0(sp_183);  sp_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_185 = self.getattr_L__mod___layer3___7___relu(sp_184);  sp_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_186 = sp_185 + getitem_57;  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_187 = self.getattr_L__mod___layer3___7___convs_1(sp_186);  sp_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_188 = self.getattr_L__mod___layer3___7___bns_1(sp_187);  sp_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_189 = self.getattr_L__mod___layer3___7___relu(sp_188);  sp_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_190 = sp_189 + getitem_58;  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_191 = self.getattr_L__mod___layer3___7___convs_2(sp_190);  sp_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_192 = self.getattr_L__mod___layer3___7___bns_2(sp_191);  sp_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_193 = self.getattr_L__mod___layer3___7___relu(sp_192);  sp_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_115 = torch.cat([sp_185, sp_189, sp_193, getitem_59], 1);  sp_185 = sp_189 = sp_193 = getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_116 = self.getattr_L__mod___layer3___7___conv3(out_115);  out_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_117 = self.getattr_L__mod___layer3___7___bn3(out_116);  out_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_117 += shortcut_17;  out_118 = out_117;  out_117 = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_18 = self.getattr_L__mod___layer3___7___relu(out_118);  out_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_120 = self.getattr_L__mod___layer3___8___conv1(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_121 = self.getattr_L__mod___layer3___8___bn1(out_120);  out_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_122 = self.getattr_L__mod___layer3___8___relu(out_121);  out_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_15 = torch.functional.split(out_122, 104, 1);  out_122 = None
    sp_195 = split_15[0]
    getitem_61 = split_15[1]
    getitem_62 = split_15[2]
    getitem_63 = split_15[3];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_196 = self.getattr_L__mod___layer3___8___convs_0(sp_195);  sp_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_197 = self.getattr_L__mod___layer3___8___bns_0(sp_196);  sp_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_198 = self.getattr_L__mod___layer3___8___relu(sp_197);  sp_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_199 = sp_198 + getitem_61;  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_200 = self.getattr_L__mod___layer3___8___convs_1(sp_199);  sp_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_201 = self.getattr_L__mod___layer3___8___bns_1(sp_200);  sp_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_202 = self.getattr_L__mod___layer3___8___relu(sp_201);  sp_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_203 = sp_202 + getitem_62;  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_204 = self.getattr_L__mod___layer3___8___convs_2(sp_203);  sp_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_205 = self.getattr_L__mod___layer3___8___bns_2(sp_204);  sp_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_206 = self.getattr_L__mod___layer3___8___relu(sp_205);  sp_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_123 = torch.cat([sp_198, sp_202, sp_206, getitem_63], 1);  sp_198 = sp_202 = sp_206 = getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_124 = self.getattr_L__mod___layer3___8___conv3(out_123);  out_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_125 = self.getattr_L__mod___layer3___8___bn3(out_124);  out_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_125 += shortcut_18;  out_126 = out_125;  out_125 = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_19 = self.getattr_L__mod___layer3___8___relu(out_126);  out_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_128 = self.getattr_L__mod___layer3___9___conv1(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_129 = self.getattr_L__mod___layer3___9___bn1(out_128);  out_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_130 = self.getattr_L__mod___layer3___9___relu(out_129);  out_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_16 = torch.functional.split(out_130, 104, 1);  out_130 = None
    sp_208 = split_16[0]
    getitem_65 = split_16[1]
    getitem_66 = split_16[2]
    getitem_67 = split_16[3];  split_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_209 = self.getattr_L__mod___layer3___9___convs_0(sp_208);  sp_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_210 = self.getattr_L__mod___layer3___9___bns_0(sp_209);  sp_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_211 = self.getattr_L__mod___layer3___9___relu(sp_210);  sp_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_212 = sp_211 + getitem_65;  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_213 = self.getattr_L__mod___layer3___9___convs_1(sp_212);  sp_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_214 = self.getattr_L__mod___layer3___9___bns_1(sp_213);  sp_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_215 = self.getattr_L__mod___layer3___9___relu(sp_214);  sp_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_216 = sp_215 + getitem_66;  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_217 = self.getattr_L__mod___layer3___9___convs_2(sp_216);  sp_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_218 = self.getattr_L__mod___layer3___9___bns_2(sp_217);  sp_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_219 = self.getattr_L__mod___layer3___9___relu(sp_218);  sp_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_131 = torch.cat([sp_211, sp_215, sp_219, getitem_67], 1);  sp_211 = sp_215 = sp_219 = getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_132 = self.getattr_L__mod___layer3___9___conv3(out_131);  out_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_133 = self.getattr_L__mod___layer3___9___bn3(out_132);  out_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_133 += shortcut_19;  out_134 = out_133;  out_133 = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_20 = self.getattr_L__mod___layer3___9___relu(out_134);  out_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_136 = self.getattr_L__mod___layer3___10___conv1(shortcut_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_137 = self.getattr_L__mod___layer3___10___bn1(out_136);  out_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_138 = self.getattr_L__mod___layer3___10___relu(out_137);  out_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_17 = torch.functional.split(out_138, 104, 1);  out_138 = None
    sp_221 = split_17[0]
    getitem_69 = split_17[1]
    getitem_70 = split_17[2]
    getitem_71 = split_17[3];  split_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_222 = self.getattr_L__mod___layer3___10___convs_0(sp_221);  sp_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_223 = self.getattr_L__mod___layer3___10___bns_0(sp_222);  sp_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_224 = self.getattr_L__mod___layer3___10___relu(sp_223);  sp_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_225 = sp_224 + getitem_69;  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_226 = self.getattr_L__mod___layer3___10___convs_1(sp_225);  sp_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_227 = self.getattr_L__mod___layer3___10___bns_1(sp_226);  sp_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_228 = self.getattr_L__mod___layer3___10___relu(sp_227);  sp_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_229 = sp_228 + getitem_70;  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_230 = self.getattr_L__mod___layer3___10___convs_2(sp_229);  sp_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_231 = self.getattr_L__mod___layer3___10___bns_2(sp_230);  sp_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_232 = self.getattr_L__mod___layer3___10___relu(sp_231);  sp_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_139 = torch.cat([sp_224, sp_228, sp_232, getitem_71], 1);  sp_224 = sp_228 = sp_232 = getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_140 = self.getattr_L__mod___layer3___10___conv3(out_139);  out_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_141 = self.getattr_L__mod___layer3___10___bn3(out_140);  out_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_141 += shortcut_20;  out_142 = out_141;  out_141 = shortcut_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_21 = self.getattr_L__mod___layer3___10___relu(out_142);  out_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_144 = self.getattr_L__mod___layer3___11___conv1(shortcut_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_145 = self.getattr_L__mod___layer3___11___bn1(out_144);  out_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_146 = self.getattr_L__mod___layer3___11___relu(out_145);  out_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_18 = torch.functional.split(out_146, 104, 1);  out_146 = None
    sp_234 = split_18[0]
    getitem_73 = split_18[1]
    getitem_74 = split_18[2]
    getitem_75 = split_18[3];  split_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_235 = self.getattr_L__mod___layer3___11___convs_0(sp_234);  sp_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_236 = self.getattr_L__mod___layer3___11___bns_0(sp_235);  sp_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_237 = self.getattr_L__mod___layer3___11___relu(sp_236);  sp_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_238 = sp_237 + getitem_73;  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_239 = self.getattr_L__mod___layer3___11___convs_1(sp_238);  sp_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_240 = self.getattr_L__mod___layer3___11___bns_1(sp_239);  sp_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_241 = self.getattr_L__mod___layer3___11___relu(sp_240);  sp_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_242 = sp_241 + getitem_74;  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_243 = self.getattr_L__mod___layer3___11___convs_2(sp_242);  sp_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_244 = self.getattr_L__mod___layer3___11___bns_2(sp_243);  sp_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_245 = self.getattr_L__mod___layer3___11___relu(sp_244);  sp_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_147 = torch.cat([sp_237, sp_241, sp_245, getitem_75], 1);  sp_237 = sp_241 = sp_245 = getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_148 = self.getattr_L__mod___layer3___11___conv3(out_147);  out_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_149 = self.getattr_L__mod___layer3___11___bn3(out_148);  out_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_149 += shortcut_21;  out_150 = out_149;  out_149 = shortcut_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_22 = self.getattr_L__mod___layer3___11___relu(out_150);  out_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_152 = self.getattr_L__mod___layer3___12___conv1(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_153 = self.getattr_L__mod___layer3___12___bn1(out_152);  out_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_154 = self.getattr_L__mod___layer3___12___relu(out_153);  out_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_19 = torch.functional.split(out_154, 104, 1);  out_154 = None
    sp_247 = split_19[0]
    getitem_77 = split_19[1]
    getitem_78 = split_19[2]
    getitem_79 = split_19[3];  split_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_248 = self.getattr_L__mod___layer3___12___convs_0(sp_247);  sp_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_249 = self.getattr_L__mod___layer3___12___bns_0(sp_248);  sp_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_250 = self.getattr_L__mod___layer3___12___relu(sp_249);  sp_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_251 = sp_250 + getitem_77;  getitem_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_252 = self.getattr_L__mod___layer3___12___convs_1(sp_251);  sp_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_253 = self.getattr_L__mod___layer3___12___bns_1(sp_252);  sp_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_254 = self.getattr_L__mod___layer3___12___relu(sp_253);  sp_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_255 = sp_254 + getitem_78;  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_256 = self.getattr_L__mod___layer3___12___convs_2(sp_255);  sp_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_257 = self.getattr_L__mod___layer3___12___bns_2(sp_256);  sp_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_258 = self.getattr_L__mod___layer3___12___relu(sp_257);  sp_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_155 = torch.cat([sp_250, sp_254, sp_258, getitem_79], 1);  sp_250 = sp_254 = sp_258 = getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_156 = self.getattr_L__mod___layer3___12___conv3(out_155);  out_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_157 = self.getattr_L__mod___layer3___12___bn3(out_156);  out_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_157 += shortcut_22;  out_158 = out_157;  out_157 = shortcut_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_23 = self.getattr_L__mod___layer3___12___relu(out_158);  out_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_160 = self.getattr_L__mod___layer3___13___conv1(shortcut_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_161 = self.getattr_L__mod___layer3___13___bn1(out_160);  out_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_162 = self.getattr_L__mod___layer3___13___relu(out_161);  out_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_20 = torch.functional.split(out_162, 104, 1);  out_162 = None
    sp_260 = split_20[0]
    getitem_81 = split_20[1]
    getitem_82 = split_20[2]
    getitem_83 = split_20[3];  split_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_261 = self.getattr_L__mod___layer3___13___convs_0(sp_260);  sp_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_262 = self.getattr_L__mod___layer3___13___bns_0(sp_261);  sp_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_263 = self.getattr_L__mod___layer3___13___relu(sp_262);  sp_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_264 = sp_263 + getitem_81;  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_265 = self.getattr_L__mod___layer3___13___convs_1(sp_264);  sp_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_266 = self.getattr_L__mod___layer3___13___bns_1(sp_265);  sp_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_267 = self.getattr_L__mod___layer3___13___relu(sp_266);  sp_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_268 = sp_267 + getitem_82;  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_269 = self.getattr_L__mod___layer3___13___convs_2(sp_268);  sp_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_270 = self.getattr_L__mod___layer3___13___bns_2(sp_269);  sp_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_271 = self.getattr_L__mod___layer3___13___relu(sp_270);  sp_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_163 = torch.cat([sp_263, sp_267, sp_271, getitem_83], 1);  sp_263 = sp_267 = sp_271 = getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_164 = self.getattr_L__mod___layer3___13___conv3(out_163);  out_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_165 = self.getattr_L__mod___layer3___13___bn3(out_164);  out_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_165 += shortcut_23;  out_166 = out_165;  out_165 = shortcut_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_24 = self.getattr_L__mod___layer3___13___relu(out_166);  out_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_168 = self.getattr_L__mod___layer3___14___conv1(shortcut_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_169 = self.getattr_L__mod___layer3___14___bn1(out_168);  out_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_170 = self.getattr_L__mod___layer3___14___relu(out_169);  out_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_21 = torch.functional.split(out_170, 104, 1);  out_170 = None
    sp_273 = split_21[0]
    getitem_85 = split_21[1]
    getitem_86 = split_21[2]
    getitem_87 = split_21[3];  split_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_274 = self.getattr_L__mod___layer3___14___convs_0(sp_273);  sp_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_275 = self.getattr_L__mod___layer3___14___bns_0(sp_274);  sp_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_276 = self.getattr_L__mod___layer3___14___relu(sp_275);  sp_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_277 = sp_276 + getitem_85;  getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_278 = self.getattr_L__mod___layer3___14___convs_1(sp_277);  sp_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_279 = self.getattr_L__mod___layer3___14___bns_1(sp_278);  sp_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_280 = self.getattr_L__mod___layer3___14___relu(sp_279);  sp_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_281 = sp_280 + getitem_86;  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_282 = self.getattr_L__mod___layer3___14___convs_2(sp_281);  sp_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_283 = self.getattr_L__mod___layer3___14___bns_2(sp_282);  sp_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_284 = self.getattr_L__mod___layer3___14___relu(sp_283);  sp_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_171 = torch.cat([sp_276, sp_280, sp_284, getitem_87], 1);  sp_276 = sp_280 = sp_284 = getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_172 = self.getattr_L__mod___layer3___14___conv3(out_171);  out_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_173 = self.getattr_L__mod___layer3___14___bn3(out_172);  out_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_173 += shortcut_24;  out_174 = out_173;  out_173 = shortcut_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_25 = self.getattr_L__mod___layer3___14___relu(out_174);  out_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_176 = self.getattr_L__mod___layer3___15___conv1(shortcut_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_177 = self.getattr_L__mod___layer3___15___bn1(out_176);  out_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_178 = self.getattr_L__mod___layer3___15___relu(out_177);  out_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_22 = torch.functional.split(out_178, 104, 1);  out_178 = None
    sp_286 = split_22[0]
    getitem_89 = split_22[1]
    getitem_90 = split_22[2]
    getitem_91 = split_22[3];  split_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_287 = self.getattr_L__mod___layer3___15___convs_0(sp_286);  sp_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_288 = self.getattr_L__mod___layer3___15___bns_0(sp_287);  sp_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_289 = self.getattr_L__mod___layer3___15___relu(sp_288);  sp_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_290 = sp_289 + getitem_89;  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_291 = self.getattr_L__mod___layer3___15___convs_1(sp_290);  sp_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_292 = self.getattr_L__mod___layer3___15___bns_1(sp_291);  sp_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_293 = self.getattr_L__mod___layer3___15___relu(sp_292);  sp_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_294 = sp_293 + getitem_90;  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_295 = self.getattr_L__mod___layer3___15___convs_2(sp_294);  sp_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_296 = self.getattr_L__mod___layer3___15___bns_2(sp_295);  sp_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_297 = self.getattr_L__mod___layer3___15___relu(sp_296);  sp_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_179 = torch.cat([sp_289, sp_293, sp_297, getitem_91], 1);  sp_289 = sp_293 = sp_297 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_180 = self.getattr_L__mod___layer3___15___conv3(out_179);  out_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_181 = self.getattr_L__mod___layer3___15___bn3(out_180);  out_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_181 += shortcut_25;  out_182 = out_181;  out_181 = shortcut_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_26 = self.getattr_L__mod___layer3___15___relu(out_182);  out_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_184 = self.getattr_L__mod___layer3___16___conv1(shortcut_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_185 = self.getattr_L__mod___layer3___16___bn1(out_184);  out_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_186 = self.getattr_L__mod___layer3___16___relu(out_185);  out_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_23 = torch.functional.split(out_186, 104, 1);  out_186 = None
    sp_299 = split_23[0]
    getitem_93 = split_23[1]
    getitem_94 = split_23[2]
    getitem_95 = split_23[3];  split_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_300 = self.getattr_L__mod___layer3___16___convs_0(sp_299);  sp_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_301 = self.getattr_L__mod___layer3___16___bns_0(sp_300);  sp_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_302 = self.getattr_L__mod___layer3___16___relu(sp_301);  sp_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_303 = sp_302 + getitem_93;  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_304 = self.getattr_L__mod___layer3___16___convs_1(sp_303);  sp_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_305 = self.getattr_L__mod___layer3___16___bns_1(sp_304);  sp_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_306 = self.getattr_L__mod___layer3___16___relu(sp_305);  sp_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_307 = sp_306 + getitem_94;  getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_308 = self.getattr_L__mod___layer3___16___convs_2(sp_307);  sp_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_309 = self.getattr_L__mod___layer3___16___bns_2(sp_308);  sp_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_310 = self.getattr_L__mod___layer3___16___relu(sp_309);  sp_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_187 = torch.cat([sp_302, sp_306, sp_310, getitem_95], 1);  sp_302 = sp_306 = sp_310 = getitem_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_188 = self.getattr_L__mod___layer3___16___conv3(out_187);  out_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_189 = self.getattr_L__mod___layer3___16___bn3(out_188);  out_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_189 += shortcut_26;  out_190 = out_189;  out_189 = shortcut_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_27 = self.getattr_L__mod___layer3___16___relu(out_190);  out_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_192 = self.getattr_L__mod___layer3___17___conv1(shortcut_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_193 = self.getattr_L__mod___layer3___17___bn1(out_192);  out_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_194 = self.getattr_L__mod___layer3___17___relu(out_193);  out_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_24 = torch.functional.split(out_194, 104, 1);  out_194 = None
    sp_312 = split_24[0]
    getitem_97 = split_24[1]
    getitem_98 = split_24[2]
    getitem_99 = split_24[3];  split_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_313 = self.getattr_L__mod___layer3___17___convs_0(sp_312);  sp_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_314 = self.getattr_L__mod___layer3___17___bns_0(sp_313);  sp_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_315 = self.getattr_L__mod___layer3___17___relu(sp_314);  sp_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_316 = sp_315 + getitem_97;  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_317 = self.getattr_L__mod___layer3___17___convs_1(sp_316);  sp_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_318 = self.getattr_L__mod___layer3___17___bns_1(sp_317);  sp_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_319 = self.getattr_L__mod___layer3___17___relu(sp_318);  sp_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_320 = sp_319 + getitem_98;  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_321 = self.getattr_L__mod___layer3___17___convs_2(sp_320);  sp_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_322 = self.getattr_L__mod___layer3___17___bns_2(sp_321);  sp_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_323 = self.getattr_L__mod___layer3___17___relu(sp_322);  sp_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_195 = torch.cat([sp_315, sp_319, sp_323, getitem_99], 1);  sp_315 = sp_319 = sp_323 = getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_196 = self.getattr_L__mod___layer3___17___conv3(out_195);  out_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_197 = self.getattr_L__mod___layer3___17___bn3(out_196);  out_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_197 += shortcut_27;  out_198 = out_197;  out_197 = shortcut_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_28 = self.getattr_L__mod___layer3___17___relu(out_198);  out_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_200 = self.getattr_L__mod___layer3___18___conv1(shortcut_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_201 = self.getattr_L__mod___layer3___18___bn1(out_200);  out_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_202 = self.getattr_L__mod___layer3___18___relu(out_201);  out_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_25 = torch.functional.split(out_202, 104, 1);  out_202 = None
    sp_325 = split_25[0]
    getitem_101 = split_25[1]
    getitem_102 = split_25[2]
    getitem_103 = split_25[3];  split_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_326 = self.getattr_L__mod___layer3___18___convs_0(sp_325);  sp_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_327 = self.getattr_L__mod___layer3___18___bns_0(sp_326);  sp_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_328 = self.getattr_L__mod___layer3___18___relu(sp_327);  sp_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_329 = sp_328 + getitem_101;  getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_330 = self.getattr_L__mod___layer3___18___convs_1(sp_329);  sp_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_331 = self.getattr_L__mod___layer3___18___bns_1(sp_330);  sp_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_332 = self.getattr_L__mod___layer3___18___relu(sp_331);  sp_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_333 = sp_332 + getitem_102;  getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_334 = self.getattr_L__mod___layer3___18___convs_2(sp_333);  sp_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_335 = self.getattr_L__mod___layer3___18___bns_2(sp_334);  sp_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_336 = self.getattr_L__mod___layer3___18___relu(sp_335);  sp_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_203 = torch.cat([sp_328, sp_332, sp_336, getitem_103], 1);  sp_328 = sp_332 = sp_336 = getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_204 = self.getattr_L__mod___layer3___18___conv3(out_203);  out_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_205 = self.getattr_L__mod___layer3___18___bn3(out_204);  out_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_205 += shortcut_28;  out_206 = out_205;  out_205 = shortcut_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_29 = self.getattr_L__mod___layer3___18___relu(out_206);  out_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_208 = self.getattr_L__mod___layer3___19___conv1(shortcut_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_209 = self.getattr_L__mod___layer3___19___bn1(out_208);  out_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_210 = self.getattr_L__mod___layer3___19___relu(out_209);  out_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_26 = torch.functional.split(out_210, 104, 1);  out_210 = None
    sp_338 = split_26[0]
    getitem_105 = split_26[1]
    getitem_106 = split_26[2]
    getitem_107 = split_26[3];  split_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_339 = self.getattr_L__mod___layer3___19___convs_0(sp_338);  sp_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_340 = self.getattr_L__mod___layer3___19___bns_0(sp_339);  sp_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_341 = self.getattr_L__mod___layer3___19___relu(sp_340);  sp_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_342 = sp_341 + getitem_105;  getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_343 = self.getattr_L__mod___layer3___19___convs_1(sp_342);  sp_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_344 = self.getattr_L__mod___layer3___19___bns_1(sp_343);  sp_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_345 = self.getattr_L__mod___layer3___19___relu(sp_344);  sp_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_346 = sp_345 + getitem_106;  getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_347 = self.getattr_L__mod___layer3___19___convs_2(sp_346);  sp_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_348 = self.getattr_L__mod___layer3___19___bns_2(sp_347);  sp_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_349 = self.getattr_L__mod___layer3___19___relu(sp_348);  sp_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_211 = torch.cat([sp_341, sp_345, sp_349, getitem_107], 1);  sp_341 = sp_345 = sp_349 = getitem_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_212 = self.getattr_L__mod___layer3___19___conv3(out_211);  out_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_213 = self.getattr_L__mod___layer3___19___bn3(out_212);  out_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_213 += shortcut_29;  out_214 = out_213;  out_213 = shortcut_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_30 = self.getattr_L__mod___layer3___19___relu(out_214);  out_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_216 = self.getattr_L__mod___layer3___20___conv1(shortcut_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_217 = self.getattr_L__mod___layer3___20___bn1(out_216);  out_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_218 = self.getattr_L__mod___layer3___20___relu(out_217);  out_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_27 = torch.functional.split(out_218, 104, 1);  out_218 = None
    sp_351 = split_27[0]
    getitem_109 = split_27[1]
    getitem_110 = split_27[2]
    getitem_111 = split_27[3];  split_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_352 = self.getattr_L__mod___layer3___20___convs_0(sp_351);  sp_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_353 = self.getattr_L__mod___layer3___20___bns_0(sp_352);  sp_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_354 = self.getattr_L__mod___layer3___20___relu(sp_353);  sp_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_355 = sp_354 + getitem_109;  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_356 = self.getattr_L__mod___layer3___20___convs_1(sp_355);  sp_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_357 = self.getattr_L__mod___layer3___20___bns_1(sp_356);  sp_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_358 = self.getattr_L__mod___layer3___20___relu(sp_357);  sp_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_359 = sp_358 + getitem_110;  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_360 = self.getattr_L__mod___layer3___20___convs_2(sp_359);  sp_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_361 = self.getattr_L__mod___layer3___20___bns_2(sp_360);  sp_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_362 = self.getattr_L__mod___layer3___20___relu(sp_361);  sp_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_219 = torch.cat([sp_354, sp_358, sp_362, getitem_111], 1);  sp_354 = sp_358 = sp_362 = getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_220 = self.getattr_L__mod___layer3___20___conv3(out_219);  out_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_221 = self.getattr_L__mod___layer3___20___bn3(out_220);  out_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_221 += shortcut_30;  out_222 = out_221;  out_221 = shortcut_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_31 = self.getattr_L__mod___layer3___20___relu(out_222);  out_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_224 = self.getattr_L__mod___layer3___21___conv1(shortcut_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_225 = self.getattr_L__mod___layer3___21___bn1(out_224);  out_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_226 = self.getattr_L__mod___layer3___21___relu(out_225);  out_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_28 = torch.functional.split(out_226, 104, 1);  out_226 = None
    sp_364 = split_28[0]
    getitem_113 = split_28[1]
    getitem_114 = split_28[2]
    getitem_115 = split_28[3];  split_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_365 = self.getattr_L__mod___layer3___21___convs_0(sp_364);  sp_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_366 = self.getattr_L__mod___layer3___21___bns_0(sp_365);  sp_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_367 = self.getattr_L__mod___layer3___21___relu(sp_366);  sp_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_368 = sp_367 + getitem_113;  getitem_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_369 = self.getattr_L__mod___layer3___21___convs_1(sp_368);  sp_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_370 = self.getattr_L__mod___layer3___21___bns_1(sp_369);  sp_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_371 = self.getattr_L__mod___layer3___21___relu(sp_370);  sp_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_372 = sp_371 + getitem_114;  getitem_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_373 = self.getattr_L__mod___layer3___21___convs_2(sp_372);  sp_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_374 = self.getattr_L__mod___layer3___21___bns_2(sp_373);  sp_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_375 = self.getattr_L__mod___layer3___21___relu(sp_374);  sp_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_227 = torch.cat([sp_367, sp_371, sp_375, getitem_115], 1);  sp_367 = sp_371 = sp_375 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_228 = self.getattr_L__mod___layer3___21___conv3(out_227);  out_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_229 = self.getattr_L__mod___layer3___21___bn3(out_228);  out_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_229 += shortcut_31;  out_230 = out_229;  out_229 = shortcut_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_32 = self.getattr_L__mod___layer3___21___relu(out_230);  out_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_232 = self.getattr_L__mod___layer3___22___conv1(shortcut_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_233 = self.getattr_L__mod___layer3___22___bn1(out_232);  out_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_234 = self.getattr_L__mod___layer3___22___relu(out_233);  out_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_29 = torch.functional.split(out_234, 104, 1);  out_234 = None
    sp_377 = split_29[0]
    getitem_117 = split_29[1]
    getitem_118 = split_29[2]
    getitem_119 = split_29[3];  split_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_378 = self.getattr_L__mod___layer3___22___convs_0(sp_377);  sp_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_379 = self.getattr_L__mod___layer3___22___bns_0(sp_378);  sp_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_380 = self.getattr_L__mod___layer3___22___relu(sp_379);  sp_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_381 = sp_380 + getitem_117;  getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_382 = self.getattr_L__mod___layer3___22___convs_1(sp_381);  sp_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_383 = self.getattr_L__mod___layer3___22___bns_1(sp_382);  sp_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_384 = self.getattr_L__mod___layer3___22___relu(sp_383);  sp_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_385 = sp_384 + getitem_118;  getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_386 = self.getattr_L__mod___layer3___22___convs_2(sp_385);  sp_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_387 = self.getattr_L__mod___layer3___22___bns_2(sp_386);  sp_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_388 = self.getattr_L__mod___layer3___22___relu(sp_387);  sp_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_235 = torch.cat([sp_380, sp_384, sp_388, getitem_119], 1);  sp_380 = sp_384 = sp_388 = getitem_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_236 = self.getattr_L__mod___layer3___22___conv3(out_235);  out_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_237 = self.getattr_L__mod___layer3___22___bn3(out_236);  out_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_237 += shortcut_32;  out_238 = out_237;  out_237 = shortcut_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_33 = self.getattr_L__mod___layer3___22___relu(out_238);  out_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_240 = self.getattr_L__mod___layer4___0___conv1(shortcut_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_241 = self.getattr_L__mod___layer4___0___bn1(out_240);  out_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_242 = self.getattr_L__mod___layer4___0___relu(out_241);  out_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_30 = torch.functional.split(out_242, 208, 1);  out_242 = None
    sp_390 = split_30[0]
    sp_394 = split_30[1]
    sp_398 = split_30[2]
    getitem_123 = split_30[3];  split_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_391 = self.getattr_L__mod___layer4___0___convs_0(sp_390);  sp_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_392 = self.getattr_L__mod___layer4___0___bns_0(sp_391);  sp_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_393 = self.getattr_L__mod___layer4___0___relu(sp_392);  sp_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_395 = self.getattr_L__mod___layer4___0___convs_1(sp_394);  sp_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_396 = self.getattr_L__mod___layer4___0___bns_1(sp_395);  sp_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_397 = self.getattr_L__mod___layer4___0___relu(sp_396);  sp_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_399 = self.getattr_L__mod___layer4___0___convs_2(sp_398);  sp_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_400 = self.getattr_L__mod___layer4___0___bns_2(sp_399);  sp_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_401 = self.getattr_L__mod___layer4___0___relu(sp_400);  sp_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer4___0___pool = self.getattr_L__mod___layer4___0___pool(getitem_123);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_243 = torch.cat([sp_393, sp_397, sp_401, getattr_l__mod___layer4___0___pool], 1);  sp_393 = sp_397 = sp_401 = getattr_l__mod___layer4___0___pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_244 = self.getattr_L__mod___layer4___0___conv3(out_243);  out_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_245 = self.getattr_L__mod___layer4___0___bn3(out_244);  out_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(shortcut_33);  shortcut_33 = None
    shortcut_34 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_245 += shortcut_34;  out_246 = out_245;  out_245 = shortcut_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_35 = self.getattr_L__mod___layer4___0___relu(out_246);  out_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_248 = self.getattr_L__mod___layer4___1___conv1(shortcut_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_249 = self.getattr_L__mod___layer4___1___bn1(out_248);  out_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_250 = self.getattr_L__mod___layer4___1___relu(out_249);  out_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_31 = torch.functional.split(out_250, 208, 1);  out_250 = None
    sp_403 = split_31[0]
    getitem_125 = split_31[1]
    getitem_126 = split_31[2]
    getitem_127 = split_31[3];  split_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_404 = self.getattr_L__mod___layer4___1___convs_0(sp_403);  sp_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_405 = self.getattr_L__mod___layer4___1___bns_0(sp_404);  sp_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_406 = self.getattr_L__mod___layer4___1___relu(sp_405);  sp_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_407 = sp_406 + getitem_125;  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_408 = self.getattr_L__mod___layer4___1___convs_1(sp_407);  sp_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_409 = self.getattr_L__mod___layer4___1___bns_1(sp_408);  sp_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_410 = self.getattr_L__mod___layer4___1___relu(sp_409);  sp_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_411 = sp_410 + getitem_126;  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_412 = self.getattr_L__mod___layer4___1___convs_2(sp_411);  sp_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_413 = self.getattr_L__mod___layer4___1___bns_2(sp_412);  sp_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_414 = self.getattr_L__mod___layer4___1___relu(sp_413);  sp_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_251 = torch.cat([sp_406, sp_410, sp_414, getitem_127], 1);  sp_406 = sp_410 = sp_414 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_252 = self.getattr_L__mod___layer4___1___conv3(out_251);  out_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_253 = self.getattr_L__mod___layer4___1___bn3(out_252);  out_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_253 += shortcut_35;  out_254 = out_253;  out_253 = shortcut_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_36 = self.getattr_L__mod___layer4___1___relu(out_254);  out_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_256 = self.getattr_L__mod___layer4___2___conv1(shortcut_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_257 = self.getattr_L__mod___layer4___2___bn1(out_256);  out_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_258 = self.getattr_L__mod___layer4___2___relu(out_257);  out_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_32 = torch.functional.split(out_258, 208, 1);  out_258 = None
    sp_416 = split_32[0]
    getitem_129 = split_32[1]
    getitem_130 = split_32[2]
    getitem_131 = split_32[3];  split_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_417 = self.getattr_L__mod___layer4___2___convs_0(sp_416);  sp_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_418 = self.getattr_L__mod___layer4___2___bns_0(sp_417);  sp_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_419 = self.getattr_L__mod___layer4___2___relu(sp_418);  sp_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_420 = sp_419 + getitem_129;  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_421 = self.getattr_L__mod___layer4___2___convs_1(sp_420);  sp_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_422 = self.getattr_L__mod___layer4___2___bns_1(sp_421);  sp_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_423 = self.getattr_L__mod___layer4___2___relu(sp_422);  sp_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_424 = sp_423 + getitem_130;  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_425 = self.getattr_L__mod___layer4___2___convs_2(sp_424);  sp_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_426 = self.getattr_L__mod___layer4___2___bns_2(sp_425);  sp_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_427 = self.getattr_L__mod___layer4___2___relu(sp_426);  sp_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_259 = torch.cat([sp_419, sp_423, sp_427, getitem_131], 1);  sp_419 = sp_423 = sp_427 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_260 = self.getattr_L__mod___layer4___2___conv3(out_259);  out_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_261 = self.getattr_L__mod___layer4___2___bn3(out_260);  out_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_261 += shortcut_36;  out_262 = out_261;  out_261 = shortcut_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    x_8 = self.getattr_L__mod___layer4___2___relu(out_262);  out_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_9 = self.L__mod___global_pool_pool(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_11 = self.L__mod___global_pool_flatten(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    x_12 = self.L__mod___fc(x_11);  x_11 = None
    return (x_12,)
    