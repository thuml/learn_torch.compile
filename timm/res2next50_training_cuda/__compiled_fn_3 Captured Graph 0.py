from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    x = self.L__mod___conv1(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
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
    split = torch.functional.split(out_2, 32, 1);  out_2 = None
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
    split_1 = torch.functional.split(out_10, 32, 1);  out_10 = None
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
    split_2 = torch.functional.split(out_18, 32, 1);  out_18 = None
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
    split_3 = torch.functional.split(out_26, 64, 1);  out_26 = None
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
    split_4 = torch.functional.split(out_34, 64, 1);  out_34 = None
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
    split_5 = torch.functional.split(out_42, 64, 1);  out_42 = None
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
    split_6 = torch.functional.split(out_50, 64, 1);  out_50 = None
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
    split_7 = torch.functional.split(out_58, 128, 1);  out_58 = None
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
    split_8 = torch.functional.split(out_66, 128, 1);  out_66 = None
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
    split_9 = torch.functional.split(out_74, 128, 1);  out_74 = None
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
    split_10 = torch.functional.split(out_82, 128, 1);  out_82 = None
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
    split_11 = torch.functional.split(out_90, 128, 1);  out_90 = None
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
    split_12 = torch.functional.split(out_98, 128, 1);  out_98 = None
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
    out_104 = self.getattr_L__mod___layer4___0___conv1(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_105 = self.getattr_L__mod___layer4___0___bn1(out_104);  out_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_106 = self.getattr_L__mod___layer4___0___relu(out_105);  out_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_13 = torch.functional.split(out_106, 256, 1);  out_106 = None
    sp_169 = split_13[0]
    sp_173 = split_13[1]
    sp_177 = split_13[2]
    getitem_55 = split_13[3];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_170 = self.getattr_L__mod___layer4___0___convs_0(sp_169);  sp_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_171 = self.getattr_L__mod___layer4___0___bns_0(sp_170);  sp_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_172 = self.getattr_L__mod___layer4___0___relu(sp_171);  sp_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_174 = self.getattr_L__mod___layer4___0___convs_1(sp_173);  sp_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_175 = self.getattr_L__mod___layer4___0___bns_1(sp_174);  sp_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_176 = self.getattr_L__mod___layer4___0___relu(sp_175);  sp_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_178 = self.getattr_L__mod___layer4___0___convs_2(sp_177);  sp_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_179 = self.getattr_L__mod___layer4___0___bns_2(sp_178);  sp_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_180 = self.getattr_L__mod___layer4___0___relu(sp_179);  sp_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer4___0___pool = self.getattr_L__mod___layer4___0___pool(getitem_55);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_107 = torch.cat([sp_172, sp_176, sp_180, getattr_l__mod___layer4___0___pool], 1);  sp_172 = sp_176 = sp_180 = getattr_l__mod___layer4___0___pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_108 = self.getattr_L__mod___layer4___0___conv3(out_107);  out_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_109 = self.getattr_L__mod___layer4___0___bn3(out_108);  out_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(shortcut_16);  shortcut_16 = None
    shortcut_17 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_109 += shortcut_17;  out_110 = out_109;  out_109 = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_18 = self.getattr_L__mod___layer4___0___relu(out_110);  out_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_112 = self.getattr_L__mod___layer4___1___conv1(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_113 = self.getattr_L__mod___layer4___1___bn1(out_112);  out_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_114 = self.getattr_L__mod___layer4___1___relu(out_113);  out_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_14 = torch.functional.split(out_114, 256, 1);  out_114 = None
    sp_182 = split_14[0]
    getitem_57 = split_14[1]
    getitem_58 = split_14[2]
    getitem_59 = split_14[3];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_183 = self.getattr_L__mod___layer4___1___convs_0(sp_182);  sp_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_184 = self.getattr_L__mod___layer4___1___bns_0(sp_183);  sp_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_185 = self.getattr_L__mod___layer4___1___relu(sp_184);  sp_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_186 = sp_185 + getitem_57;  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_187 = self.getattr_L__mod___layer4___1___convs_1(sp_186);  sp_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_188 = self.getattr_L__mod___layer4___1___bns_1(sp_187);  sp_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_189 = self.getattr_L__mod___layer4___1___relu(sp_188);  sp_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_190 = sp_189 + getitem_58;  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_191 = self.getattr_L__mod___layer4___1___convs_2(sp_190);  sp_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_192 = self.getattr_L__mod___layer4___1___bns_2(sp_191);  sp_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_193 = self.getattr_L__mod___layer4___1___relu(sp_192);  sp_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_115 = torch.cat([sp_185, sp_189, sp_193, getitem_59], 1);  sp_185 = sp_189 = sp_193 = getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_116 = self.getattr_L__mod___layer4___1___conv3(out_115);  out_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_117 = self.getattr_L__mod___layer4___1___bn3(out_116);  out_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_117 += shortcut_18;  out_118 = out_117;  out_117 = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    shortcut_19 = self.getattr_L__mod___layer4___1___relu(out_118);  out_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    out_120 = self.getattr_L__mod___layer4___2___conv1(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    out_121 = self.getattr_L__mod___layer4___2___bn1(out_120);  out_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    out_122 = self.getattr_L__mod___layer4___2___relu(out_121);  out_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    split_15 = torch.functional.split(out_122, 256, 1);  out_122 = None
    sp_195 = split_15[0]
    getitem_61 = split_15[1]
    getitem_62 = split_15[2]
    getitem_63 = split_15[3];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_196 = self.getattr_L__mod___layer4___2___convs_0(sp_195);  sp_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_197 = self.getattr_L__mod___layer4___2___bns_0(sp_196);  sp_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_198 = self.getattr_L__mod___layer4___2___relu(sp_197);  sp_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_199 = sp_198 + getitem_61;  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_200 = self.getattr_L__mod___layer4___2___convs_1(sp_199);  sp_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_201 = self.getattr_L__mod___layer4___2___bns_1(sp_200);  sp_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_202 = self.getattr_L__mod___layer4___2___relu(sp_201);  sp_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_203 = sp_202 + getitem_62;  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_204 = self.getattr_L__mod___layer4___2___convs_2(sp_203);  sp_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_205 = self.getattr_L__mod___layer4___2___bns_2(sp_204);  sp_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_206 = self.getattr_L__mod___layer4___2___relu(sp_205);  sp_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_123 = torch.cat([sp_198, sp_202, sp_206, getitem_63], 1);  sp_198 = sp_202 = sp_206 = getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    out_124 = self.getattr_L__mod___layer4___2___conv3(out_123);  out_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    out_125 = self.getattr_L__mod___layer4___2___bn3(out_124);  out_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    out_125 += shortcut_19;  out_126 = out_125;  out_125 = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    x_8 = self.getattr_L__mod___layer4___2___relu(out_126);  out_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_9 = self.L__mod___global_pool_pool(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_11 = self.L__mod___global_pool_flatten(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    pred = self.L__mod___fc(x_11);  x_11 = None
    return (pred,)
    