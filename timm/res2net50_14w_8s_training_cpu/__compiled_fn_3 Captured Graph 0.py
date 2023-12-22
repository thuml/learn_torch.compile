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
    split = torch.functional.split(out_2, 14, 1);  out_2 = None
    sp = split[0]
    sp_4 = split[1]
    sp_8 = split[2]
    sp_12 = split[3]
    sp_16 = split[4]
    sp_20 = split[5]
    sp_24 = split[6]
    getitem_7 = split[7];  split = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_13 = self.getattr_L__mod___layer1___0___convs_3(sp_12);  sp_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_14 = self.getattr_L__mod___layer1___0___bns_3(sp_13);  sp_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_15 = self.getattr_L__mod___layer1___0___relu(sp_14);  sp_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_17 = self.getattr_L__mod___layer1___0___convs_4(sp_16);  sp_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_18 = self.getattr_L__mod___layer1___0___bns_4(sp_17);  sp_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_19 = self.getattr_L__mod___layer1___0___relu(sp_18);  sp_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_21 = self.getattr_L__mod___layer1___0___convs_5(sp_20);  sp_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_22 = self.getattr_L__mod___layer1___0___bns_5(sp_21);  sp_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_23 = self.getattr_L__mod___layer1___0___relu(sp_22);  sp_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_25 = self.getattr_L__mod___layer1___0___convs_6(sp_24);  sp_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_26 = self.getattr_L__mod___layer1___0___bns_6(sp_25);  sp_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_27 = self.getattr_L__mod___layer1___0___relu(sp_26);  sp_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer1___0___pool = self.getattr_L__mod___layer1___0___pool(getitem_7);  getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_3 = torch.cat([sp_3, sp_7, sp_11, sp_15, sp_19, sp_23, sp_27, getattr_l__mod___layer1___0___pool], 1);  sp_3 = sp_7 = sp_11 = sp_15 = sp_19 = sp_23 = sp_27 = getattr_l__mod___layer1___0___pool = None
    
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
    split_1 = torch.functional.split(out_10, 14, 1);  out_10 = None
    sp_29 = split_1[0]
    getitem_9 = split_1[1]
    getitem_10 = split_1[2]
    getitem_11 = split_1[3]
    getitem_12 = split_1[4]
    getitem_13 = split_1[5]
    getitem_14 = split_1[6]
    getitem_15 = split_1[7];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_30 = self.getattr_L__mod___layer1___1___convs_0(sp_29);  sp_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_31 = self.getattr_L__mod___layer1___1___bns_0(sp_30);  sp_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_32 = self.getattr_L__mod___layer1___1___relu(sp_31);  sp_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_33 = sp_32 + getitem_9;  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_34 = self.getattr_L__mod___layer1___1___convs_1(sp_33);  sp_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_35 = self.getattr_L__mod___layer1___1___bns_1(sp_34);  sp_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_36 = self.getattr_L__mod___layer1___1___relu(sp_35);  sp_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_37 = sp_36 + getitem_10;  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_38 = self.getattr_L__mod___layer1___1___convs_2(sp_37);  sp_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_39 = self.getattr_L__mod___layer1___1___bns_2(sp_38);  sp_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_40 = self.getattr_L__mod___layer1___1___relu(sp_39);  sp_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_41 = sp_40 + getitem_11;  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_42 = self.getattr_L__mod___layer1___1___convs_3(sp_41);  sp_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_43 = self.getattr_L__mod___layer1___1___bns_3(sp_42);  sp_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_44 = self.getattr_L__mod___layer1___1___relu(sp_43);  sp_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_45 = sp_44 + getitem_12;  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_46 = self.getattr_L__mod___layer1___1___convs_4(sp_45);  sp_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_47 = self.getattr_L__mod___layer1___1___bns_4(sp_46);  sp_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_48 = self.getattr_L__mod___layer1___1___relu(sp_47);  sp_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_49 = sp_48 + getitem_13;  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_50 = self.getattr_L__mod___layer1___1___convs_5(sp_49);  sp_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_51 = self.getattr_L__mod___layer1___1___bns_5(sp_50);  sp_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_52 = self.getattr_L__mod___layer1___1___relu(sp_51);  sp_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_53 = sp_52 + getitem_14;  getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_54 = self.getattr_L__mod___layer1___1___convs_6(sp_53);  sp_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_55 = self.getattr_L__mod___layer1___1___bns_6(sp_54);  sp_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_56 = self.getattr_L__mod___layer1___1___relu(sp_55);  sp_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_11 = torch.cat([sp_32, sp_36, sp_40, sp_44, sp_48, sp_52, sp_56, getitem_15], 1);  sp_32 = sp_36 = sp_40 = sp_44 = sp_48 = sp_52 = sp_56 = getitem_15 = None
    
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
    split_2 = torch.functional.split(out_18, 14, 1);  out_18 = None
    sp_58 = split_2[0]
    getitem_17 = split_2[1]
    getitem_18 = split_2[2]
    getitem_19 = split_2[3]
    getitem_20 = split_2[4]
    getitem_21 = split_2[5]
    getitem_22 = split_2[6]
    getitem_23 = split_2[7];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_59 = self.getattr_L__mod___layer1___2___convs_0(sp_58);  sp_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_60 = self.getattr_L__mod___layer1___2___bns_0(sp_59);  sp_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_61 = self.getattr_L__mod___layer1___2___relu(sp_60);  sp_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_62 = sp_61 + getitem_17;  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_63 = self.getattr_L__mod___layer1___2___convs_1(sp_62);  sp_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_64 = self.getattr_L__mod___layer1___2___bns_1(sp_63);  sp_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_65 = self.getattr_L__mod___layer1___2___relu(sp_64);  sp_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_66 = sp_65 + getitem_18;  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_67 = self.getattr_L__mod___layer1___2___convs_2(sp_66);  sp_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_68 = self.getattr_L__mod___layer1___2___bns_2(sp_67);  sp_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_69 = self.getattr_L__mod___layer1___2___relu(sp_68);  sp_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_70 = sp_69 + getitem_19;  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_71 = self.getattr_L__mod___layer1___2___convs_3(sp_70);  sp_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_72 = self.getattr_L__mod___layer1___2___bns_3(sp_71);  sp_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_73 = self.getattr_L__mod___layer1___2___relu(sp_72);  sp_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_74 = sp_73 + getitem_20;  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_75 = self.getattr_L__mod___layer1___2___convs_4(sp_74);  sp_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_76 = self.getattr_L__mod___layer1___2___bns_4(sp_75);  sp_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_77 = self.getattr_L__mod___layer1___2___relu(sp_76);  sp_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_78 = sp_77 + getitem_21;  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_79 = self.getattr_L__mod___layer1___2___convs_5(sp_78);  sp_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_80 = self.getattr_L__mod___layer1___2___bns_5(sp_79);  sp_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_81 = self.getattr_L__mod___layer1___2___relu(sp_80);  sp_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_82 = sp_81 + getitem_22;  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_83 = self.getattr_L__mod___layer1___2___convs_6(sp_82);  sp_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_84 = self.getattr_L__mod___layer1___2___bns_6(sp_83);  sp_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_85 = self.getattr_L__mod___layer1___2___relu(sp_84);  sp_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_19 = torch.cat([sp_61, sp_65, sp_69, sp_73, sp_77, sp_81, sp_85, getitem_23], 1);  sp_61 = sp_65 = sp_69 = sp_73 = sp_77 = sp_81 = sp_85 = getitem_23 = None
    
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
    split_3 = torch.functional.split(out_26, 28, 1);  out_26 = None
    sp_87 = split_3[0]
    sp_91 = split_3[1]
    sp_95 = split_3[2]
    sp_99 = split_3[3]
    sp_103 = split_3[4]
    sp_107 = split_3[5]
    sp_111 = split_3[6]
    getitem_31 = split_3[7];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_88 = self.getattr_L__mod___layer2___0___convs_0(sp_87);  sp_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_89 = self.getattr_L__mod___layer2___0___bns_0(sp_88);  sp_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_90 = self.getattr_L__mod___layer2___0___relu(sp_89);  sp_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_92 = self.getattr_L__mod___layer2___0___convs_1(sp_91);  sp_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_93 = self.getattr_L__mod___layer2___0___bns_1(sp_92);  sp_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_94 = self.getattr_L__mod___layer2___0___relu(sp_93);  sp_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_96 = self.getattr_L__mod___layer2___0___convs_2(sp_95);  sp_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_97 = self.getattr_L__mod___layer2___0___bns_2(sp_96);  sp_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_98 = self.getattr_L__mod___layer2___0___relu(sp_97);  sp_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_100 = self.getattr_L__mod___layer2___0___convs_3(sp_99);  sp_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_101 = self.getattr_L__mod___layer2___0___bns_3(sp_100);  sp_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_102 = self.getattr_L__mod___layer2___0___relu(sp_101);  sp_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_104 = self.getattr_L__mod___layer2___0___convs_4(sp_103);  sp_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_105 = self.getattr_L__mod___layer2___0___bns_4(sp_104);  sp_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_106 = self.getattr_L__mod___layer2___0___relu(sp_105);  sp_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_108 = self.getattr_L__mod___layer2___0___convs_5(sp_107);  sp_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_109 = self.getattr_L__mod___layer2___0___bns_5(sp_108);  sp_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_110 = self.getattr_L__mod___layer2___0___relu(sp_109);  sp_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_112 = self.getattr_L__mod___layer2___0___convs_6(sp_111);  sp_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_113 = self.getattr_L__mod___layer2___0___bns_6(sp_112);  sp_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_114 = self.getattr_L__mod___layer2___0___relu(sp_113);  sp_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer2___0___pool = self.getattr_L__mod___layer2___0___pool(getitem_31);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_27 = torch.cat([sp_90, sp_94, sp_98, sp_102, sp_106, sp_110, sp_114, getattr_l__mod___layer2___0___pool], 1);  sp_90 = sp_94 = sp_98 = sp_102 = sp_106 = sp_110 = sp_114 = getattr_l__mod___layer2___0___pool = None
    
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
    split_4 = torch.functional.split(out_34, 28, 1);  out_34 = None
    sp_116 = split_4[0]
    getitem_33 = split_4[1]
    getitem_34 = split_4[2]
    getitem_35 = split_4[3]
    getitem_36 = split_4[4]
    getitem_37 = split_4[5]
    getitem_38 = split_4[6]
    getitem_39 = split_4[7];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_117 = self.getattr_L__mod___layer2___1___convs_0(sp_116);  sp_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_118 = self.getattr_L__mod___layer2___1___bns_0(sp_117);  sp_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_119 = self.getattr_L__mod___layer2___1___relu(sp_118);  sp_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_120 = sp_119 + getitem_33;  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_121 = self.getattr_L__mod___layer2___1___convs_1(sp_120);  sp_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_122 = self.getattr_L__mod___layer2___1___bns_1(sp_121);  sp_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_123 = self.getattr_L__mod___layer2___1___relu(sp_122);  sp_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_124 = sp_123 + getitem_34;  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_125 = self.getattr_L__mod___layer2___1___convs_2(sp_124);  sp_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_126 = self.getattr_L__mod___layer2___1___bns_2(sp_125);  sp_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_127 = self.getattr_L__mod___layer2___1___relu(sp_126);  sp_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_128 = sp_127 + getitem_35;  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_129 = self.getattr_L__mod___layer2___1___convs_3(sp_128);  sp_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_130 = self.getattr_L__mod___layer2___1___bns_3(sp_129);  sp_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_131 = self.getattr_L__mod___layer2___1___relu(sp_130);  sp_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_132 = sp_131 + getitem_36;  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_133 = self.getattr_L__mod___layer2___1___convs_4(sp_132);  sp_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_134 = self.getattr_L__mod___layer2___1___bns_4(sp_133);  sp_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_135 = self.getattr_L__mod___layer2___1___relu(sp_134);  sp_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_136 = sp_135 + getitem_37;  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_137 = self.getattr_L__mod___layer2___1___convs_5(sp_136);  sp_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_138 = self.getattr_L__mod___layer2___1___bns_5(sp_137);  sp_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_139 = self.getattr_L__mod___layer2___1___relu(sp_138);  sp_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_140 = sp_139 + getitem_38;  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_141 = self.getattr_L__mod___layer2___1___convs_6(sp_140);  sp_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_142 = self.getattr_L__mod___layer2___1___bns_6(sp_141);  sp_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_143 = self.getattr_L__mod___layer2___1___relu(sp_142);  sp_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_35 = torch.cat([sp_119, sp_123, sp_127, sp_131, sp_135, sp_139, sp_143, getitem_39], 1);  sp_119 = sp_123 = sp_127 = sp_131 = sp_135 = sp_139 = sp_143 = getitem_39 = None
    
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
    split_5 = torch.functional.split(out_42, 28, 1);  out_42 = None
    sp_145 = split_5[0]
    getitem_41 = split_5[1]
    getitem_42 = split_5[2]
    getitem_43 = split_5[3]
    getitem_44 = split_5[4]
    getitem_45 = split_5[5]
    getitem_46 = split_5[6]
    getitem_47 = split_5[7];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_146 = self.getattr_L__mod___layer2___2___convs_0(sp_145);  sp_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_147 = self.getattr_L__mod___layer2___2___bns_0(sp_146);  sp_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_148 = self.getattr_L__mod___layer2___2___relu(sp_147);  sp_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_149 = sp_148 + getitem_41;  getitem_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_150 = self.getattr_L__mod___layer2___2___convs_1(sp_149);  sp_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_151 = self.getattr_L__mod___layer2___2___bns_1(sp_150);  sp_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_152 = self.getattr_L__mod___layer2___2___relu(sp_151);  sp_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_153 = sp_152 + getitem_42;  getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_154 = self.getattr_L__mod___layer2___2___convs_2(sp_153);  sp_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_155 = self.getattr_L__mod___layer2___2___bns_2(sp_154);  sp_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_156 = self.getattr_L__mod___layer2___2___relu(sp_155);  sp_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_157 = sp_156 + getitem_43;  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_158 = self.getattr_L__mod___layer2___2___convs_3(sp_157);  sp_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_159 = self.getattr_L__mod___layer2___2___bns_3(sp_158);  sp_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_160 = self.getattr_L__mod___layer2___2___relu(sp_159);  sp_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_161 = sp_160 + getitem_44;  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_162 = self.getattr_L__mod___layer2___2___convs_4(sp_161);  sp_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_163 = self.getattr_L__mod___layer2___2___bns_4(sp_162);  sp_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_164 = self.getattr_L__mod___layer2___2___relu(sp_163);  sp_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_165 = sp_164 + getitem_45;  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_166 = self.getattr_L__mod___layer2___2___convs_5(sp_165);  sp_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_167 = self.getattr_L__mod___layer2___2___bns_5(sp_166);  sp_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_168 = self.getattr_L__mod___layer2___2___relu(sp_167);  sp_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_169 = sp_168 + getitem_46;  getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_170 = self.getattr_L__mod___layer2___2___convs_6(sp_169);  sp_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_171 = self.getattr_L__mod___layer2___2___bns_6(sp_170);  sp_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_172 = self.getattr_L__mod___layer2___2___relu(sp_171);  sp_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_43 = torch.cat([sp_148, sp_152, sp_156, sp_160, sp_164, sp_168, sp_172, getitem_47], 1);  sp_148 = sp_152 = sp_156 = sp_160 = sp_164 = sp_168 = sp_172 = getitem_47 = None
    
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
    split_6 = torch.functional.split(out_50, 28, 1);  out_50 = None
    sp_174 = split_6[0]
    getitem_49 = split_6[1]
    getitem_50 = split_6[2]
    getitem_51 = split_6[3]
    getitem_52 = split_6[4]
    getitem_53 = split_6[5]
    getitem_54 = split_6[6]
    getitem_55 = split_6[7];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_175 = self.getattr_L__mod___layer2___3___convs_0(sp_174);  sp_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_176 = self.getattr_L__mod___layer2___3___bns_0(sp_175);  sp_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_177 = self.getattr_L__mod___layer2___3___relu(sp_176);  sp_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_178 = sp_177 + getitem_49;  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_179 = self.getattr_L__mod___layer2___3___convs_1(sp_178);  sp_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_180 = self.getattr_L__mod___layer2___3___bns_1(sp_179);  sp_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_181 = self.getattr_L__mod___layer2___3___relu(sp_180);  sp_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_182 = sp_181 + getitem_50;  getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_183 = self.getattr_L__mod___layer2___3___convs_2(sp_182);  sp_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_184 = self.getattr_L__mod___layer2___3___bns_2(sp_183);  sp_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_185 = self.getattr_L__mod___layer2___3___relu(sp_184);  sp_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_186 = sp_185 + getitem_51;  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_187 = self.getattr_L__mod___layer2___3___convs_3(sp_186);  sp_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_188 = self.getattr_L__mod___layer2___3___bns_3(sp_187);  sp_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_189 = self.getattr_L__mod___layer2___3___relu(sp_188);  sp_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_190 = sp_189 + getitem_52;  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_191 = self.getattr_L__mod___layer2___3___convs_4(sp_190);  sp_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_192 = self.getattr_L__mod___layer2___3___bns_4(sp_191);  sp_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_193 = self.getattr_L__mod___layer2___3___relu(sp_192);  sp_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_194 = sp_193 + getitem_53;  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_195 = self.getattr_L__mod___layer2___3___convs_5(sp_194);  sp_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_196 = self.getattr_L__mod___layer2___3___bns_5(sp_195);  sp_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_197 = self.getattr_L__mod___layer2___3___relu(sp_196);  sp_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_198 = sp_197 + getitem_54;  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_199 = self.getattr_L__mod___layer2___3___convs_6(sp_198);  sp_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_200 = self.getattr_L__mod___layer2___3___bns_6(sp_199);  sp_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_201 = self.getattr_L__mod___layer2___3___relu(sp_200);  sp_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_51 = torch.cat([sp_177, sp_181, sp_185, sp_189, sp_193, sp_197, sp_201, getitem_55], 1);  sp_177 = sp_181 = sp_185 = sp_189 = sp_193 = sp_197 = sp_201 = getitem_55 = None
    
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
    split_7 = torch.functional.split(out_58, 56, 1);  out_58 = None
    sp_203 = split_7[0]
    sp_207 = split_7[1]
    sp_211 = split_7[2]
    sp_215 = split_7[3]
    sp_219 = split_7[4]
    sp_223 = split_7[5]
    sp_227 = split_7[6]
    getitem_63 = split_7[7];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_204 = self.getattr_L__mod___layer3___0___convs_0(sp_203);  sp_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_205 = self.getattr_L__mod___layer3___0___bns_0(sp_204);  sp_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_206 = self.getattr_L__mod___layer3___0___relu(sp_205);  sp_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_208 = self.getattr_L__mod___layer3___0___convs_1(sp_207);  sp_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_209 = self.getattr_L__mod___layer3___0___bns_1(sp_208);  sp_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_210 = self.getattr_L__mod___layer3___0___relu(sp_209);  sp_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_212 = self.getattr_L__mod___layer3___0___convs_2(sp_211);  sp_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_213 = self.getattr_L__mod___layer3___0___bns_2(sp_212);  sp_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_214 = self.getattr_L__mod___layer3___0___relu(sp_213);  sp_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_216 = self.getattr_L__mod___layer3___0___convs_3(sp_215);  sp_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_217 = self.getattr_L__mod___layer3___0___bns_3(sp_216);  sp_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_218 = self.getattr_L__mod___layer3___0___relu(sp_217);  sp_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_220 = self.getattr_L__mod___layer3___0___convs_4(sp_219);  sp_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_221 = self.getattr_L__mod___layer3___0___bns_4(sp_220);  sp_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_222 = self.getattr_L__mod___layer3___0___relu(sp_221);  sp_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_224 = self.getattr_L__mod___layer3___0___convs_5(sp_223);  sp_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_225 = self.getattr_L__mod___layer3___0___bns_5(sp_224);  sp_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_226 = self.getattr_L__mod___layer3___0___relu(sp_225);  sp_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_228 = self.getattr_L__mod___layer3___0___convs_6(sp_227);  sp_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_229 = self.getattr_L__mod___layer3___0___bns_6(sp_228);  sp_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_230 = self.getattr_L__mod___layer3___0___relu(sp_229);  sp_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer3___0___pool = self.getattr_L__mod___layer3___0___pool(getitem_63);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_59 = torch.cat([sp_206, sp_210, sp_214, sp_218, sp_222, sp_226, sp_230, getattr_l__mod___layer3___0___pool], 1);  sp_206 = sp_210 = sp_214 = sp_218 = sp_222 = sp_226 = sp_230 = getattr_l__mod___layer3___0___pool = None
    
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
    split_8 = torch.functional.split(out_66, 56, 1);  out_66 = None
    sp_232 = split_8[0]
    getitem_65 = split_8[1]
    getitem_66 = split_8[2]
    getitem_67 = split_8[3]
    getitem_68 = split_8[4]
    getitem_69 = split_8[5]
    getitem_70 = split_8[6]
    getitem_71 = split_8[7];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_233 = self.getattr_L__mod___layer3___1___convs_0(sp_232);  sp_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_234 = self.getattr_L__mod___layer3___1___bns_0(sp_233);  sp_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_235 = self.getattr_L__mod___layer3___1___relu(sp_234);  sp_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_236 = sp_235 + getitem_65;  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_237 = self.getattr_L__mod___layer3___1___convs_1(sp_236);  sp_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_238 = self.getattr_L__mod___layer3___1___bns_1(sp_237);  sp_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_239 = self.getattr_L__mod___layer3___1___relu(sp_238);  sp_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_240 = sp_239 + getitem_66;  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_241 = self.getattr_L__mod___layer3___1___convs_2(sp_240);  sp_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_242 = self.getattr_L__mod___layer3___1___bns_2(sp_241);  sp_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_243 = self.getattr_L__mod___layer3___1___relu(sp_242);  sp_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_244 = sp_243 + getitem_67;  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_245 = self.getattr_L__mod___layer3___1___convs_3(sp_244);  sp_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_246 = self.getattr_L__mod___layer3___1___bns_3(sp_245);  sp_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_247 = self.getattr_L__mod___layer3___1___relu(sp_246);  sp_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_248 = sp_247 + getitem_68;  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_249 = self.getattr_L__mod___layer3___1___convs_4(sp_248);  sp_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_250 = self.getattr_L__mod___layer3___1___bns_4(sp_249);  sp_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_251 = self.getattr_L__mod___layer3___1___relu(sp_250);  sp_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_252 = sp_251 + getitem_69;  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_253 = self.getattr_L__mod___layer3___1___convs_5(sp_252);  sp_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_254 = self.getattr_L__mod___layer3___1___bns_5(sp_253);  sp_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_255 = self.getattr_L__mod___layer3___1___relu(sp_254);  sp_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_256 = sp_255 + getitem_70;  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_257 = self.getattr_L__mod___layer3___1___convs_6(sp_256);  sp_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_258 = self.getattr_L__mod___layer3___1___bns_6(sp_257);  sp_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_259 = self.getattr_L__mod___layer3___1___relu(sp_258);  sp_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_67 = torch.cat([sp_235, sp_239, sp_243, sp_247, sp_251, sp_255, sp_259, getitem_71], 1);  sp_235 = sp_239 = sp_243 = sp_247 = sp_251 = sp_255 = sp_259 = getitem_71 = None
    
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
    split_9 = torch.functional.split(out_74, 56, 1);  out_74 = None
    sp_261 = split_9[0]
    getitem_73 = split_9[1]
    getitem_74 = split_9[2]
    getitem_75 = split_9[3]
    getitem_76 = split_9[4]
    getitem_77 = split_9[5]
    getitem_78 = split_9[6]
    getitem_79 = split_9[7];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_262 = self.getattr_L__mod___layer3___2___convs_0(sp_261);  sp_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_263 = self.getattr_L__mod___layer3___2___bns_0(sp_262);  sp_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_264 = self.getattr_L__mod___layer3___2___relu(sp_263);  sp_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_265 = sp_264 + getitem_73;  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_266 = self.getattr_L__mod___layer3___2___convs_1(sp_265);  sp_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_267 = self.getattr_L__mod___layer3___2___bns_1(sp_266);  sp_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_268 = self.getattr_L__mod___layer3___2___relu(sp_267);  sp_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_269 = sp_268 + getitem_74;  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_270 = self.getattr_L__mod___layer3___2___convs_2(sp_269);  sp_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_271 = self.getattr_L__mod___layer3___2___bns_2(sp_270);  sp_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_272 = self.getattr_L__mod___layer3___2___relu(sp_271);  sp_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_273 = sp_272 + getitem_75;  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_274 = self.getattr_L__mod___layer3___2___convs_3(sp_273);  sp_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_275 = self.getattr_L__mod___layer3___2___bns_3(sp_274);  sp_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_276 = self.getattr_L__mod___layer3___2___relu(sp_275);  sp_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_277 = sp_276 + getitem_76;  getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_278 = self.getattr_L__mod___layer3___2___convs_4(sp_277);  sp_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_279 = self.getattr_L__mod___layer3___2___bns_4(sp_278);  sp_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_280 = self.getattr_L__mod___layer3___2___relu(sp_279);  sp_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_281 = sp_280 + getitem_77;  getitem_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_282 = self.getattr_L__mod___layer3___2___convs_5(sp_281);  sp_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_283 = self.getattr_L__mod___layer3___2___bns_5(sp_282);  sp_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_284 = self.getattr_L__mod___layer3___2___relu(sp_283);  sp_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_285 = sp_284 + getitem_78;  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_286 = self.getattr_L__mod___layer3___2___convs_6(sp_285);  sp_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_287 = self.getattr_L__mod___layer3___2___bns_6(sp_286);  sp_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_288 = self.getattr_L__mod___layer3___2___relu(sp_287);  sp_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_75 = torch.cat([sp_264, sp_268, sp_272, sp_276, sp_280, sp_284, sp_288, getitem_79], 1);  sp_264 = sp_268 = sp_272 = sp_276 = sp_280 = sp_284 = sp_288 = getitem_79 = None
    
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
    split_10 = torch.functional.split(out_82, 56, 1);  out_82 = None
    sp_290 = split_10[0]
    getitem_81 = split_10[1]
    getitem_82 = split_10[2]
    getitem_83 = split_10[3]
    getitem_84 = split_10[4]
    getitem_85 = split_10[5]
    getitem_86 = split_10[6]
    getitem_87 = split_10[7];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_291 = self.getattr_L__mod___layer3___3___convs_0(sp_290);  sp_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_292 = self.getattr_L__mod___layer3___3___bns_0(sp_291);  sp_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_293 = self.getattr_L__mod___layer3___3___relu(sp_292);  sp_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_294 = sp_293 + getitem_81;  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_295 = self.getattr_L__mod___layer3___3___convs_1(sp_294);  sp_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_296 = self.getattr_L__mod___layer3___3___bns_1(sp_295);  sp_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_297 = self.getattr_L__mod___layer3___3___relu(sp_296);  sp_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_298 = sp_297 + getitem_82;  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_299 = self.getattr_L__mod___layer3___3___convs_2(sp_298);  sp_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_300 = self.getattr_L__mod___layer3___3___bns_2(sp_299);  sp_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_301 = self.getattr_L__mod___layer3___3___relu(sp_300);  sp_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_302 = sp_301 + getitem_83;  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_303 = self.getattr_L__mod___layer3___3___convs_3(sp_302);  sp_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_304 = self.getattr_L__mod___layer3___3___bns_3(sp_303);  sp_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_305 = self.getattr_L__mod___layer3___3___relu(sp_304);  sp_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_306 = sp_305 + getitem_84;  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_307 = self.getattr_L__mod___layer3___3___convs_4(sp_306);  sp_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_308 = self.getattr_L__mod___layer3___3___bns_4(sp_307);  sp_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_309 = self.getattr_L__mod___layer3___3___relu(sp_308);  sp_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_310 = sp_309 + getitem_85;  getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_311 = self.getattr_L__mod___layer3___3___convs_5(sp_310);  sp_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_312 = self.getattr_L__mod___layer3___3___bns_5(sp_311);  sp_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_313 = self.getattr_L__mod___layer3___3___relu(sp_312);  sp_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_314 = sp_313 + getitem_86;  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_315 = self.getattr_L__mod___layer3___3___convs_6(sp_314);  sp_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_316 = self.getattr_L__mod___layer3___3___bns_6(sp_315);  sp_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_317 = self.getattr_L__mod___layer3___3___relu(sp_316);  sp_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_83 = torch.cat([sp_293, sp_297, sp_301, sp_305, sp_309, sp_313, sp_317, getitem_87], 1);  sp_293 = sp_297 = sp_301 = sp_305 = sp_309 = sp_313 = sp_317 = getitem_87 = None
    
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
    split_11 = torch.functional.split(out_90, 56, 1);  out_90 = None
    sp_319 = split_11[0]
    getitem_89 = split_11[1]
    getitem_90 = split_11[2]
    getitem_91 = split_11[3]
    getitem_92 = split_11[4]
    getitem_93 = split_11[5]
    getitem_94 = split_11[6]
    getitem_95 = split_11[7];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_320 = self.getattr_L__mod___layer3___4___convs_0(sp_319);  sp_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_321 = self.getattr_L__mod___layer3___4___bns_0(sp_320);  sp_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_322 = self.getattr_L__mod___layer3___4___relu(sp_321);  sp_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_323 = sp_322 + getitem_89;  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_324 = self.getattr_L__mod___layer3___4___convs_1(sp_323);  sp_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_325 = self.getattr_L__mod___layer3___4___bns_1(sp_324);  sp_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_326 = self.getattr_L__mod___layer3___4___relu(sp_325);  sp_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_327 = sp_326 + getitem_90;  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_328 = self.getattr_L__mod___layer3___4___convs_2(sp_327);  sp_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_329 = self.getattr_L__mod___layer3___4___bns_2(sp_328);  sp_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_330 = self.getattr_L__mod___layer3___4___relu(sp_329);  sp_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_331 = sp_330 + getitem_91;  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_332 = self.getattr_L__mod___layer3___4___convs_3(sp_331);  sp_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_333 = self.getattr_L__mod___layer3___4___bns_3(sp_332);  sp_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_334 = self.getattr_L__mod___layer3___4___relu(sp_333);  sp_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_335 = sp_334 + getitem_92;  getitem_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_336 = self.getattr_L__mod___layer3___4___convs_4(sp_335);  sp_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_337 = self.getattr_L__mod___layer3___4___bns_4(sp_336);  sp_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_338 = self.getattr_L__mod___layer3___4___relu(sp_337);  sp_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_339 = sp_338 + getitem_93;  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_340 = self.getattr_L__mod___layer3___4___convs_5(sp_339);  sp_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_341 = self.getattr_L__mod___layer3___4___bns_5(sp_340);  sp_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_342 = self.getattr_L__mod___layer3___4___relu(sp_341);  sp_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_343 = sp_342 + getitem_94;  getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_344 = self.getattr_L__mod___layer3___4___convs_6(sp_343);  sp_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_345 = self.getattr_L__mod___layer3___4___bns_6(sp_344);  sp_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_346 = self.getattr_L__mod___layer3___4___relu(sp_345);  sp_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_91 = torch.cat([sp_322, sp_326, sp_330, sp_334, sp_338, sp_342, sp_346, getitem_95], 1);  sp_322 = sp_326 = sp_330 = sp_334 = sp_338 = sp_342 = sp_346 = getitem_95 = None
    
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
    split_12 = torch.functional.split(out_98, 56, 1);  out_98 = None
    sp_348 = split_12[0]
    getitem_97 = split_12[1]
    getitem_98 = split_12[2]
    getitem_99 = split_12[3]
    getitem_100 = split_12[4]
    getitem_101 = split_12[5]
    getitem_102 = split_12[6]
    getitem_103 = split_12[7];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_349 = self.getattr_L__mod___layer3___5___convs_0(sp_348);  sp_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_350 = self.getattr_L__mod___layer3___5___bns_0(sp_349);  sp_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_351 = self.getattr_L__mod___layer3___5___relu(sp_350);  sp_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_352 = sp_351 + getitem_97;  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_353 = self.getattr_L__mod___layer3___5___convs_1(sp_352);  sp_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_354 = self.getattr_L__mod___layer3___5___bns_1(sp_353);  sp_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_355 = self.getattr_L__mod___layer3___5___relu(sp_354);  sp_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_356 = sp_355 + getitem_98;  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_357 = self.getattr_L__mod___layer3___5___convs_2(sp_356);  sp_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_358 = self.getattr_L__mod___layer3___5___bns_2(sp_357);  sp_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_359 = self.getattr_L__mod___layer3___5___relu(sp_358);  sp_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_360 = sp_359 + getitem_99;  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_361 = self.getattr_L__mod___layer3___5___convs_3(sp_360);  sp_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_362 = self.getattr_L__mod___layer3___5___bns_3(sp_361);  sp_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_363 = self.getattr_L__mod___layer3___5___relu(sp_362);  sp_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_364 = sp_363 + getitem_100;  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_365 = self.getattr_L__mod___layer3___5___convs_4(sp_364);  sp_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_366 = self.getattr_L__mod___layer3___5___bns_4(sp_365);  sp_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_367 = self.getattr_L__mod___layer3___5___relu(sp_366);  sp_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_368 = sp_367 + getitem_101;  getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_369 = self.getattr_L__mod___layer3___5___convs_5(sp_368);  sp_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_370 = self.getattr_L__mod___layer3___5___bns_5(sp_369);  sp_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_371 = self.getattr_L__mod___layer3___5___relu(sp_370);  sp_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_372 = sp_371 + getitem_102;  getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_373 = self.getattr_L__mod___layer3___5___convs_6(sp_372);  sp_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_374 = self.getattr_L__mod___layer3___5___bns_6(sp_373);  sp_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_375 = self.getattr_L__mod___layer3___5___relu(sp_374);  sp_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_99 = torch.cat([sp_351, sp_355, sp_359, sp_363, sp_367, sp_371, sp_375, getitem_103], 1);  sp_351 = sp_355 = sp_359 = sp_363 = sp_367 = sp_371 = sp_375 = getitem_103 = None
    
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
    split_13 = torch.functional.split(out_106, 112, 1);  out_106 = None
    sp_377 = split_13[0]
    sp_381 = split_13[1]
    sp_385 = split_13[2]
    sp_389 = split_13[3]
    sp_393 = split_13[4]
    sp_397 = split_13[5]
    sp_401 = split_13[6]
    getitem_111 = split_13[7];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_378 = self.getattr_L__mod___layer4___0___convs_0(sp_377);  sp_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_379 = self.getattr_L__mod___layer4___0___bns_0(sp_378);  sp_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_380 = self.getattr_L__mod___layer4___0___relu(sp_379);  sp_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_382 = self.getattr_L__mod___layer4___0___convs_1(sp_381);  sp_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_383 = self.getattr_L__mod___layer4___0___bns_1(sp_382);  sp_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_384 = self.getattr_L__mod___layer4___0___relu(sp_383);  sp_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_386 = self.getattr_L__mod___layer4___0___convs_2(sp_385);  sp_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_387 = self.getattr_L__mod___layer4___0___bns_2(sp_386);  sp_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_388 = self.getattr_L__mod___layer4___0___relu(sp_387);  sp_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_390 = self.getattr_L__mod___layer4___0___convs_3(sp_389);  sp_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_391 = self.getattr_L__mod___layer4___0___bns_3(sp_390);  sp_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_392 = self.getattr_L__mod___layer4___0___relu(sp_391);  sp_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_394 = self.getattr_L__mod___layer4___0___convs_4(sp_393);  sp_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_395 = self.getattr_L__mod___layer4___0___bns_4(sp_394);  sp_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_396 = self.getattr_L__mod___layer4___0___relu(sp_395);  sp_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_398 = self.getattr_L__mod___layer4___0___convs_5(sp_397);  sp_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_399 = self.getattr_L__mod___layer4___0___bns_5(sp_398);  sp_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_400 = self.getattr_L__mod___layer4___0___relu(sp_399);  sp_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_402 = self.getattr_L__mod___layer4___0___convs_6(sp_401);  sp_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_403 = self.getattr_L__mod___layer4___0___bns_6(sp_402);  sp_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_404 = self.getattr_L__mod___layer4___0___relu(sp_403);  sp_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getattr_l__mod___layer4___0___pool = self.getattr_L__mod___layer4___0___pool(getitem_111);  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_107 = torch.cat([sp_380, sp_384, sp_388, sp_392, sp_396, sp_400, sp_404, getattr_l__mod___layer4___0___pool], 1);  sp_380 = sp_384 = sp_388 = sp_392 = sp_396 = sp_400 = sp_404 = getattr_l__mod___layer4___0___pool = None
    
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
    split_14 = torch.functional.split(out_114, 112, 1);  out_114 = None
    sp_406 = split_14[0]
    getitem_113 = split_14[1]
    getitem_114 = split_14[2]
    getitem_115 = split_14[3]
    getitem_116 = split_14[4]
    getitem_117 = split_14[5]
    getitem_118 = split_14[6]
    getitem_119 = split_14[7];  split_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_407 = self.getattr_L__mod___layer4___1___convs_0(sp_406);  sp_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_408 = self.getattr_L__mod___layer4___1___bns_0(sp_407);  sp_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_409 = self.getattr_L__mod___layer4___1___relu(sp_408);  sp_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_410 = sp_409 + getitem_113;  getitem_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_411 = self.getattr_L__mod___layer4___1___convs_1(sp_410);  sp_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_412 = self.getattr_L__mod___layer4___1___bns_1(sp_411);  sp_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_413 = self.getattr_L__mod___layer4___1___relu(sp_412);  sp_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_414 = sp_413 + getitem_114;  getitem_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_415 = self.getattr_L__mod___layer4___1___convs_2(sp_414);  sp_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_416 = self.getattr_L__mod___layer4___1___bns_2(sp_415);  sp_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_417 = self.getattr_L__mod___layer4___1___relu(sp_416);  sp_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_418 = sp_417 + getitem_115;  getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_419 = self.getattr_L__mod___layer4___1___convs_3(sp_418);  sp_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_420 = self.getattr_L__mod___layer4___1___bns_3(sp_419);  sp_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_421 = self.getattr_L__mod___layer4___1___relu(sp_420);  sp_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_422 = sp_421 + getitem_116;  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_423 = self.getattr_L__mod___layer4___1___convs_4(sp_422);  sp_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_424 = self.getattr_L__mod___layer4___1___bns_4(sp_423);  sp_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_425 = self.getattr_L__mod___layer4___1___relu(sp_424);  sp_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_426 = sp_425 + getitem_117;  getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_427 = self.getattr_L__mod___layer4___1___convs_5(sp_426);  sp_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_428 = self.getattr_L__mod___layer4___1___bns_5(sp_427);  sp_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_429 = self.getattr_L__mod___layer4___1___relu(sp_428);  sp_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_430 = sp_429 + getitem_118;  getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_431 = self.getattr_L__mod___layer4___1___convs_6(sp_430);  sp_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_432 = self.getattr_L__mod___layer4___1___bns_6(sp_431);  sp_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_433 = self.getattr_L__mod___layer4___1___relu(sp_432);  sp_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_115 = torch.cat([sp_409, sp_413, sp_417, sp_421, sp_425, sp_429, sp_433, getitem_119], 1);  sp_409 = sp_413 = sp_417 = sp_421 = sp_425 = sp_429 = sp_433 = getitem_119 = None
    
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
    split_15 = torch.functional.split(out_122, 112, 1);  out_122 = None
    sp_435 = split_15[0]
    getitem_121 = split_15[1]
    getitem_122 = split_15[2]
    getitem_123 = split_15[3]
    getitem_124 = split_15[4]
    getitem_125 = split_15[5]
    getitem_126 = split_15[6]
    getitem_127 = split_15[7];  split_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_436 = self.getattr_L__mod___layer4___2___convs_0(sp_435);  sp_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_437 = self.getattr_L__mod___layer4___2___bns_0(sp_436);  sp_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_438 = self.getattr_L__mod___layer4___2___relu(sp_437);  sp_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_439 = sp_438 + getitem_121;  getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_440 = self.getattr_L__mod___layer4___2___convs_1(sp_439);  sp_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_441 = self.getattr_L__mod___layer4___2___bns_1(sp_440);  sp_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_442 = self.getattr_L__mod___layer4___2___relu(sp_441);  sp_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_443 = sp_442 + getitem_122;  getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_444 = self.getattr_L__mod___layer4___2___convs_2(sp_443);  sp_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_445 = self.getattr_L__mod___layer4___2___bns_2(sp_444);  sp_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_446 = self.getattr_L__mod___layer4___2___relu(sp_445);  sp_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_447 = sp_446 + getitem_123;  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_448 = self.getattr_L__mod___layer4___2___convs_3(sp_447);  sp_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_449 = self.getattr_L__mod___layer4___2___bns_3(sp_448);  sp_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_450 = self.getattr_L__mod___layer4___2___relu(sp_449);  sp_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_451 = sp_450 + getitem_124;  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_452 = self.getattr_L__mod___layer4___2___convs_4(sp_451);  sp_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_453 = self.getattr_L__mod___layer4___2___bns_4(sp_452);  sp_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_454 = self.getattr_L__mod___layer4___2___relu(sp_453);  sp_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_455 = sp_454 + getitem_125;  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_456 = self.getattr_L__mod___layer4___2___convs_5(sp_455);  sp_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_457 = self.getattr_L__mod___layer4___2___bns_5(sp_456);  sp_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_458 = self.getattr_L__mod___layer4___2___relu(sp_457);  sp_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    sp_459 = sp_458 + getitem_126;  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    sp_460 = self.getattr_L__mod___layer4___2___convs_6(sp_459);  sp_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    sp_461 = self.getattr_L__mod___layer4___2___bns_6(sp_460);  sp_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    sp_462 = self.getattr_L__mod___layer4___2___relu(sp_461);  sp_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    out_123 = torch.cat([sp_438, sp_442, sp_446, sp_450, sp_454, sp_458, sp_462, getitem_127], 1);  sp_438 = sp_442 = sp_446 = sp_450 = sp_454 = sp_458 = sp_462 = getitem_127 = None
    
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
    