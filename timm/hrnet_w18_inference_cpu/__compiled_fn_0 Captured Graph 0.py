from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:762, code: x = self.conv1(x)
    x = self.L__mod___conv1(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:763, code: x = self.bn1(x)
    x_1 = self.L__mod___bn1(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:764, code: x = self.act1(x)
    x_2 = self.L__mod___act1(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:765, code: x = self.conv2(x)
    x_3 = self.L__mod___conv2(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:766, code: x = self.bn2(x)
    x_4 = self.L__mod___bn2(x_3);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:767, code: x = self.act2(x)
    shortcut = self.L__mod___act2(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_6 = self.getattr_L__mod___layer1___0___conv1(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_7 = self.getattr_L__mod___layer1___0___bn1(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_8 = self.getattr_L__mod___layer1___0___act1(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_9 = self.getattr_L__mod___layer1___0___conv2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_10 = self.getattr_L__mod___layer1___0___bn2(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_11 = self.getattr_L__mod___layer1___0___drop_block(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_12 = self.getattr_L__mod___layer1___0___act2(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_13 = self.getattr_L__mod___layer1___0___aa(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_14 = self.getattr_L__mod___layer1___0___conv3(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_15 = self.getattr_L__mod___layer1___0___bn3(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___layer1___0___downsample_0 = self.getattr_L__mod___layer1___0___downsample_0(shortcut);  shortcut = None
    shortcut_1 = self.getattr_L__mod___layer1___0___downsample_1(getattr_l__mod___layer1___0___downsample_0);  getattr_l__mod___layer1___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_15 += shortcut_1;  x_16 = x_15;  x_15 = shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_2 = self.getattr_L__mod___layer1___0___act3(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_18 = self.getattr_L__mod___layer1___1___conv1(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_19 = self.getattr_L__mod___layer1___1___bn1(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_20 = self.getattr_L__mod___layer1___1___act1(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_21 = self.getattr_L__mod___layer1___1___conv2(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_22 = self.getattr_L__mod___layer1___1___bn2(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_23 = self.getattr_L__mod___layer1___1___drop_block(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_24 = self.getattr_L__mod___layer1___1___act2(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_25 = self.getattr_L__mod___layer1___1___aa(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_26 = self.getattr_L__mod___layer1___1___conv3(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_27 = self.getattr_L__mod___layer1___1___bn3(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_27 += shortcut_2;  x_28 = x_27;  x_27 = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_3 = self.getattr_L__mod___layer1___1___act3(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_30 = self.getattr_L__mod___layer1___2___conv1(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_31 = self.getattr_L__mod___layer1___2___bn1(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_32 = self.getattr_L__mod___layer1___2___act1(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_33 = self.getattr_L__mod___layer1___2___conv2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_34 = self.getattr_L__mod___layer1___2___bn2(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_35 = self.getattr_L__mod___layer1___2___drop_block(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_36 = self.getattr_L__mod___layer1___2___act2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_37 = self.getattr_L__mod___layer1___2___aa(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_38 = self.getattr_L__mod___layer1___2___conv3(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_39 = self.getattr_L__mod___layer1___2___bn3(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_39 += shortcut_3;  x_40 = x_39;  x_39 = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_4 = self.getattr_L__mod___layer1___2___act3(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_42 = self.getattr_L__mod___layer1___3___conv1(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_43 = self.getattr_L__mod___layer1___3___bn1(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_44 = self.getattr_L__mod___layer1___3___act1(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_45 = self.getattr_L__mod___layer1___3___conv2(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_46 = self.getattr_L__mod___layer1___3___bn2(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_47 = self.getattr_L__mod___layer1___3___drop_block(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_48 = self.getattr_L__mod___layer1___3___act2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_49 = self.getattr_L__mod___layer1___3___aa(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_50 = self.getattr_L__mod___layer1___3___conv3(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_51 = self.getattr_L__mod___layer1___3___bn3(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_51 += shortcut_4;  x_52 = x_51;  x_51 = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    x_53 = self.getattr_L__mod___layer1___3___act3(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:750, code: xl = [t(x) for i, t in enumerate(self.transition1)]
    l__mod___transition1_0_0 = self.L__mod___transition1_0_0(x_53)
    l__mod___transition1_0_1 = self.L__mod___transition1_0_1(l__mod___transition1_0_0);  l__mod___transition1_0_0 = None
    shortcut_5 = self.L__mod___transition1_0_2(l__mod___transition1_0_1);  l__mod___transition1_0_1 = None
    l__mod___transition1_1_0_0 = self.L__mod___transition1_1_0_0(x_53);  x_53 = None
    l__mod___transition1_1_0_1 = self.L__mod___transition1_1_0_1(l__mod___transition1_1_0_0);  l__mod___transition1_1_0_0 = None
    shortcut_9 = self.L__mod___transition1_1_0_2(l__mod___transition1_1_0_1);  l__mod___transition1_1_0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_54 = self.getattr_L__mod___stage2_0_branches_0___0___conv1(shortcut_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_55 = self.getattr_L__mod___stage2_0_branches_0___0___bn1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_56 = self.getattr_L__mod___stage2_0_branches_0___0___drop_block(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_57 = self.getattr_L__mod___stage2_0_branches_0___0___act1(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_58 = self.getattr_L__mod___stage2_0_branches_0___0___aa(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_59 = self.getattr_L__mod___stage2_0_branches_0___0___conv2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_60 = self.getattr_L__mod___stage2_0_branches_0___0___bn2(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_60 += shortcut_5;  x_61 = x_60;  x_60 = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_6 = self.getattr_L__mod___stage2_0_branches_0___0___act2(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_63 = self.getattr_L__mod___stage2_0_branches_0___1___conv1(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_64 = self.getattr_L__mod___stage2_0_branches_0___1___bn1(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_65 = self.getattr_L__mod___stage2_0_branches_0___1___drop_block(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_66 = self.getattr_L__mod___stage2_0_branches_0___1___act1(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_67 = self.getattr_L__mod___stage2_0_branches_0___1___aa(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_68 = self.getattr_L__mod___stage2_0_branches_0___1___conv2(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_69 = self.getattr_L__mod___stage2_0_branches_0___1___bn2(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_69 += shortcut_6;  x_70 = x_69;  x_69 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_7 = self.getattr_L__mod___stage2_0_branches_0___1___act2(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_72 = self.getattr_L__mod___stage2_0_branches_0___2___conv1(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_73 = self.getattr_L__mod___stage2_0_branches_0___2___bn1(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_74 = self.getattr_L__mod___stage2_0_branches_0___2___drop_block(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_75 = self.getattr_L__mod___stage2_0_branches_0___2___act1(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_76 = self.getattr_L__mod___stage2_0_branches_0___2___aa(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_77 = self.getattr_L__mod___stage2_0_branches_0___2___conv2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_78 = self.getattr_L__mod___stage2_0_branches_0___2___bn2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_78 += shortcut_7;  x_79 = x_78;  x_78 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_8 = self.getattr_L__mod___stage2_0_branches_0___2___act2(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_81 = self.getattr_L__mod___stage2_0_branches_0___3___conv1(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_82 = self.getattr_L__mod___stage2_0_branches_0___3___bn1(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_83 = self.getattr_L__mod___stage2_0_branches_0___3___drop_block(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_84 = self.getattr_L__mod___stage2_0_branches_0___3___act1(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_85 = self.getattr_L__mod___stage2_0_branches_0___3___aa(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_86 = self.getattr_L__mod___stage2_0_branches_0___3___conv2(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_87 = self.getattr_L__mod___stage2_0_branches_0___3___bn2(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_87 += shortcut_8;  x_88 = x_87;  x_87 = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_89 = self.getattr_L__mod___stage2_0_branches_0___3___act2(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_90 = self.getattr_L__mod___stage2_0_branches_1___0___conv1(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_91 = self.getattr_L__mod___stage2_0_branches_1___0___bn1(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_92 = self.getattr_L__mod___stage2_0_branches_1___0___drop_block(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_93 = self.getattr_L__mod___stage2_0_branches_1___0___act1(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_94 = self.getattr_L__mod___stage2_0_branches_1___0___aa(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_95 = self.getattr_L__mod___stage2_0_branches_1___0___conv2(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_96 = self.getattr_L__mod___stage2_0_branches_1___0___bn2(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_96 += shortcut_9;  x_97 = x_96;  x_96 = shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_10 = self.getattr_L__mod___stage2_0_branches_1___0___act2(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_99 = self.getattr_L__mod___stage2_0_branches_1___1___conv1(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_100 = self.getattr_L__mod___stage2_0_branches_1___1___bn1(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_101 = self.getattr_L__mod___stage2_0_branches_1___1___drop_block(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_102 = self.getattr_L__mod___stage2_0_branches_1___1___act1(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_103 = self.getattr_L__mod___stage2_0_branches_1___1___aa(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_104 = self.getattr_L__mod___stage2_0_branches_1___1___conv2(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_105 = self.getattr_L__mod___stage2_0_branches_1___1___bn2(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_105 += shortcut_10;  x_106 = x_105;  x_105 = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_11 = self.getattr_L__mod___stage2_0_branches_1___1___act2(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_108 = self.getattr_L__mod___stage2_0_branches_1___2___conv1(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_109 = self.getattr_L__mod___stage2_0_branches_1___2___bn1(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_110 = self.getattr_L__mod___stage2_0_branches_1___2___drop_block(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_111 = self.getattr_L__mod___stage2_0_branches_1___2___act1(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_112 = self.getattr_L__mod___stage2_0_branches_1___2___aa(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_113 = self.getattr_L__mod___stage2_0_branches_1___2___conv2(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_114 = self.getattr_L__mod___stage2_0_branches_1___2___bn2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_114 += shortcut_11;  x_115 = x_114;  x_114 = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_12 = self.getattr_L__mod___stage2_0_branches_1___2___act2(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_117 = self.getattr_L__mod___stage2_0_branches_1___3___conv1(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_118 = self.getattr_L__mod___stage2_0_branches_1___3___bn1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_119 = self.getattr_L__mod___stage2_0_branches_1___3___drop_block(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_120 = self.getattr_L__mod___stage2_0_branches_1___3___act1(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_121 = self.getattr_L__mod___stage2_0_branches_1___3___aa(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_122 = self.getattr_L__mod___stage2_0_branches_1___3___conv2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_123 = self.getattr_L__mod___stage2_0_branches_1___3___bn2(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_123 += shortcut_12;  x_124 = x_123;  x_123 = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_125 = self.getattr_L__mod___stage2_0_branches_1___3___act2(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y = self.L__mod___stage2_0_fuse_layers_0_0(x_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage2_0_fuse_layers_0_1_0 = self.L__mod___stage2_0_fuse_layers_0_1_0(x_125)
    l__mod___stage2_0_fuse_layers_0_1_1 = self.L__mod___stage2_0_fuse_layers_0_1_1(l__mod___stage2_0_fuse_layers_0_1_0);  l__mod___stage2_0_fuse_layers_0_1_0 = None
    l__mod___stage2_0_fuse_layers_0_1_2 = self.L__mod___stage2_0_fuse_layers_0_1_2(l__mod___stage2_0_fuse_layers_0_1_1);  l__mod___stage2_0_fuse_layers_0_1_1 = None
    y_1 = y + l__mod___stage2_0_fuse_layers_0_1_2;  y = l__mod___stage2_0_fuse_layers_0_1_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_13 = self.L__mod___stage2_0_fuse_act(y_1);  y_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage2_0_fuse_layers_1_0_0_0 = self.L__mod___stage2_0_fuse_layers_1_0_0_0(x_89);  x_89 = None
    y_2 = self.L__mod___stage2_0_fuse_layers_1_0_0_1(l__mod___stage2_0_fuse_layers_1_0_0_0);  l__mod___stage2_0_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage2_0_fuse_layers_1_1 = self.L__mod___stage2_0_fuse_layers_1_1(x_125);  x_125 = None
    y_3 = y_2 + l__mod___stage2_0_fuse_layers_1_1;  y_2 = l__mod___stage2_0_fuse_layers_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_17 = self.L__mod___stage2_0_fuse_act(y_3);  y_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:753, code: xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition2)]
    l__mod___transition2_2_0_0 = self.L__mod___transition2_2_0_0(shortcut_17)
    l__mod___transition2_2_0_1 = self.L__mod___transition2_2_0_1(l__mod___transition2_2_0_0);  l__mod___transition2_2_0_0 = None
    shortcut_21 = self.L__mod___transition2_2_0_2(l__mod___transition2_2_0_1);  l__mod___transition2_2_0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_126 = self.getattr_L__mod___stage3_0_branches_0___0___conv1(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_127 = self.getattr_L__mod___stage3_0_branches_0___0___bn1(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_128 = self.getattr_L__mod___stage3_0_branches_0___0___drop_block(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_129 = self.getattr_L__mod___stage3_0_branches_0___0___act1(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_130 = self.getattr_L__mod___stage3_0_branches_0___0___aa(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_131 = self.getattr_L__mod___stage3_0_branches_0___0___conv2(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_132 = self.getattr_L__mod___stage3_0_branches_0___0___bn2(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_132 += shortcut_13;  x_133 = x_132;  x_132 = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_14 = self.getattr_L__mod___stage3_0_branches_0___0___act2(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_135 = self.getattr_L__mod___stage3_0_branches_0___1___conv1(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_136 = self.getattr_L__mod___stage3_0_branches_0___1___bn1(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_137 = self.getattr_L__mod___stage3_0_branches_0___1___drop_block(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_138 = self.getattr_L__mod___stage3_0_branches_0___1___act1(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_139 = self.getattr_L__mod___stage3_0_branches_0___1___aa(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_140 = self.getattr_L__mod___stage3_0_branches_0___1___conv2(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_141 = self.getattr_L__mod___stage3_0_branches_0___1___bn2(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_141 += shortcut_14;  x_142 = x_141;  x_141 = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_15 = self.getattr_L__mod___stage3_0_branches_0___1___act2(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_144 = self.getattr_L__mod___stage3_0_branches_0___2___conv1(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_145 = self.getattr_L__mod___stage3_0_branches_0___2___bn1(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_146 = self.getattr_L__mod___stage3_0_branches_0___2___drop_block(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_147 = self.getattr_L__mod___stage3_0_branches_0___2___act1(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_148 = self.getattr_L__mod___stage3_0_branches_0___2___aa(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_149 = self.getattr_L__mod___stage3_0_branches_0___2___conv2(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_150 = self.getattr_L__mod___stage3_0_branches_0___2___bn2(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_150 += shortcut_15;  x_151 = x_150;  x_150 = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_16 = self.getattr_L__mod___stage3_0_branches_0___2___act2(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_153 = self.getattr_L__mod___stage3_0_branches_0___3___conv1(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_154 = self.getattr_L__mod___stage3_0_branches_0___3___bn1(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_155 = self.getattr_L__mod___stage3_0_branches_0___3___drop_block(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_156 = self.getattr_L__mod___stage3_0_branches_0___3___act1(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_157 = self.getattr_L__mod___stage3_0_branches_0___3___aa(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_158 = self.getattr_L__mod___stage3_0_branches_0___3___conv2(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_159 = self.getattr_L__mod___stage3_0_branches_0___3___bn2(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_159 += shortcut_16;  x_160 = x_159;  x_159 = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_161 = self.getattr_L__mod___stage3_0_branches_0___3___act2(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_162 = self.getattr_L__mod___stage3_0_branches_1___0___conv1(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_163 = self.getattr_L__mod___stage3_0_branches_1___0___bn1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_164 = self.getattr_L__mod___stage3_0_branches_1___0___drop_block(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_165 = self.getattr_L__mod___stage3_0_branches_1___0___act1(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_166 = self.getattr_L__mod___stage3_0_branches_1___0___aa(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_167 = self.getattr_L__mod___stage3_0_branches_1___0___conv2(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_168 = self.getattr_L__mod___stage3_0_branches_1___0___bn2(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_168 += shortcut_17;  x_169 = x_168;  x_168 = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_18 = self.getattr_L__mod___stage3_0_branches_1___0___act2(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_171 = self.getattr_L__mod___stage3_0_branches_1___1___conv1(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_172 = self.getattr_L__mod___stage3_0_branches_1___1___bn1(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_173 = self.getattr_L__mod___stage3_0_branches_1___1___drop_block(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_174 = self.getattr_L__mod___stage3_0_branches_1___1___act1(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_175 = self.getattr_L__mod___stage3_0_branches_1___1___aa(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_176 = self.getattr_L__mod___stage3_0_branches_1___1___conv2(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_177 = self.getattr_L__mod___stage3_0_branches_1___1___bn2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_177 += shortcut_18;  x_178 = x_177;  x_177 = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_19 = self.getattr_L__mod___stage3_0_branches_1___1___act2(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_180 = self.getattr_L__mod___stage3_0_branches_1___2___conv1(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_181 = self.getattr_L__mod___stage3_0_branches_1___2___bn1(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_182 = self.getattr_L__mod___stage3_0_branches_1___2___drop_block(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_183 = self.getattr_L__mod___stage3_0_branches_1___2___act1(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_184 = self.getattr_L__mod___stage3_0_branches_1___2___aa(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_185 = self.getattr_L__mod___stage3_0_branches_1___2___conv2(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_186 = self.getattr_L__mod___stage3_0_branches_1___2___bn2(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_186 += shortcut_19;  x_187 = x_186;  x_186 = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_20 = self.getattr_L__mod___stage3_0_branches_1___2___act2(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_189 = self.getattr_L__mod___stage3_0_branches_1___3___conv1(shortcut_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_190 = self.getattr_L__mod___stage3_0_branches_1___3___bn1(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_191 = self.getattr_L__mod___stage3_0_branches_1___3___drop_block(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_192 = self.getattr_L__mod___stage3_0_branches_1___3___act1(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_193 = self.getattr_L__mod___stage3_0_branches_1___3___aa(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_194 = self.getattr_L__mod___stage3_0_branches_1___3___conv2(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_195 = self.getattr_L__mod___stage3_0_branches_1___3___bn2(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_195 += shortcut_20;  x_196 = x_195;  x_195 = shortcut_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_197 = self.getattr_L__mod___stage3_0_branches_1___3___act2(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_198 = self.getattr_L__mod___stage3_0_branches_2___0___conv1(shortcut_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_199 = self.getattr_L__mod___stage3_0_branches_2___0___bn1(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_200 = self.getattr_L__mod___stage3_0_branches_2___0___drop_block(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_201 = self.getattr_L__mod___stage3_0_branches_2___0___act1(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_202 = self.getattr_L__mod___stage3_0_branches_2___0___aa(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_203 = self.getattr_L__mod___stage3_0_branches_2___0___conv2(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_204 = self.getattr_L__mod___stage3_0_branches_2___0___bn2(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_204 += shortcut_21;  x_205 = x_204;  x_204 = shortcut_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_22 = self.getattr_L__mod___stage3_0_branches_2___0___act2(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_207 = self.getattr_L__mod___stage3_0_branches_2___1___conv1(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_208 = self.getattr_L__mod___stage3_0_branches_2___1___bn1(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_209 = self.getattr_L__mod___stage3_0_branches_2___1___drop_block(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_210 = self.getattr_L__mod___stage3_0_branches_2___1___act1(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_211 = self.getattr_L__mod___stage3_0_branches_2___1___aa(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_212 = self.getattr_L__mod___stage3_0_branches_2___1___conv2(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_213 = self.getattr_L__mod___stage3_0_branches_2___1___bn2(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_213 += shortcut_22;  x_214 = x_213;  x_213 = shortcut_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_23 = self.getattr_L__mod___stage3_0_branches_2___1___act2(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_216 = self.getattr_L__mod___stage3_0_branches_2___2___conv1(shortcut_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_217 = self.getattr_L__mod___stage3_0_branches_2___2___bn1(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_218 = self.getattr_L__mod___stage3_0_branches_2___2___drop_block(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_219 = self.getattr_L__mod___stage3_0_branches_2___2___act1(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_220 = self.getattr_L__mod___stage3_0_branches_2___2___aa(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_221 = self.getattr_L__mod___stage3_0_branches_2___2___conv2(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_222 = self.getattr_L__mod___stage3_0_branches_2___2___bn2(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_222 += shortcut_23;  x_223 = x_222;  x_222 = shortcut_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_24 = self.getattr_L__mod___stage3_0_branches_2___2___act2(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_225 = self.getattr_L__mod___stage3_0_branches_2___3___conv1(shortcut_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_226 = self.getattr_L__mod___stage3_0_branches_2___3___bn1(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_227 = self.getattr_L__mod___stage3_0_branches_2___3___drop_block(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_228 = self.getattr_L__mod___stage3_0_branches_2___3___act1(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_229 = self.getattr_L__mod___stage3_0_branches_2___3___aa(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_230 = self.getattr_L__mod___stage3_0_branches_2___3___conv2(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_231 = self.getattr_L__mod___stage3_0_branches_2___3___bn2(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_231 += shortcut_24;  x_232 = x_231;  x_231 = shortcut_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_233 = self.getattr_L__mod___stage3_0_branches_2___3___act2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_4 = self.L__mod___stage3_0_fuse_layers_0_0(x_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_0_fuse_layers_0_1_0 = self.L__mod___stage3_0_fuse_layers_0_1_0(x_197)
    l__mod___stage3_0_fuse_layers_0_1_1 = self.L__mod___stage3_0_fuse_layers_0_1_1(l__mod___stage3_0_fuse_layers_0_1_0);  l__mod___stage3_0_fuse_layers_0_1_0 = None
    l__mod___stage3_0_fuse_layers_0_1_2 = self.L__mod___stage3_0_fuse_layers_0_1_2(l__mod___stage3_0_fuse_layers_0_1_1);  l__mod___stage3_0_fuse_layers_0_1_1 = None
    y_5 = y_4 + l__mod___stage3_0_fuse_layers_0_1_2;  y_4 = l__mod___stage3_0_fuse_layers_0_1_2 = None
    l__mod___stage3_0_fuse_layers_0_2_0 = self.L__mod___stage3_0_fuse_layers_0_2_0(x_233)
    l__mod___stage3_0_fuse_layers_0_2_1 = self.L__mod___stage3_0_fuse_layers_0_2_1(l__mod___stage3_0_fuse_layers_0_2_0);  l__mod___stage3_0_fuse_layers_0_2_0 = None
    l__mod___stage3_0_fuse_layers_0_2_2 = self.L__mod___stage3_0_fuse_layers_0_2_2(l__mod___stage3_0_fuse_layers_0_2_1);  l__mod___stage3_0_fuse_layers_0_2_1 = None
    y_6 = y_5 + l__mod___stage3_0_fuse_layers_0_2_2;  y_5 = l__mod___stage3_0_fuse_layers_0_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_25 = self.L__mod___stage3_0_fuse_act(y_6);  y_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_0_fuse_layers_1_0_0_0 = self.L__mod___stage3_0_fuse_layers_1_0_0_0(x_161)
    y_7 = self.L__mod___stage3_0_fuse_layers_1_0_0_1(l__mod___stage3_0_fuse_layers_1_0_0_0);  l__mod___stage3_0_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_0_fuse_layers_1_1 = self.L__mod___stage3_0_fuse_layers_1_1(x_197)
    y_8 = y_7 + l__mod___stage3_0_fuse_layers_1_1;  y_7 = l__mod___stage3_0_fuse_layers_1_1 = None
    l__mod___stage3_0_fuse_layers_1_2_0 = self.L__mod___stage3_0_fuse_layers_1_2_0(x_233)
    l__mod___stage3_0_fuse_layers_1_2_1 = self.L__mod___stage3_0_fuse_layers_1_2_1(l__mod___stage3_0_fuse_layers_1_2_0);  l__mod___stage3_0_fuse_layers_1_2_0 = None
    l__mod___stage3_0_fuse_layers_1_2_2 = self.L__mod___stage3_0_fuse_layers_1_2_2(l__mod___stage3_0_fuse_layers_1_2_1);  l__mod___stage3_0_fuse_layers_1_2_1 = None
    y_9 = y_8 + l__mod___stage3_0_fuse_layers_1_2_2;  y_8 = l__mod___stage3_0_fuse_layers_1_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_29 = self.L__mod___stage3_0_fuse_act(y_9);  y_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_0_fuse_layers_2_0_0_0 = self.L__mod___stage3_0_fuse_layers_2_0_0_0(x_161);  x_161 = None
    l__mod___stage3_0_fuse_layers_2_0_0_1 = self.L__mod___stage3_0_fuse_layers_2_0_0_1(l__mod___stage3_0_fuse_layers_2_0_0_0);  l__mod___stage3_0_fuse_layers_2_0_0_0 = None
    l__mod___stage3_0_fuse_layers_2_0_0_2 = self.L__mod___stage3_0_fuse_layers_2_0_0_2(l__mod___stage3_0_fuse_layers_2_0_0_1);  l__mod___stage3_0_fuse_layers_2_0_0_1 = None
    l__mod___stage3_0_fuse_layers_2_0_1_0 = self.L__mod___stage3_0_fuse_layers_2_0_1_0(l__mod___stage3_0_fuse_layers_2_0_0_2);  l__mod___stage3_0_fuse_layers_2_0_0_2 = None
    y_10 = self.L__mod___stage3_0_fuse_layers_2_0_1_1(l__mod___stage3_0_fuse_layers_2_0_1_0);  l__mod___stage3_0_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_0_fuse_layers_2_1_0_0 = self.L__mod___stage3_0_fuse_layers_2_1_0_0(x_197);  x_197 = None
    l__mod___stage3_0_fuse_layers_2_1_0_1 = self.L__mod___stage3_0_fuse_layers_2_1_0_1(l__mod___stage3_0_fuse_layers_2_1_0_0);  l__mod___stage3_0_fuse_layers_2_1_0_0 = None
    y_11 = y_10 + l__mod___stage3_0_fuse_layers_2_1_0_1;  y_10 = l__mod___stage3_0_fuse_layers_2_1_0_1 = None
    l__mod___stage3_0_fuse_layers_2_2 = self.L__mod___stage3_0_fuse_layers_2_2(x_233);  x_233 = None
    y_12 = y_11 + l__mod___stage3_0_fuse_layers_2_2;  y_11 = l__mod___stage3_0_fuse_layers_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_33 = self.L__mod___stage3_0_fuse_act(y_12);  y_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_234 = self.getattr_L__mod___stage3_1_branches_0___0___conv1(shortcut_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_235 = self.getattr_L__mod___stage3_1_branches_0___0___bn1(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_236 = self.getattr_L__mod___stage3_1_branches_0___0___drop_block(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_237 = self.getattr_L__mod___stage3_1_branches_0___0___act1(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_238 = self.getattr_L__mod___stage3_1_branches_0___0___aa(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_239 = self.getattr_L__mod___stage3_1_branches_0___0___conv2(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_240 = self.getattr_L__mod___stage3_1_branches_0___0___bn2(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_240 += shortcut_25;  x_241 = x_240;  x_240 = shortcut_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_26 = self.getattr_L__mod___stage3_1_branches_0___0___act2(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_243 = self.getattr_L__mod___stage3_1_branches_0___1___conv1(shortcut_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_244 = self.getattr_L__mod___stage3_1_branches_0___1___bn1(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_245 = self.getattr_L__mod___stage3_1_branches_0___1___drop_block(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_246 = self.getattr_L__mod___stage3_1_branches_0___1___act1(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_247 = self.getattr_L__mod___stage3_1_branches_0___1___aa(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_248 = self.getattr_L__mod___stage3_1_branches_0___1___conv2(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_249 = self.getattr_L__mod___stage3_1_branches_0___1___bn2(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_249 += shortcut_26;  x_250 = x_249;  x_249 = shortcut_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_27 = self.getattr_L__mod___stage3_1_branches_0___1___act2(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_252 = self.getattr_L__mod___stage3_1_branches_0___2___conv1(shortcut_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_253 = self.getattr_L__mod___stage3_1_branches_0___2___bn1(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_254 = self.getattr_L__mod___stage3_1_branches_0___2___drop_block(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_255 = self.getattr_L__mod___stage3_1_branches_0___2___act1(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_256 = self.getattr_L__mod___stage3_1_branches_0___2___aa(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_257 = self.getattr_L__mod___stage3_1_branches_0___2___conv2(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_258 = self.getattr_L__mod___stage3_1_branches_0___2___bn2(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_258 += shortcut_27;  x_259 = x_258;  x_258 = shortcut_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_28 = self.getattr_L__mod___stage3_1_branches_0___2___act2(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_261 = self.getattr_L__mod___stage3_1_branches_0___3___conv1(shortcut_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_262 = self.getattr_L__mod___stage3_1_branches_0___3___bn1(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_263 = self.getattr_L__mod___stage3_1_branches_0___3___drop_block(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_264 = self.getattr_L__mod___stage3_1_branches_0___3___act1(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_265 = self.getattr_L__mod___stage3_1_branches_0___3___aa(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_266 = self.getattr_L__mod___stage3_1_branches_0___3___conv2(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_267 = self.getattr_L__mod___stage3_1_branches_0___3___bn2(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_267 += shortcut_28;  x_268 = x_267;  x_267 = shortcut_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_269 = self.getattr_L__mod___stage3_1_branches_0___3___act2(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_270 = self.getattr_L__mod___stage3_1_branches_1___0___conv1(shortcut_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_271 = self.getattr_L__mod___stage3_1_branches_1___0___bn1(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_272 = self.getattr_L__mod___stage3_1_branches_1___0___drop_block(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_273 = self.getattr_L__mod___stage3_1_branches_1___0___act1(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_274 = self.getattr_L__mod___stage3_1_branches_1___0___aa(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_275 = self.getattr_L__mod___stage3_1_branches_1___0___conv2(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_276 = self.getattr_L__mod___stage3_1_branches_1___0___bn2(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_276 += shortcut_29;  x_277 = x_276;  x_276 = shortcut_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_30 = self.getattr_L__mod___stage3_1_branches_1___0___act2(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_279 = self.getattr_L__mod___stage3_1_branches_1___1___conv1(shortcut_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_280 = self.getattr_L__mod___stage3_1_branches_1___1___bn1(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_281 = self.getattr_L__mod___stage3_1_branches_1___1___drop_block(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_282 = self.getattr_L__mod___stage3_1_branches_1___1___act1(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_283 = self.getattr_L__mod___stage3_1_branches_1___1___aa(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_284 = self.getattr_L__mod___stage3_1_branches_1___1___conv2(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_285 = self.getattr_L__mod___stage3_1_branches_1___1___bn2(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_285 += shortcut_30;  x_286 = x_285;  x_285 = shortcut_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_31 = self.getattr_L__mod___stage3_1_branches_1___1___act2(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_288 = self.getattr_L__mod___stage3_1_branches_1___2___conv1(shortcut_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_289 = self.getattr_L__mod___stage3_1_branches_1___2___bn1(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_290 = self.getattr_L__mod___stage3_1_branches_1___2___drop_block(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_291 = self.getattr_L__mod___stage3_1_branches_1___2___act1(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_292 = self.getattr_L__mod___stage3_1_branches_1___2___aa(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_293 = self.getattr_L__mod___stage3_1_branches_1___2___conv2(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_294 = self.getattr_L__mod___stage3_1_branches_1___2___bn2(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_294 += shortcut_31;  x_295 = x_294;  x_294 = shortcut_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_32 = self.getattr_L__mod___stage3_1_branches_1___2___act2(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_297 = self.getattr_L__mod___stage3_1_branches_1___3___conv1(shortcut_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_298 = self.getattr_L__mod___stage3_1_branches_1___3___bn1(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_299 = self.getattr_L__mod___stage3_1_branches_1___3___drop_block(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_300 = self.getattr_L__mod___stage3_1_branches_1___3___act1(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_301 = self.getattr_L__mod___stage3_1_branches_1___3___aa(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_302 = self.getattr_L__mod___stage3_1_branches_1___3___conv2(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_303 = self.getattr_L__mod___stage3_1_branches_1___3___bn2(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_303 += shortcut_32;  x_304 = x_303;  x_303 = shortcut_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_305 = self.getattr_L__mod___stage3_1_branches_1___3___act2(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_306 = self.getattr_L__mod___stage3_1_branches_2___0___conv1(shortcut_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_307 = self.getattr_L__mod___stage3_1_branches_2___0___bn1(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_308 = self.getattr_L__mod___stage3_1_branches_2___0___drop_block(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_309 = self.getattr_L__mod___stage3_1_branches_2___0___act1(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_310 = self.getattr_L__mod___stage3_1_branches_2___0___aa(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_311 = self.getattr_L__mod___stage3_1_branches_2___0___conv2(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_312 = self.getattr_L__mod___stage3_1_branches_2___0___bn2(x_311);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_312 += shortcut_33;  x_313 = x_312;  x_312 = shortcut_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_34 = self.getattr_L__mod___stage3_1_branches_2___0___act2(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_315 = self.getattr_L__mod___stage3_1_branches_2___1___conv1(shortcut_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_316 = self.getattr_L__mod___stage3_1_branches_2___1___bn1(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_317 = self.getattr_L__mod___stage3_1_branches_2___1___drop_block(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_318 = self.getattr_L__mod___stage3_1_branches_2___1___act1(x_317);  x_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_319 = self.getattr_L__mod___stage3_1_branches_2___1___aa(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_320 = self.getattr_L__mod___stage3_1_branches_2___1___conv2(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_321 = self.getattr_L__mod___stage3_1_branches_2___1___bn2(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_321 += shortcut_34;  x_322 = x_321;  x_321 = shortcut_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_35 = self.getattr_L__mod___stage3_1_branches_2___1___act2(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_324 = self.getattr_L__mod___stage3_1_branches_2___2___conv1(shortcut_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_325 = self.getattr_L__mod___stage3_1_branches_2___2___bn1(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_326 = self.getattr_L__mod___stage3_1_branches_2___2___drop_block(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_327 = self.getattr_L__mod___stage3_1_branches_2___2___act1(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_328 = self.getattr_L__mod___stage3_1_branches_2___2___aa(x_327);  x_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_329 = self.getattr_L__mod___stage3_1_branches_2___2___conv2(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_330 = self.getattr_L__mod___stage3_1_branches_2___2___bn2(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_330 += shortcut_35;  x_331 = x_330;  x_330 = shortcut_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_36 = self.getattr_L__mod___stage3_1_branches_2___2___act2(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_333 = self.getattr_L__mod___stage3_1_branches_2___3___conv1(shortcut_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_334 = self.getattr_L__mod___stage3_1_branches_2___3___bn1(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_335 = self.getattr_L__mod___stage3_1_branches_2___3___drop_block(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_336 = self.getattr_L__mod___stage3_1_branches_2___3___act1(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_337 = self.getattr_L__mod___stage3_1_branches_2___3___aa(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_338 = self.getattr_L__mod___stage3_1_branches_2___3___conv2(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_339 = self.getattr_L__mod___stage3_1_branches_2___3___bn2(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_339 += shortcut_36;  x_340 = x_339;  x_339 = shortcut_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_341 = self.getattr_L__mod___stage3_1_branches_2___3___act2(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_13 = self.L__mod___stage3_1_fuse_layers_0_0(x_269)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_1_fuse_layers_0_1_0 = self.L__mod___stage3_1_fuse_layers_0_1_0(x_305)
    l__mod___stage3_1_fuse_layers_0_1_1 = self.L__mod___stage3_1_fuse_layers_0_1_1(l__mod___stage3_1_fuse_layers_0_1_0);  l__mod___stage3_1_fuse_layers_0_1_0 = None
    l__mod___stage3_1_fuse_layers_0_1_2 = self.L__mod___stage3_1_fuse_layers_0_1_2(l__mod___stage3_1_fuse_layers_0_1_1);  l__mod___stage3_1_fuse_layers_0_1_1 = None
    y_14 = y_13 + l__mod___stage3_1_fuse_layers_0_1_2;  y_13 = l__mod___stage3_1_fuse_layers_0_1_2 = None
    l__mod___stage3_1_fuse_layers_0_2_0 = self.L__mod___stage3_1_fuse_layers_0_2_0(x_341)
    l__mod___stage3_1_fuse_layers_0_2_1 = self.L__mod___stage3_1_fuse_layers_0_2_1(l__mod___stage3_1_fuse_layers_0_2_0);  l__mod___stage3_1_fuse_layers_0_2_0 = None
    l__mod___stage3_1_fuse_layers_0_2_2 = self.L__mod___stage3_1_fuse_layers_0_2_2(l__mod___stage3_1_fuse_layers_0_2_1);  l__mod___stage3_1_fuse_layers_0_2_1 = None
    y_15 = y_14 + l__mod___stage3_1_fuse_layers_0_2_2;  y_14 = l__mod___stage3_1_fuse_layers_0_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_37 = self.L__mod___stage3_1_fuse_act(y_15);  y_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_1_fuse_layers_1_0_0_0 = self.L__mod___stage3_1_fuse_layers_1_0_0_0(x_269)
    y_16 = self.L__mod___stage3_1_fuse_layers_1_0_0_1(l__mod___stage3_1_fuse_layers_1_0_0_0);  l__mod___stage3_1_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_1_fuse_layers_1_1 = self.L__mod___stage3_1_fuse_layers_1_1(x_305)
    y_17 = y_16 + l__mod___stage3_1_fuse_layers_1_1;  y_16 = l__mod___stage3_1_fuse_layers_1_1 = None
    l__mod___stage3_1_fuse_layers_1_2_0 = self.L__mod___stage3_1_fuse_layers_1_2_0(x_341)
    l__mod___stage3_1_fuse_layers_1_2_1 = self.L__mod___stage3_1_fuse_layers_1_2_1(l__mod___stage3_1_fuse_layers_1_2_0);  l__mod___stage3_1_fuse_layers_1_2_0 = None
    l__mod___stage3_1_fuse_layers_1_2_2 = self.L__mod___stage3_1_fuse_layers_1_2_2(l__mod___stage3_1_fuse_layers_1_2_1);  l__mod___stage3_1_fuse_layers_1_2_1 = None
    y_18 = y_17 + l__mod___stage3_1_fuse_layers_1_2_2;  y_17 = l__mod___stage3_1_fuse_layers_1_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_41 = self.L__mod___stage3_1_fuse_act(y_18);  y_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_1_fuse_layers_2_0_0_0 = self.L__mod___stage3_1_fuse_layers_2_0_0_0(x_269);  x_269 = None
    l__mod___stage3_1_fuse_layers_2_0_0_1 = self.L__mod___stage3_1_fuse_layers_2_0_0_1(l__mod___stage3_1_fuse_layers_2_0_0_0);  l__mod___stage3_1_fuse_layers_2_0_0_0 = None
    l__mod___stage3_1_fuse_layers_2_0_0_2 = self.L__mod___stage3_1_fuse_layers_2_0_0_2(l__mod___stage3_1_fuse_layers_2_0_0_1);  l__mod___stage3_1_fuse_layers_2_0_0_1 = None
    l__mod___stage3_1_fuse_layers_2_0_1_0 = self.L__mod___stage3_1_fuse_layers_2_0_1_0(l__mod___stage3_1_fuse_layers_2_0_0_2);  l__mod___stage3_1_fuse_layers_2_0_0_2 = None
    y_19 = self.L__mod___stage3_1_fuse_layers_2_0_1_1(l__mod___stage3_1_fuse_layers_2_0_1_0);  l__mod___stage3_1_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_1_fuse_layers_2_1_0_0 = self.L__mod___stage3_1_fuse_layers_2_1_0_0(x_305);  x_305 = None
    l__mod___stage3_1_fuse_layers_2_1_0_1 = self.L__mod___stage3_1_fuse_layers_2_1_0_1(l__mod___stage3_1_fuse_layers_2_1_0_0);  l__mod___stage3_1_fuse_layers_2_1_0_0 = None
    y_20 = y_19 + l__mod___stage3_1_fuse_layers_2_1_0_1;  y_19 = l__mod___stage3_1_fuse_layers_2_1_0_1 = None
    l__mod___stage3_1_fuse_layers_2_2 = self.L__mod___stage3_1_fuse_layers_2_2(x_341);  x_341 = None
    y_21 = y_20 + l__mod___stage3_1_fuse_layers_2_2;  y_20 = l__mod___stage3_1_fuse_layers_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_45 = self.L__mod___stage3_1_fuse_act(y_21);  y_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_342 = self.getattr_L__mod___stage3_2_branches_0___0___conv1(shortcut_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_343 = self.getattr_L__mod___stage3_2_branches_0___0___bn1(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_344 = self.getattr_L__mod___stage3_2_branches_0___0___drop_block(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_345 = self.getattr_L__mod___stage3_2_branches_0___0___act1(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_346 = self.getattr_L__mod___stage3_2_branches_0___0___aa(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_347 = self.getattr_L__mod___stage3_2_branches_0___0___conv2(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_348 = self.getattr_L__mod___stage3_2_branches_0___0___bn2(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_348 += shortcut_37;  x_349 = x_348;  x_348 = shortcut_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_38 = self.getattr_L__mod___stage3_2_branches_0___0___act2(x_349);  x_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_351 = self.getattr_L__mod___stage3_2_branches_0___1___conv1(shortcut_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_352 = self.getattr_L__mod___stage3_2_branches_0___1___bn1(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_353 = self.getattr_L__mod___stage3_2_branches_0___1___drop_block(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_354 = self.getattr_L__mod___stage3_2_branches_0___1___act1(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_355 = self.getattr_L__mod___stage3_2_branches_0___1___aa(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_356 = self.getattr_L__mod___stage3_2_branches_0___1___conv2(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_357 = self.getattr_L__mod___stage3_2_branches_0___1___bn2(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_357 += shortcut_38;  x_358 = x_357;  x_357 = shortcut_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_39 = self.getattr_L__mod___stage3_2_branches_0___1___act2(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_360 = self.getattr_L__mod___stage3_2_branches_0___2___conv1(shortcut_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_361 = self.getattr_L__mod___stage3_2_branches_0___2___bn1(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_362 = self.getattr_L__mod___stage3_2_branches_0___2___drop_block(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_363 = self.getattr_L__mod___stage3_2_branches_0___2___act1(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_364 = self.getattr_L__mod___stage3_2_branches_0___2___aa(x_363);  x_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_365 = self.getattr_L__mod___stage3_2_branches_0___2___conv2(x_364);  x_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_366 = self.getattr_L__mod___stage3_2_branches_0___2___bn2(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_366 += shortcut_39;  x_367 = x_366;  x_366 = shortcut_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_40 = self.getattr_L__mod___stage3_2_branches_0___2___act2(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_369 = self.getattr_L__mod___stage3_2_branches_0___3___conv1(shortcut_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_370 = self.getattr_L__mod___stage3_2_branches_0___3___bn1(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_371 = self.getattr_L__mod___stage3_2_branches_0___3___drop_block(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_372 = self.getattr_L__mod___stage3_2_branches_0___3___act1(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_373 = self.getattr_L__mod___stage3_2_branches_0___3___aa(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_374 = self.getattr_L__mod___stage3_2_branches_0___3___conv2(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_375 = self.getattr_L__mod___stage3_2_branches_0___3___bn2(x_374);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_375 += shortcut_40;  x_376 = x_375;  x_375 = shortcut_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_377 = self.getattr_L__mod___stage3_2_branches_0___3___act2(x_376);  x_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_378 = self.getattr_L__mod___stage3_2_branches_1___0___conv1(shortcut_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_379 = self.getattr_L__mod___stage3_2_branches_1___0___bn1(x_378);  x_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_380 = self.getattr_L__mod___stage3_2_branches_1___0___drop_block(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_381 = self.getattr_L__mod___stage3_2_branches_1___0___act1(x_380);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_382 = self.getattr_L__mod___stage3_2_branches_1___0___aa(x_381);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_383 = self.getattr_L__mod___stage3_2_branches_1___0___conv2(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_384 = self.getattr_L__mod___stage3_2_branches_1___0___bn2(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_384 += shortcut_41;  x_385 = x_384;  x_384 = shortcut_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_42 = self.getattr_L__mod___stage3_2_branches_1___0___act2(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_387 = self.getattr_L__mod___stage3_2_branches_1___1___conv1(shortcut_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_388 = self.getattr_L__mod___stage3_2_branches_1___1___bn1(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_389 = self.getattr_L__mod___stage3_2_branches_1___1___drop_block(x_388);  x_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_390 = self.getattr_L__mod___stage3_2_branches_1___1___act1(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_391 = self.getattr_L__mod___stage3_2_branches_1___1___aa(x_390);  x_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_392 = self.getattr_L__mod___stage3_2_branches_1___1___conv2(x_391);  x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_393 = self.getattr_L__mod___stage3_2_branches_1___1___bn2(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_393 += shortcut_42;  x_394 = x_393;  x_393 = shortcut_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_43 = self.getattr_L__mod___stage3_2_branches_1___1___act2(x_394);  x_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_396 = self.getattr_L__mod___stage3_2_branches_1___2___conv1(shortcut_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_397 = self.getattr_L__mod___stage3_2_branches_1___2___bn1(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_398 = self.getattr_L__mod___stage3_2_branches_1___2___drop_block(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_399 = self.getattr_L__mod___stage3_2_branches_1___2___act1(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_400 = self.getattr_L__mod___stage3_2_branches_1___2___aa(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_401 = self.getattr_L__mod___stage3_2_branches_1___2___conv2(x_400);  x_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_402 = self.getattr_L__mod___stage3_2_branches_1___2___bn2(x_401);  x_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_402 += shortcut_43;  x_403 = x_402;  x_402 = shortcut_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_44 = self.getattr_L__mod___stage3_2_branches_1___2___act2(x_403);  x_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_405 = self.getattr_L__mod___stage3_2_branches_1___3___conv1(shortcut_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_406 = self.getattr_L__mod___stage3_2_branches_1___3___bn1(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_407 = self.getattr_L__mod___stage3_2_branches_1___3___drop_block(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_408 = self.getattr_L__mod___stage3_2_branches_1___3___act1(x_407);  x_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_409 = self.getattr_L__mod___stage3_2_branches_1___3___aa(x_408);  x_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_410 = self.getattr_L__mod___stage3_2_branches_1___3___conv2(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_411 = self.getattr_L__mod___stage3_2_branches_1___3___bn2(x_410);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_411 += shortcut_44;  x_412 = x_411;  x_411 = shortcut_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_413 = self.getattr_L__mod___stage3_2_branches_1___3___act2(x_412);  x_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_414 = self.getattr_L__mod___stage3_2_branches_2___0___conv1(shortcut_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_415 = self.getattr_L__mod___stage3_2_branches_2___0___bn1(x_414);  x_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_416 = self.getattr_L__mod___stage3_2_branches_2___0___drop_block(x_415);  x_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_417 = self.getattr_L__mod___stage3_2_branches_2___0___act1(x_416);  x_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_418 = self.getattr_L__mod___stage3_2_branches_2___0___aa(x_417);  x_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_419 = self.getattr_L__mod___stage3_2_branches_2___0___conv2(x_418);  x_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_420 = self.getattr_L__mod___stage3_2_branches_2___0___bn2(x_419);  x_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_420 += shortcut_45;  x_421 = x_420;  x_420 = shortcut_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_46 = self.getattr_L__mod___stage3_2_branches_2___0___act2(x_421);  x_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_423 = self.getattr_L__mod___stage3_2_branches_2___1___conv1(shortcut_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_424 = self.getattr_L__mod___stage3_2_branches_2___1___bn1(x_423);  x_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_425 = self.getattr_L__mod___stage3_2_branches_2___1___drop_block(x_424);  x_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_426 = self.getattr_L__mod___stage3_2_branches_2___1___act1(x_425);  x_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_427 = self.getattr_L__mod___stage3_2_branches_2___1___aa(x_426);  x_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_428 = self.getattr_L__mod___stage3_2_branches_2___1___conv2(x_427);  x_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_429 = self.getattr_L__mod___stage3_2_branches_2___1___bn2(x_428);  x_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_429 += shortcut_46;  x_430 = x_429;  x_429 = shortcut_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_47 = self.getattr_L__mod___stage3_2_branches_2___1___act2(x_430);  x_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_432 = self.getattr_L__mod___stage3_2_branches_2___2___conv1(shortcut_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_433 = self.getattr_L__mod___stage3_2_branches_2___2___bn1(x_432);  x_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_434 = self.getattr_L__mod___stage3_2_branches_2___2___drop_block(x_433);  x_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_435 = self.getattr_L__mod___stage3_2_branches_2___2___act1(x_434);  x_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_436 = self.getattr_L__mod___stage3_2_branches_2___2___aa(x_435);  x_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_437 = self.getattr_L__mod___stage3_2_branches_2___2___conv2(x_436);  x_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_438 = self.getattr_L__mod___stage3_2_branches_2___2___bn2(x_437);  x_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_438 += shortcut_47;  x_439 = x_438;  x_438 = shortcut_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_48 = self.getattr_L__mod___stage3_2_branches_2___2___act2(x_439);  x_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_441 = self.getattr_L__mod___stage3_2_branches_2___3___conv1(shortcut_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_442 = self.getattr_L__mod___stage3_2_branches_2___3___bn1(x_441);  x_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_443 = self.getattr_L__mod___stage3_2_branches_2___3___drop_block(x_442);  x_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_444 = self.getattr_L__mod___stage3_2_branches_2___3___act1(x_443);  x_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_445 = self.getattr_L__mod___stage3_2_branches_2___3___aa(x_444);  x_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_446 = self.getattr_L__mod___stage3_2_branches_2___3___conv2(x_445);  x_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_447 = self.getattr_L__mod___stage3_2_branches_2___3___bn2(x_446);  x_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_447 += shortcut_48;  x_448 = x_447;  x_447 = shortcut_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_449 = self.getattr_L__mod___stage3_2_branches_2___3___act2(x_448);  x_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_22 = self.L__mod___stage3_2_fuse_layers_0_0(x_377)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_2_fuse_layers_0_1_0 = self.L__mod___stage3_2_fuse_layers_0_1_0(x_413)
    l__mod___stage3_2_fuse_layers_0_1_1 = self.L__mod___stage3_2_fuse_layers_0_1_1(l__mod___stage3_2_fuse_layers_0_1_0);  l__mod___stage3_2_fuse_layers_0_1_0 = None
    l__mod___stage3_2_fuse_layers_0_1_2 = self.L__mod___stage3_2_fuse_layers_0_1_2(l__mod___stage3_2_fuse_layers_0_1_1);  l__mod___stage3_2_fuse_layers_0_1_1 = None
    y_23 = y_22 + l__mod___stage3_2_fuse_layers_0_1_2;  y_22 = l__mod___stage3_2_fuse_layers_0_1_2 = None
    l__mod___stage3_2_fuse_layers_0_2_0 = self.L__mod___stage3_2_fuse_layers_0_2_0(x_449)
    l__mod___stage3_2_fuse_layers_0_2_1 = self.L__mod___stage3_2_fuse_layers_0_2_1(l__mod___stage3_2_fuse_layers_0_2_0);  l__mod___stage3_2_fuse_layers_0_2_0 = None
    l__mod___stage3_2_fuse_layers_0_2_2 = self.L__mod___stage3_2_fuse_layers_0_2_2(l__mod___stage3_2_fuse_layers_0_2_1);  l__mod___stage3_2_fuse_layers_0_2_1 = None
    y_24 = y_23 + l__mod___stage3_2_fuse_layers_0_2_2;  y_23 = l__mod___stage3_2_fuse_layers_0_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_49 = self.L__mod___stage3_2_fuse_act(y_24);  y_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_2_fuse_layers_1_0_0_0 = self.L__mod___stage3_2_fuse_layers_1_0_0_0(x_377)
    y_25 = self.L__mod___stage3_2_fuse_layers_1_0_0_1(l__mod___stage3_2_fuse_layers_1_0_0_0);  l__mod___stage3_2_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_2_fuse_layers_1_1 = self.L__mod___stage3_2_fuse_layers_1_1(x_413)
    y_26 = y_25 + l__mod___stage3_2_fuse_layers_1_1;  y_25 = l__mod___stage3_2_fuse_layers_1_1 = None
    l__mod___stage3_2_fuse_layers_1_2_0 = self.L__mod___stage3_2_fuse_layers_1_2_0(x_449)
    l__mod___stage3_2_fuse_layers_1_2_1 = self.L__mod___stage3_2_fuse_layers_1_2_1(l__mod___stage3_2_fuse_layers_1_2_0);  l__mod___stage3_2_fuse_layers_1_2_0 = None
    l__mod___stage3_2_fuse_layers_1_2_2 = self.L__mod___stage3_2_fuse_layers_1_2_2(l__mod___stage3_2_fuse_layers_1_2_1);  l__mod___stage3_2_fuse_layers_1_2_1 = None
    y_27 = y_26 + l__mod___stage3_2_fuse_layers_1_2_2;  y_26 = l__mod___stage3_2_fuse_layers_1_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_53 = self.L__mod___stage3_2_fuse_act(y_27);  y_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_2_fuse_layers_2_0_0_0 = self.L__mod___stage3_2_fuse_layers_2_0_0_0(x_377);  x_377 = None
    l__mod___stage3_2_fuse_layers_2_0_0_1 = self.L__mod___stage3_2_fuse_layers_2_0_0_1(l__mod___stage3_2_fuse_layers_2_0_0_0);  l__mod___stage3_2_fuse_layers_2_0_0_0 = None
    l__mod___stage3_2_fuse_layers_2_0_0_2 = self.L__mod___stage3_2_fuse_layers_2_0_0_2(l__mod___stage3_2_fuse_layers_2_0_0_1);  l__mod___stage3_2_fuse_layers_2_0_0_1 = None
    l__mod___stage3_2_fuse_layers_2_0_1_0 = self.L__mod___stage3_2_fuse_layers_2_0_1_0(l__mod___stage3_2_fuse_layers_2_0_0_2);  l__mod___stage3_2_fuse_layers_2_0_0_2 = None
    y_28 = self.L__mod___stage3_2_fuse_layers_2_0_1_1(l__mod___stage3_2_fuse_layers_2_0_1_0);  l__mod___stage3_2_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_2_fuse_layers_2_1_0_0 = self.L__mod___stage3_2_fuse_layers_2_1_0_0(x_413);  x_413 = None
    l__mod___stage3_2_fuse_layers_2_1_0_1 = self.L__mod___stage3_2_fuse_layers_2_1_0_1(l__mod___stage3_2_fuse_layers_2_1_0_0);  l__mod___stage3_2_fuse_layers_2_1_0_0 = None
    y_29 = y_28 + l__mod___stage3_2_fuse_layers_2_1_0_1;  y_28 = l__mod___stage3_2_fuse_layers_2_1_0_1 = None
    l__mod___stage3_2_fuse_layers_2_2 = self.L__mod___stage3_2_fuse_layers_2_2(x_449);  x_449 = None
    y_30 = y_29 + l__mod___stage3_2_fuse_layers_2_2;  y_29 = l__mod___stage3_2_fuse_layers_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_57 = self.L__mod___stage3_2_fuse_act(y_30);  y_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_450 = self.getattr_L__mod___stage3_3_branches_0___0___conv1(shortcut_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_451 = self.getattr_L__mod___stage3_3_branches_0___0___bn1(x_450);  x_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_452 = self.getattr_L__mod___stage3_3_branches_0___0___drop_block(x_451);  x_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_453 = self.getattr_L__mod___stage3_3_branches_0___0___act1(x_452);  x_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_454 = self.getattr_L__mod___stage3_3_branches_0___0___aa(x_453);  x_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_455 = self.getattr_L__mod___stage3_3_branches_0___0___conv2(x_454);  x_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_456 = self.getattr_L__mod___stage3_3_branches_0___0___bn2(x_455);  x_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_456 += shortcut_49;  x_457 = x_456;  x_456 = shortcut_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_50 = self.getattr_L__mod___stage3_3_branches_0___0___act2(x_457);  x_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_459 = self.getattr_L__mod___stage3_3_branches_0___1___conv1(shortcut_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_460 = self.getattr_L__mod___stage3_3_branches_0___1___bn1(x_459);  x_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_461 = self.getattr_L__mod___stage3_3_branches_0___1___drop_block(x_460);  x_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_462 = self.getattr_L__mod___stage3_3_branches_0___1___act1(x_461);  x_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_463 = self.getattr_L__mod___stage3_3_branches_0___1___aa(x_462);  x_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_464 = self.getattr_L__mod___stage3_3_branches_0___1___conv2(x_463);  x_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_465 = self.getattr_L__mod___stage3_3_branches_0___1___bn2(x_464);  x_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_465 += shortcut_50;  x_466 = x_465;  x_465 = shortcut_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_51 = self.getattr_L__mod___stage3_3_branches_0___1___act2(x_466);  x_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_468 = self.getattr_L__mod___stage3_3_branches_0___2___conv1(shortcut_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_469 = self.getattr_L__mod___stage3_3_branches_0___2___bn1(x_468);  x_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_470 = self.getattr_L__mod___stage3_3_branches_0___2___drop_block(x_469);  x_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_471 = self.getattr_L__mod___stage3_3_branches_0___2___act1(x_470);  x_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_472 = self.getattr_L__mod___stage3_3_branches_0___2___aa(x_471);  x_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_473 = self.getattr_L__mod___stage3_3_branches_0___2___conv2(x_472);  x_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_474 = self.getattr_L__mod___stage3_3_branches_0___2___bn2(x_473);  x_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_474 += shortcut_51;  x_475 = x_474;  x_474 = shortcut_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_52 = self.getattr_L__mod___stage3_3_branches_0___2___act2(x_475);  x_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_477 = self.getattr_L__mod___stage3_3_branches_0___3___conv1(shortcut_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_478 = self.getattr_L__mod___stage3_3_branches_0___3___bn1(x_477);  x_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_479 = self.getattr_L__mod___stage3_3_branches_0___3___drop_block(x_478);  x_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_480 = self.getattr_L__mod___stage3_3_branches_0___3___act1(x_479);  x_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_481 = self.getattr_L__mod___stage3_3_branches_0___3___aa(x_480);  x_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_482 = self.getattr_L__mod___stage3_3_branches_0___3___conv2(x_481);  x_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_483 = self.getattr_L__mod___stage3_3_branches_0___3___bn2(x_482);  x_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_483 += shortcut_52;  x_484 = x_483;  x_483 = shortcut_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_485 = self.getattr_L__mod___stage3_3_branches_0___3___act2(x_484);  x_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_486 = self.getattr_L__mod___stage3_3_branches_1___0___conv1(shortcut_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_487 = self.getattr_L__mod___stage3_3_branches_1___0___bn1(x_486);  x_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_488 = self.getattr_L__mod___stage3_3_branches_1___0___drop_block(x_487);  x_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_489 = self.getattr_L__mod___stage3_3_branches_1___0___act1(x_488);  x_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_490 = self.getattr_L__mod___stage3_3_branches_1___0___aa(x_489);  x_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_491 = self.getattr_L__mod___stage3_3_branches_1___0___conv2(x_490);  x_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_492 = self.getattr_L__mod___stage3_3_branches_1___0___bn2(x_491);  x_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_492 += shortcut_53;  x_493 = x_492;  x_492 = shortcut_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_54 = self.getattr_L__mod___stage3_3_branches_1___0___act2(x_493);  x_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_495 = self.getattr_L__mod___stage3_3_branches_1___1___conv1(shortcut_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_496 = self.getattr_L__mod___stage3_3_branches_1___1___bn1(x_495);  x_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_497 = self.getattr_L__mod___stage3_3_branches_1___1___drop_block(x_496);  x_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_498 = self.getattr_L__mod___stage3_3_branches_1___1___act1(x_497);  x_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_499 = self.getattr_L__mod___stage3_3_branches_1___1___aa(x_498);  x_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_500 = self.getattr_L__mod___stage3_3_branches_1___1___conv2(x_499);  x_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_501 = self.getattr_L__mod___stage3_3_branches_1___1___bn2(x_500);  x_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_501 += shortcut_54;  x_502 = x_501;  x_501 = shortcut_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_55 = self.getattr_L__mod___stage3_3_branches_1___1___act2(x_502);  x_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_504 = self.getattr_L__mod___stage3_3_branches_1___2___conv1(shortcut_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_505 = self.getattr_L__mod___stage3_3_branches_1___2___bn1(x_504);  x_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_506 = self.getattr_L__mod___stage3_3_branches_1___2___drop_block(x_505);  x_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_507 = self.getattr_L__mod___stage3_3_branches_1___2___act1(x_506);  x_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_508 = self.getattr_L__mod___stage3_3_branches_1___2___aa(x_507);  x_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_509 = self.getattr_L__mod___stage3_3_branches_1___2___conv2(x_508);  x_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_510 = self.getattr_L__mod___stage3_3_branches_1___2___bn2(x_509);  x_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_510 += shortcut_55;  x_511 = x_510;  x_510 = shortcut_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_56 = self.getattr_L__mod___stage3_3_branches_1___2___act2(x_511);  x_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_513 = self.getattr_L__mod___stage3_3_branches_1___3___conv1(shortcut_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_514 = self.getattr_L__mod___stage3_3_branches_1___3___bn1(x_513);  x_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_515 = self.getattr_L__mod___stage3_3_branches_1___3___drop_block(x_514);  x_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_516 = self.getattr_L__mod___stage3_3_branches_1___3___act1(x_515);  x_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_517 = self.getattr_L__mod___stage3_3_branches_1___3___aa(x_516);  x_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_518 = self.getattr_L__mod___stage3_3_branches_1___3___conv2(x_517);  x_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_519 = self.getattr_L__mod___stage3_3_branches_1___3___bn2(x_518);  x_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_519 += shortcut_56;  x_520 = x_519;  x_519 = shortcut_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_521 = self.getattr_L__mod___stage3_3_branches_1___3___act2(x_520);  x_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_522 = self.getattr_L__mod___stage3_3_branches_2___0___conv1(shortcut_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_523 = self.getattr_L__mod___stage3_3_branches_2___0___bn1(x_522);  x_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_524 = self.getattr_L__mod___stage3_3_branches_2___0___drop_block(x_523);  x_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_525 = self.getattr_L__mod___stage3_3_branches_2___0___act1(x_524);  x_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_526 = self.getattr_L__mod___stage3_3_branches_2___0___aa(x_525);  x_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_527 = self.getattr_L__mod___stage3_3_branches_2___0___conv2(x_526);  x_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_528 = self.getattr_L__mod___stage3_3_branches_2___0___bn2(x_527);  x_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_528 += shortcut_57;  x_529 = x_528;  x_528 = shortcut_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_58 = self.getattr_L__mod___stage3_3_branches_2___0___act2(x_529);  x_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_531 = self.getattr_L__mod___stage3_3_branches_2___1___conv1(shortcut_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_532 = self.getattr_L__mod___stage3_3_branches_2___1___bn1(x_531);  x_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_533 = self.getattr_L__mod___stage3_3_branches_2___1___drop_block(x_532);  x_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_534 = self.getattr_L__mod___stage3_3_branches_2___1___act1(x_533);  x_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_535 = self.getattr_L__mod___stage3_3_branches_2___1___aa(x_534);  x_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_536 = self.getattr_L__mod___stage3_3_branches_2___1___conv2(x_535);  x_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_537 = self.getattr_L__mod___stage3_3_branches_2___1___bn2(x_536);  x_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_537 += shortcut_58;  x_538 = x_537;  x_537 = shortcut_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_59 = self.getattr_L__mod___stage3_3_branches_2___1___act2(x_538);  x_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_540 = self.getattr_L__mod___stage3_3_branches_2___2___conv1(shortcut_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_541 = self.getattr_L__mod___stage3_3_branches_2___2___bn1(x_540);  x_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_542 = self.getattr_L__mod___stage3_3_branches_2___2___drop_block(x_541);  x_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_543 = self.getattr_L__mod___stage3_3_branches_2___2___act1(x_542);  x_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_544 = self.getattr_L__mod___stage3_3_branches_2___2___aa(x_543);  x_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_545 = self.getattr_L__mod___stage3_3_branches_2___2___conv2(x_544);  x_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_546 = self.getattr_L__mod___stage3_3_branches_2___2___bn2(x_545);  x_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_546 += shortcut_59;  x_547 = x_546;  x_546 = shortcut_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_60 = self.getattr_L__mod___stage3_3_branches_2___2___act2(x_547);  x_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_549 = self.getattr_L__mod___stage3_3_branches_2___3___conv1(shortcut_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_550 = self.getattr_L__mod___stage3_3_branches_2___3___bn1(x_549);  x_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_551 = self.getattr_L__mod___stage3_3_branches_2___3___drop_block(x_550);  x_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_552 = self.getattr_L__mod___stage3_3_branches_2___3___act1(x_551);  x_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_553 = self.getattr_L__mod___stage3_3_branches_2___3___aa(x_552);  x_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_554 = self.getattr_L__mod___stage3_3_branches_2___3___conv2(x_553);  x_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_555 = self.getattr_L__mod___stage3_3_branches_2___3___bn2(x_554);  x_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_555 += shortcut_60;  x_556 = x_555;  x_555 = shortcut_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_557 = self.getattr_L__mod___stage3_3_branches_2___3___act2(x_556);  x_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_31 = self.L__mod___stage3_3_fuse_layers_0_0(x_485)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_3_fuse_layers_0_1_0 = self.L__mod___stage3_3_fuse_layers_0_1_0(x_521)
    l__mod___stage3_3_fuse_layers_0_1_1 = self.L__mod___stage3_3_fuse_layers_0_1_1(l__mod___stage3_3_fuse_layers_0_1_0);  l__mod___stage3_3_fuse_layers_0_1_0 = None
    l__mod___stage3_3_fuse_layers_0_1_2 = self.L__mod___stage3_3_fuse_layers_0_1_2(l__mod___stage3_3_fuse_layers_0_1_1);  l__mod___stage3_3_fuse_layers_0_1_1 = None
    y_32 = y_31 + l__mod___stage3_3_fuse_layers_0_1_2;  y_31 = l__mod___stage3_3_fuse_layers_0_1_2 = None
    l__mod___stage3_3_fuse_layers_0_2_0 = self.L__mod___stage3_3_fuse_layers_0_2_0(x_557)
    l__mod___stage3_3_fuse_layers_0_2_1 = self.L__mod___stage3_3_fuse_layers_0_2_1(l__mod___stage3_3_fuse_layers_0_2_0);  l__mod___stage3_3_fuse_layers_0_2_0 = None
    l__mod___stage3_3_fuse_layers_0_2_2 = self.L__mod___stage3_3_fuse_layers_0_2_2(l__mod___stage3_3_fuse_layers_0_2_1);  l__mod___stage3_3_fuse_layers_0_2_1 = None
    y_33 = y_32 + l__mod___stage3_3_fuse_layers_0_2_2;  y_32 = l__mod___stage3_3_fuse_layers_0_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_61 = self.L__mod___stage3_3_fuse_act(y_33);  y_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_3_fuse_layers_1_0_0_0 = self.L__mod___stage3_3_fuse_layers_1_0_0_0(x_485)
    y_34 = self.L__mod___stage3_3_fuse_layers_1_0_0_1(l__mod___stage3_3_fuse_layers_1_0_0_0);  l__mod___stage3_3_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_3_fuse_layers_1_1 = self.L__mod___stage3_3_fuse_layers_1_1(x_521)
    y_35 = y_34 + l__mod___stage3_3_fuse_layers_1_1;  y_34 = l__mod___stage3_3_fuse_layers_1_1 = None
    l__mod___stage3_3_fuse_layers_1_2_0 = self.L__mod___stage3_3_fuse_layers_1_2_0(x_557)
    l__mod___stage3_3_fuse_layers_1_2_1 = self.L__mod___stage3_3_fuse_layers_1_2_1(l__mod___stage3_3_fuse_layers_1_2_0);  l__mod___stage3_3_fuse_layers_1_2_0 = None
    l__mod___stage3_3_fuse_layers_1_2_2 = self.L__mod___stage3_3_fuse_layers_1_2_2(l__mod___stage3_3_fuse_layers_1_2_1);  l__mod___stage3_3_fuse_layers_1_2_1 = None
    y_36 = y_35 + l__mod___stage3_3_fuse_layers_1_2_2;  y_35 = l__mod___stage3_3_fuse_layers_1_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_65 = self.L__mod___stage3_3_fuse_act(y_36);  y_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage3_3_fuse_layers_2_0_0_0 = self.L__mod___stage3_3_fuse_layers_2_0_0_0(x_485);  x_485 = None
    l__mod___stage3_3_fuse_layers_2_0_0_1 = self.L__mod___stage3_3_fuse_layers_2_0_0_1(l__mod___stage3_3_fuse_layers_2_0_0_0);  l__mod___stage3_3_fuse_layers_2_0_0_0 = None
    l__mod___stage3_3_fuse_layers_2_0_0_2 = self.L__mod___stage3_3_fuse_layers_2_0_0_2(l__mod___stage3_3_fuse_layers_2_0_0_1);  l__mod___stage3_3_fuse_layers_2_0_0_1 = None
    l__mod___stage3_3_fuse_layers_2_0_1_0 = self.L__mod___stage3_3_fuse_layers_2_0_1_0(l__mod___stage3_3_fuse_layers_2_0_0_2);  l__mod___stage3_3_fuse_layers_2_0_0_2 = None
    y_37 = self.L__mod___stage3_3_fuse_layers_2_0_1_1(l__mod___stage3_3_fuse_layers_2_0_1_0);  l__mod___stage3_3_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage3_3_fuse_layers_2_1_0_0 = self.L__mod___stage3_3_fuse_layers_2_1_0_0(x_521);  x_521 = None
    l__mod___stage3_3_fuse_layers_2_1_0_1 = self.L__mod___stage3_3_fuse_layers_2_1_0_1(l__mod___stage3_3_fuse_layers_2_1_0_0);  l__mod___stage3_3_fuse_layers_2_1_0_0 = None
    y_38 = y_37 + l__mod___stage3_3_fuse_layers_2_1_0_1;  y_37 = l__mod___stage3_3_fuse_layers_2_1_0_1 = None
    l__mod___stage3_3_fuse_layers_2_2 = self.L__mod___stage3_3_fuse_layers_2_2(x_557);  x_557 = None
    y_39 = y_38 + l__mod___stage3_3_fuse_layers_2_2;  y_38 = l__mod___stage3_3_fuse_layers_2_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_69 = self.L__mod___stage3_3_fuse_act(y_39);  y_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:756, code: xl = [t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i] for i, t in enumerate(self.transition3)]
    l__mod___transition3_3_0_0 = self.L__mod___transition3_3_0_0(shortcut_69)
    l__mod___transition3_3_0_1 = self.L__mod___transition3_3_0_1(l__mod___transition3_3_0_0);  l__mod___transition3_3_0_0 = None
    shortcut_73 = self.L__mod___transition3_3_0_2(l__mod___transition3_3_0_1);  l__mod___transition3_3_0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_558 = self.getattr_L__mod___stage4_0_branches_0___0___conv1(shortcut_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_559 = self.getattr_L__mod___stage4_0_branches_0___0___bn1(x_558);  x_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_560 = self.getattr_L__mod___stage4_0_branches_0___0___drop_block(x_559);  x_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_561 = self.getattr_L__mod___stage4_0_branches_0___0___act1(x_560);  x_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_562 = self.getattr_L__mod___stage4_0_branches_0___0___aa(x_561);  x_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_563 = self.getattr_L__mod___stage4_0_branches_0___0___conv2(x_562);  x_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_564 = self.getattr_L__mod___stage4_0_branches_0___0___bn2(x_563);  x_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_564 += shortcut_61;  x_565 = x_564;  x_564 = shortcut_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_62 = self.getattr_L__mod___stage4_0_branches_0___0___act2(x_565);  x_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_567 = self.getattr_L__mod___stage4_0_branches_0___1___conv1(shortcut_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_568 = self.getattr_L__mod___stage4_0_branches_0___1___bn1(x_567);  x_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_569 = self.getattr_L__mod___stage4_0_branches_0___1___drop_block(x_568);  x_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_570 = self.getattr_L__mod___stage4_0_branches_0___1___act1(x_569);  x_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_571 = self.getattr_L__mod___stage4_0_branches_0___1___aa(x_570);  x_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_572 = self.getattr_L__mod___stage4_0_branches_0___1___conv2(x_571);  x_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_573 = self.getattr_L__mod___stage4_0_branches_0___1___bn2(x_572);  x_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_573 += shortcut_62;  x_574 = x_573;  x_573 = shortcut_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_63 = self.getattr_L__mod___stage4_0_branches_0___1___act2(x_574);  x_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_576 = self.getattr_L__mod___stage4_0_branches_0___2___conv1(shortcut_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_577 = self.getattr_L__mod___stage4_0_branches_0___2___bn1(x_576);  x_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_578 = self.getattr_L__mod___stage4_0_branches_0___2___drop_block(x_577);  x_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_579 = self.getattr_L__mod___stage4_0_branches_0___2___act1(x_578);  x_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_580 = self.getattr_L__mod___stage4_0_branches_0___2___aa(x_579);  x_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_581 = self.getattr_L__mod___stage4_0_branches_0___2___conv2(x_580);  x_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_582 = self.getattr_L__mod___stage4_0_branches_0___2___bn2(x_581);  x_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_582 += shortcut_63;  x_583 = x_582;  x_582 = shortcut_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_64 = self.getattr_L__mod___stage4_0_branches_0___2___act2(x_583);  x_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_585 = self.getattr_L__mod___stage4_0_branches_0___3___conv1(shortcut_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_586 = self.getattr_L__mod___stage4_0_branches_0___3___bn1(x_585);  x_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_587 = self.getattr_L__mod___stage4_0_branches_0___3___drop_block(x_586);  x_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_588 = self.getattr_L__mod___stage4_0_branches_0___3___act1(x_587);  x_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_589 = self.getattr_L__mod___stage4_0_branches_0___3___aa(x_588);  x_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_590 = self.getattr_L__mod___stage4_0_branches_0___3___conv2(x_589);  x_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_591 = self.getattr_L__mod___stage4_0_branches_0___3___bn2(x_590);  x_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_591 += shortcut_64;  x_592 = x_591;  x_591 = shortcut_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_593 = self.getattr_L__mod___stage4_0_branches_0___3___act2(x_592);  x_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_594 = self.getattr_L__mod___stage4_0_branches_1___0___conv1(shortcut_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_595 = self.getattr_L__mod___stage4_0_branches_1___0___bn1(x_594);  x_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_596 = self.getattr_L__mod___stage4_0_branches_1___0___drop_block(x_595);  x_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_597 = self.getattr_L__mod___stage4_0_branches_1___0___act1(x_596);  x_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_598 = self.getattr_L__mod___stage4_0_branches_1___0___aa(x_597);  x_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_599 = self.getattr_L__mod___stage4_0_branches_1___0___conv2(x_598);  x_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_600 = self.getattr_L__mod___stage4_0_branches_1___0___bn2(x_599);  x_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_600 += shortcut_65;  x_601 = x_600;  x_600 = shortcut_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_66 = self.getattr_L__mod___stage4_0_branches_1___0___act2(x_601);  x_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_603 = self.getattr_L__mod___stage4_0_branches_1___1___conv1(shortcut_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_604 = self.getattr_L__mod___stage4_0_branches_1___1___bn1(x_603);  x_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_605 = self.getattr_L__mod___stage4_0_branches_1___1___drop_block(x_604);  x_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_606 = self.getattr_L__mod___stage4_0_branches_1___1___act1(x_605);  x_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_607 = self.getattr_L__mod___stage4_0_branches_1___1___aa(x_606);  x_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_608 = self.getattr_L__mod___stage4_0_branches_1___1___conv2(x_607);  x_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_609 = self.getattr_L__mod___stage4_0_branches_1___1___bn2(x_608);  x_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_609 += shortcut_66;  x_610 = x_609;  x_609 = shortcut_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_67 = self.getattr_L__mod___stage4_0_branches_1___1___act2(x_610);  x_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_612 = self.getattr_L__mod___stage4_0_branches_1___2___conv1(shortcut_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_613 = self.getattr_L__mod___stage4_0_branches_1___2___bn1(x_612);  x_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_614 = self.getattr_L__mod___stage4_0_branches_1___2___drop_block(x_613);  x_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_615 = self.getattr_L__mod___stage4_0_branches_1___2___act1(x_614);  x_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_616 = self.getattr_L__mod___stage4_0_branches_1___2___aa(x_615);  x_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_617 = self.getattr_L__mod___stage4_0_branches_1___2___conv2(x_616);  x_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_618 = self.getattr_L__mod___stage4_0_branches_1___2___bn2(x_617);  x_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_618 += shortcut_67;  x_619 = x_618;  x_618 = shortcut_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_68 = self.getattr_L__mod___stage4_0_branches_1___2___act2(x_619);  x_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_621 = self.getattr_L__mod___stage4_0_branches_1___3___conv1(shortcut_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_622 = self.getattr_L__mod___stage4_0_branches_1___3___bn1(x_621);  x_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_623 = self.getattr_L__mod___stage4_0_branches_1___3___drop_block(x_622);  x_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_624 = self.getattr_L__mod___stage4_0_branches_1___3___act1(x_623);  x_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_625 = self.getattr_L__mod___stage4_0_branches_1___3___aa(x_624);  x_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_626 = self.getattr_L__mod___stage4_0_branches_1___3___conv2(x_625);  x_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_627 = self.getattr_L__mod___stage4_0_branches_1___3___bn2(x_626);  x_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_627 += shortcut_68;  x_628 = x_627;  x_627 = shortcut_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_629 = self.getattr_L__mod___stage4_0_branches_1___3___act2(x_628);  x_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_630 = self.getattr_L__mod___stage4_0_branches_2___0___conv1(shortcut_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_631 = self.getattr_L__mod___stage4_0_branches_2___0___bn1(x_630);  x_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_632 = self.getattr_L__mod___stage4_0_branches_2___0___drop_block(x_631);  x_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_633 = self.getattr_L__mod___stage4_0_branches_2___0___act1(x_632);  x_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_634 = self.getattr_L__mod___stage4_0_branches_2___0___aa(x_633);  x_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_635 = self.getattr_L__mod___stage4_0_branches_2___0___conv2(x_634);  x_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_636 = self.getattr_L__mod___stage4_0_branches_2___0___bn2(x_635);  x_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_636 += shortcut_69;  x_637 = x_636;  x_636 = shortcut_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_70 = self.getattr_L__mod___stage4_0_branches_2___0___act2(x_637);  x_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_639 = self.getattr_L__mod___stage4_0_branches_2___1___conv1(shortcut_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_640 = self.getattr_L__mod___stage4_0_branches_2___1___bn1(x_639);  x_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_641 = self.getattr_L__mod___stage4_0_branches_2___1___drop_block(x_640);  x_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_642 = self.getattr_L__mod___stage4_0_branches_2___1___act1(x_641);  x_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_643 = self.getattr_L__mod___stage4_0_branches_2___1___aa(x_642);  x_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_644 = self.getattr_L__mod___stage4_0_branches_2___1___conv2(x_643);  x_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_645 = self.getattr_L__mod___stage4_0_branches_2___1___bn2(x_644);  x_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_645 += shortcut_70;  x_646 = x_645;  x_645 = shortcut_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_71 = self.getattr_L__mod___stage4_0_branches_2___1___act2(x_646);  x_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_648 = self.getattr_L__mod___stage4_0_branches_2___2___conv1(shortcut_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_649 = self.getattr_L__mod___stage4_0_branches_2___2___bn1(x_648);  x_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_650 = self.getattr_L__mod___stage4_0_branches_2___2___drop_block(x_649);  x_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_651 = self.getattr_L__mod___stage4_0_branches_2___2___act1(x_650);  x_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_652 = self.getattr_L__mod___stage4_0_branches_2___2___aa(x_651);  x_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_653 = self.getattr_L__mod___stage4_0_branches_2___2___conv2(x_652);  x_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_654 = self.getattr_L__mod___stage4_0_branches_2___2___bn2(x_653);  x_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_654 += shortcut_71;  x_655 = x_654;  x_654 = shortcut_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_72 = self.getattr_L__mod___stage4_0_branches_2___2___act2(x_655);  x_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_657 = self.getattr_L__mod___stage4_0_branches_2___3___conv1(shortcut_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_658 = self.getattr_L__mod___stage4_0_branches_2___3___bn1(x_657);  x_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_659 = self.getattr_L__mod___stage4_0_branches_2___3___drop_block(x_658);  x_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_660 = self.getattr_L__mod___stage4_0_branches_2___3___act1(x_659);  x_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_661 = self.getattr_L__mod___stage4_0_branches_2___3___aa(x_660);  x_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_662 = self.getattr_L__mod___stage4_0_branches_2___3___conv2(x_661);  x_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_663 = self.getattr_L__mod___stage4_0_branches_2___3___bn2(x_662);  x_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_663 += shortcut_72;  x_664 = x_663;  x_663 = shortcut_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_665 = self.getattr_L__mod___stage4_0_branches_2___3___act2(x_664);  x_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_666 = self.getattr_L__mod___stage4_0_branches_3___0___conv1(shortcut_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_667 = self.getattr_L__mod___stage4_0_branches_3___0___bn1(x_666);  x_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_668 = self.getattr_L__mod___stage4_0_branches_3___0___drop_block(x_667);  x_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_669 = self.getattr_L__mod___stage4_0_branches_3___0___act1(x_668);  x_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_670 = self.getattr_L__mod___stage4_0_branches_3___0___aa(x_669);  x_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_671 = self.getattr_L__mod___stage4_0_branches_3___0___conv2(x_670);  x_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_672 = self.getattr_L__mod___stage4_0_branches_3___0___bn2(x_671);  x_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_672 += shortcut_73;  x_673 = x_672;  x_672 = shortcut_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_74 = self.getattr_L__mod___stage4_0_branches_3___0___act2(x_673);  x_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_675 = self.getattr_L__mod___stage4_0_branches_3___1___conv1(shortcut_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_676 = self.getattr_L__mod___stage4_0_branches_3___1___bn1(x_675);  x_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_677 = self.getattr_L__mod___stage4_0_branches_3___1___drop_block(x_676);  x_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_678 = self.getattr_L__mod___stage4_0_branches_3___1___act1(x_677);  x_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_679 = self.getattr_L__mod___stage4_0_branches_3___1___aa(x_678);  x_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_680 = self.getattr_L__mod___stage4_0_branches_3___1___conv2(x_679);  x_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_681 = self.getattr_L__mod___stage4_0_branches_3___1___bn2(x_680);  x_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_681 += shortcut_74;  x_682 = x_681;  x_681 = shortcut_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_75 = self.getattr_L__mod___stage4_0_branches_3___1___act2(x_682);  x_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_684 = self.getattr_L__mod___stage4_0_branches_3___2___conv1(shortcut_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_685 = self.getattr_L__mod___stage4_0_branches_3___2___bn1(x_684);  x_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_686 = self.getattr_L__mod___stage4_0_branches_3___2___drop_block(x_685);  x_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_687 = self.getattr_L__mod___stage4_0_branches_3___2___act1(x_686);  x_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_688 = self.getattr_L__mod___stage4_0_branches_3___2___aa(x_687);  x_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_689 = self.getattr_L__mod___stage4_0_branches_3___2___conv2(x_688);  x_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_690 = self.getattr_L__mod___stage4_0_branches_3___2___bn2(x_689);  x_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_690 += shortcut_75;  x_691 = x_690;  x_690 = shortcut_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_76 = self.getattr_L__mod___stage4_0_branches_3___2___act2(x_691);  x_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_693 = self.getattr_L__mod___stage4_0_branches_3___3___conv1(shortcut_76)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_694 = self.getattr_L__mod___stage4_0_branches_3___3___bn1(x_693);  x_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_695 = self.getattr_L__mod___stage4_0_branches_3___3___drop_block(x_694);  x_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_696 = self.getattr_L__mod___stage4_0_branches_3___3___act1(x_695);  x_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_697 = self.getattr_L__mod___stage4_0_branches_3___3___aa(x_696);  x_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_698 = self.getattr_L__mod___stage4_0_branches_3___3___conv2(x_697);  x_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_699 = self.getattr_L__mod___stage4_0_branches_3___3___bn2(x_698);  x_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_699 += shortcut_76;  x_700 = x_699;  x_699 = shortcut_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_701 = self.getattr_L__mod___stage4_0_branches_3___3___act2(x_700);  x_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_40 = self.L__mod___stage4_0_fuse_layers_0_0(x_593)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_0_fuse_layers_0_1_0 = self.L__mod___stage4_0_fuse_layers_0_1_0(x_629)
    l__mod___stage4_0_fuse_layers_0_1_1 = self.L__mod___stage4_0_fuse_layers_0_1_1(l__mod___stage4_0_fuse_layers_0_1_0);  l__mod___stage4_0_fuse_layers_0_1_0 = None
    l__mod___stage4_0_fuse_layers_0_1_2 = self.L__mod___stage4_0_fuse_layers_0_1_2(l__mod___stage4_0_fuse_layers_0_1_1);  l__mod___stage4_0_fuse_layers_0_1_1 = None
    y_41 = y_40 + l__mod___stage4_0_fuse_layers_0_1_2;  y_40 = l__mod___stage4_0_fuse_layers_0_1_2 = None
    l__mod___stage4_0_fuse_layers_0_2_0 = self.L__mod___stage4_0_fuse_layers_0_2_0(x_665)
    l__mod___stage4_0_fuse_layers_0_2_1 = self.L__mod___stage4_0_fuse_layers_0_2_1(l__mod___stage4_0_fuse_layers_0_2_0);  l__mod___stage4_0_fuse_layers_0_2_0 = None
    l__mod___stage4_0_fuse_layers_0_2_2 = self.L__mod___stage4_0_fuse_layers_0_2_2(l__mod___stage4_0_fuse_layers_0_2_1);  l__mod___stage4_0_fuse_layers_0_2_1 = None
    y_42 = y_41 + l__mod___stage4_0_fuse_layers_0_2_2;  y_41 = l__mod___stage4_0_fuse_layers_0_2_2 = None
    l__mod___stage4_0_fuse_layers_0_3_0 = self.L__mod___stage4_0_fuse_layers_0_3_0(x_701)
    l__mod___stage4_0_fuse_layers_0_3_1 = self.L__mod___stage4_0_fuse_layers_0_3_1(l__mod___stage4_0_fuse_layers_0_3_0);  l__mod___stage4_0_fuse_layers_0_3_0 = None
    l__mod___stage4_0_fuse_layers_0_3_2 = self.L__mod___stage4_0_fuse_layers_0_3_2(l__mod___stage4_0_fuse_layers_0_3_1);  l__mod___stage4_0_fuse_layers_0_3_1 = None
    y_43 = y_42 + l__mod___stage4_0_fuse_layers_0_3_2;  y_42 = l__mod___stage4_0_fuse_layers_0_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_77 = self.L__mod___stage4_0_fuse_act(y_43);  y_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_0_fuse_layers_1_0_0_0 = self.L__mod___stage4_0_fuse_layers_1_0_0_0(x_593)
    y_44 = self.L__mod___stage4_0_fuse_layers_1_0_0_1(l__mod___stage4_0_fuse_layers_1_0_0_0);  l__mod___stage4_0_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_0_fuse_layers_1_1 = self.L__mod___stage4_0_fuse_layers_1_1(x_629)
    y_45 = y_44 + l__mod___stage4_0_fuse_layers_1_1;  y_44 = l__mod___stage4_0_fuse_layers_1_1 = None
    l__mod___stage4_0_fuse_layers_1_2_0 = self.L__mod___stage4_0_fuse_layers_1_2_0(x_665)
    l__mod___stage4_0_fuse_layers_1_2_1 = self.L__mod___stage4_0_fuse_layers_1_2_1(l__mod___stage4_0_fuse_layers_1_2_0);  l__mod___stage4_0_fuse_layers_1_2_0 = None
    l__mod___stage4_0_fuse_layers_1_2_2 = self.L__mod___stage4_0_fuse_layers_1_2_2(l__mod___stage4_0_fuse_layers_1_2_1);  l__mod___stage4_0_fuse_layers_1_2_1 = None
    y_46 = y_45 + l__mod___stage4_0_fuse_layers_1_2_2;  y_45 = l__mod___stage4_0_fuse_layers_1_2_2 = None
    l__mod___stage4_0_fuse_layers_1_3_0 = self.L__mod___stage4_0_fuse_layers_1_3_0(x_701)
    l__mod___stage4_0_fuse_layers_1_3_1 = self.L__mod___stage4_0_fuse_layers_1_3_1(l__mod___stage4_0_fuse_layers_1_3_0);  l__mod___stage4_0_fuse_layers_1_3_0 = None
    l__mod___stage4_0_fuse_layers_1_3_2 = self.L__mod___stage4_0_fuse_layers_1_3_2(l__mod___stage4_0_fuse_layers_1_3_1);  l__mod___stage4_0_fuse_layers_1_3_1 = None
    y_47 = y_46 + l__mod___stage4_0_fuse_layers_1_3_2;  y_46 = l__mod___stage4_0_fuse_layers_1_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_81 = self.L__mod___stage4_0_fuse_act(y_47);  y_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_0_fuse_layers_2_0_0_0 = self.L__mod___stage4_0_fuse_layers_2_0_0_0(x_593)
    l__mod___stage4_0_fuse_layers_2_0_0_1 = self.L__mod___stage4_0_fuse_layers_2_0_0_1(l__mod___stage4_0_fuse_layers_2_0_0_0);  l__mod___stage4_0_fuse_layers_2_0_0_0 = None
    l__mod___stage4_0_fuse_layers_2_0_0_2 = self.L__mod___stage4_0_fuse_layers_2_0_0_2(l__mod___stage4_0_fuse_layers_2_0_0_1);  l__mod___stage4_0_fuse_layers_2_0_0_1 = None
    l__mod___stage4_0_fuse_layers_2_0_1_0 = self.L__mod___stage4_0_fuse_layers_2_0_1_0(l__mod___stage4_0_fuse_layers_2_0_0_2);  l__mod___stage4_0_fuse_layers_2_0_0_2 = None
    y_48 = self.L__mod___stage4_0_fuse_layers_2_0_1_1(l__mod___stage4_0_fuse_layers_2_0_1_0);  l__mod___stage4_0_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_0_fuse_layers_2_1_0_0 = self.L__mod___stage4_0_fuse_layers_2_1_0_0(x_629)
    l__mod___stage4_0_fuse_layers_2_1_0_1 = self.L__mod___stage4_0_fuse_layers_2_1_0_1(l__mod___stage4_0_fuse_layers_2_1_0_0);  l__mod___stage4_0_fuse_layers_2_1_0_0 = None
    y_49 = y_48 + l__mod___stage4_0_fuse_layers_2_1_0_1;  y_48 = l__mod___stage4_0_fuse_layers_2_1_0_1 = None
    l__mod___stage4_0_fuse_layers_2_2 = self.L__mod___stage4_0_fuse_layers_2_2(x_665)
    y_50 = y_49 + l__mod___stage4_0_fuse_layers_2_2;  y_49 = l__mod___stage4_0_fuse_layers_2_2 = None
    l__mod___stage4_0_fuse_layers_2_3_0 = self.L__mod___stage4_0_fuse_layers_2_3_0(x_701)
    l__mod___stage4_0_fuse_layers_2_3_1 = self.L__mod___stage4_0_fuse_layers_2_3_1(l__mod___stage4_0_fuse_layers_2_3_0);  l__mod___stage4_0_fuse_layers_2_3_0 = None
    l__mod___stage4_0_fuse_layers_2_3_2 = self.L__mod___stage4_0_fuse_layers_2_3_2(l__mod___stage4_0_fuse_layers_2_3_1);  l__mod___stage4_0_fuse_layers_2_3_1 = None
    y_51 = y_50 + l__mod___stage4_0_fuse_layers_2_3_2;  y_50 = l__mod___stage4_0_fuse_layers_2_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_85 = self.L__mod___stage4_0_fuse_act(y_51);  y_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_0_fuse_layers_3_0_0_0 = self.L__mod___stage4_0_fuse_layers_3_0_0_0(x_593);  x_593 = None
    l__mod___stage4_0_fuse_layers_3_0_0_1 = self.L__mod___stage4_0_fuse_layers_3_0_0_1(l__mod___stage4_0_fuse_layers_3_0_0_0);  l__mod___stage4_0_fuse_layers_3_0_0_0 = None
    l__mod___stage4_0_fuse_layers_3_0_0_2 = self.L__mod___stage4_0_fuse_layers_3_0_0_2(l__mod___stage4_0_fuse_layers_3_0_0_1);  l__mod___stage4_0_fuse_layers_3_0_0_1 = None
    l__mod___stage4_0_fuse_layers_3_0_1_0 = self.L__mod___stage4_0_fuse_layers_3_0_1_0(l__mod___stage4_0_fuse_layers_3_0_0_2);  l__mod___stage4_0_fuse_layers_3_0_0_2 = None
    l__mod___stage4_0_fuse_layers_3_0_1_1 = self.L__mod___stage4_0_fuse_layers_3_0_1_1(l__mod___stage4_0_fuse_layers_3_0_1_0);  l__mod___stage4_0_fuse_layers_3_0_1_0 = None
    l__mod___stage4_0_fuse_layers_3_0_1_2 = self.L__mod___stage4_0_fuse_layers_3_0_1_2(l__mod___stage4_0_fuse_layers_3_0_1_1);  l__mod___stage4_0_fuse_layers_3_0_1_1 = None
    l__mod___stage4_0_fuse_layers_3_0_2_0 = self.L__mod___stage4_0_fuse_layers_3_0_2_0(l__mod___stage4_0_fuse_layers_3_0_1_2);  l__mod___stage4_0_fuse_layers_3_0_1_2 = None
    y_52 = self.L__mod___stage4_0_fuse_layers_3_0_2_1(l__mod___stage4_0_fuse_layers_3_0_2_0);  l__mod___stage4_0_fuse_layers_3_0_2_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_0_fuse_layers_3_1_0_0 = self.L__mod___stage4_0_fuse_layers_3_1_0_0(x_629);  x_629 = None
    l__mod___stage4_0_fuse_layers_3_1_0_1 = self.L__mod___stage4_0_fuse_layers_3_1_0_1(l__mod___stage4_0_fuse_layers_3_1_0_0);  l__mod___stage4_0_fuse_layers_3_1_0_0 = None
    l__mod___stage4_0_fuse_layers_3_1_0_2 = self.L__mod___stage4_0_fuse_layers_3_1_0_2(l__mod___stage4_0_fuse_layers_3_1_0_1);  l__mod___stage4_0_fuse_layers_3_1_0_1 = None
    l__mod___stage4_0_fuse_layers_3_1_1_0 = self.L__mod___stage4_0_fuse_layers_3_1_1_0(l__mod___stage4_0_fuse_layers_3_1_0_2);  l__mod___stage4_0_fuse_layers_3_1_0_2 = None
    l__mod___stage4_0_fuse_layers_3_1_1_1 = self.L__mod___stage4_0_fuse_layers_3_1_1_1(l__mod___stage4_0_fuse_layers_3_1_1_0);  l__mod___stage4_0_fuse_layers_3_1_1_0 = None
    y_53 = y_52 + l__mod___stage4_0_fuse_layers_3_1_1_1;  y_52 = l__mod___stage4_0_fuse_layers_3_1_1_1 = None
    l__mod___stage4_0_fuse_layers_3_2_0_0 = self.L__mod___stage4_0_fuse_layers_3_2_0_0(x_665);  x_665 = None
    l__mod___stage4_0_fuse_layers_3_2_0_1 = self.L__mod___stage4_0_fuse_layers_3_2_0_1(l__mod___stage4_0_fuse_layers_3_2_0_0);  l__mod___stage4_0_fuse_layers_3_2_0_0 = None
    y_54 = y_53 + l__mod___stage4_0_fuse_layers_3_2_0_1;  y_53 = l__mod___stage4_0_fuse_layers_3_2_0_1 = None
    l__mod___stage4_0_fuse_layers_3_3 = self.L__mod___stage4_0_fuse_layers_3_3(x_701);  x_701 = None
    y_55 = y_54 + l__mod___stage4_0_fuse_layers_3_3;  y_54 = l__mod___stage4_0_fuse_layers_3_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_89 = self.L__mod___stage4_0_fuse_act(y_55);  y_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_702 = self.getattr_L__mod___stage4_1_branches_0___0___conv1(shortcut_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_703 = self.getattr_L__mod___stage4_1_branches_0___0___bn1(x_702);  x_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_704 = self.getattr_L__mod___stage4_1_branches_0___0___drop_block(x_703);  x_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_705 = self.getattr_L__mod___stage4_1_branches_0___0___act1(x_704);  x_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_706 = self.getattr_L__mod___stage4_1_branches_0___0___aa(x_705);  x_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_707 = self.getattr_L__mod___stage4_1_branches_0___0___conv2(x_706);  x_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_708 = self.getattr_L__mod___stage4_1_branches_0___0___bn2(x_707);  x_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_708 += shortcut_77;  x_709 = x_708;  x_708 = shortcut_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_78 = self.getattr_L__mod___stage4_1_branches_0___0___act2(x_709);  x_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_711 = self.getattr_L__mod___stage4_1_branches_0___1___conv1(shortcut_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_712 = self.getattr_L__mod___stage4_1_branches_0___1___bn1(x_711);  x_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_713 = self.getattr_L__mod___stage4_1_branches_0___1___drop_block(x_712);  x_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_714 = self.getattr_L__mod___stage4_1_branches_0___1___act1(x_713);  x_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_715 = self.getattr_L__mod___stage4_1_branches_0___1___aa(x_714);  x_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_716 = self.getattr_L__mod___stage4_1_branches_0___1___conv2(x_715);  x_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_717 = self.getattr_L__mod___stage4_1_branches_0___1___bn2(x_716);  x_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_717 += shortcut_78;  x_718 = x_717;  x_717 = shortcut_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_79 = self.getattr_L__mod___stage4_1_branches_0___1___act2(x_718);  x_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_720 = self.getattr_L__mod___stage4_1_branches_0___2___conv1(shortcut_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_721 = self.getattr_L__mod___stage4_1_branches_0___2___bn1(x_720);  x_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_722 = self.getattr_L__mod___stage4_1_branches_0___2___drop_block(x_721);  x_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_723 = self.getattr_L__mod___stage4_1_branches_0___2___act1(x_722);  x_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_724 = self.getattr_L__mod___stage4_1_branches_0___2___aa(x_723);  x_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_725 = self.getattr_L__mod___stage4_1_branches_0___2___conv2(x_724);  x_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_726 = self.getattr_L__mod___stage4_1_branches_0___2___bn2(x_725);  x_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_726 += shortcut_79;  x_727 = x_726;  x_726 = shortcut_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_80 = self.getattr_L__mod___stage4_1_branches_0___2___act2(x_727);  x_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_729 = self.getattr_L__mod___stage4_1_branches_0___3___conv1(shortcut_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_730 = self.getattr_L__mod___stage4_1_branches_0___3___bn1(x_729);  x_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_731 = self.getattr_L__mod___stage4_1_branches_0___3___drop_block(x_730);  x_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_732 = self.getattr_L__mod___stage4_1_branches_0___3___act1(x_731);  x_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_733 = self.getattr_L__mod___stage4_1_branches_0___3___aa(x_732);  x_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_734 = self.getattr_L__mod___stage4_1_branches_0___3___conv2(x_733);  x_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_735 = self.getattr_L__mod___stage4_1_branches_0___3___bn2(x_734);  x_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_735 += shortcut_80;  x_736 = x_735;  x_735 = shortcut_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_737 = self.getattr_L__mod___stage4_1_branches_0___3___act2(x_736);  x_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_738 = self.getattr_L__mod___stage4_1_branches_1___0___conv1(shortcut_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_739 = self.getattr_L__mod___stage4_1_branches_1___0___bn1(x_738);  x_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_740 = self.getattr_L__mod___stage4_1_branches_1___0___drop_block(x_739);  x_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_741 = self.getattr_L__mod___stage4_1_branches_1___0___act1(x_740);  x_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_742 = self.getattr_L__mod___stage4_1_branches_1___0___aa(x_741);  x_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_743 = self.getattr_L__mod___stage4_1_branches_1___0___conv2(x_742);  x_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_744 = self.getattr_L__mod___stage4_1_branches_1___0___bn2(x_743);  x_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_744 += shortcut_81;  x_745 = x_744;  x_744 = shortcut_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_82 = self.getattr_L__mod___stage4_1_branches_1___0___act2(x_745);  x_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_747 = self.getattr_L__mod___stage4_1_branches_1___1___conv1(shortcut_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_748 = self.getattr_L__mod___stage4_1_branches_1___1___bn1(x_747);  x_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_749 = self.getattr_L__mod___stage4_1_branches_1___1___drop_block(x_748);  x_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_750 = self.getattr_L__mod___stage4_1_branches_1___1___act1(x_749);  x_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_751 = self.getattr_L__mod___stage4_1_branches_1___1___aa(x_750);  x_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_752 = self.getattr_L__mod___stage4_1_branches_1___1___conv2(x_751);  x_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_753 = self.getattr_L__mod___stage4_1_branches_1___1___bn2(x_752);  x_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_753 += shortcut_82;  x_754 = x_753;  x_753 = shortcut_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_83 = self.getattr_L__mod___stage4_1_branches_1___1___act2(x_754);  x_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_756 = self.getattr_L__mod___stage4_1_branches_1___2___conv1(shortcut_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_757 = self.getattr_L__mod___stage4_1_branches_1___2___bn1(x_756);  x_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_758 = self.getattr_L__mod___stage4_1_branches_1___2___drop_block(x_757);  x_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_759 = self.getattr_L__mod___stage4_1_branches_1___2___act1(x_758);  x_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_760 = self.getattr_L__mod___stage4_1_branches_1___2___aa(x_759);  x_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_761 = self.getattr_L__mod___stage4_1_branches_1___2___conv2(x_760);  x_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_762 = self.getattr_L__mod___stage4_1_branches_1___2___bn2(x_761);  x_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_762 += shortcut_83;  x_763 = x_762;  x_762 = shortcut_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_84 = self.getattr_L__mod___stage4_1_branches_1___2___act2(x_763);  x_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_765 = self.getattr_L__mod___stage4_1_branches_1___3___conv1(shortcut_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_766 = self.getattr_L__mod___stage4_1_branches_1___3___bn1(x_765);  x_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_767 = self.getattr_L__mod___stage4_1_branches_1___3___drop_block(x_766);  x_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_768 = self.getattr_L__mod___stage4_1_branches_1___3___act1(x_767);  x_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_769 = self.getattr_L__mod___stage4_1_branches_1___3___aa(x_768);  x_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_770 = self.getattr_L__mod___stage4_1_branches_1___3___conv2(x_769);  x_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_771 = self.getattr_L__mod___stage4_1_branches_1___3___bn2(x_770);  x_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_771 += shortcut_84;  x_772 = x_771;  x_771 = shortcut_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_773 = self.getattr_L__mod___stage4_1_branches_1___3___act2(x_772);  x_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_774 = self.getattr_L__mod___stage4_1_branches_2___0___conv1(shortcut_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_775 = self.getattr_L__mod___stage4_1_branches_2___0___bn1(x_774);  x_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_776 = self.getattr_L__mod___stage4_1_branches_2___0___drop_block(x_775);  x_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_777 = self.getattr_L__mod___stage4_1_branches_2___0___act1(x_776);  x_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_778 = self.getattr_L__mod___stage4_1_branches_2___0___aa(x_777);  x_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_779 = self.getattr_L__mod___stage4_1_branches_2___0___conv2(x_778);  x_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_780 = self.getattr_L__mod___stage4_1_branches_2___0___bn2(x_779);  x_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_780 += shortcut_85;  x_781 = x_780;  x_780 = shortcut_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_86 = self.getattr_L__mod___stage4_1_branches_2___0___act2(x_781);  x_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_783 = self.getattr_L__mod___stage4_1_branches_2___1___conv1(shortcut_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_784 = self.getattr_L__mod___stage4_1_branches_2___1___bn1(x_783);  x_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_785 = self.getattr_L__mod___stage4_1_branches_2___1___drop_block(x_784);  x_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_786 = self.getattr_L__mod___stage4_1_branches_2___1___act1(x_785);  x_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_787 = self.getattr_L__mod___stage4_1_branches_2___1___aa(x_786);  x_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_788 = self.getattr_L__mod___stage4_1_branches_2___1___conv2(x_787);  x_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_789 = self.getattr_L__mod___stage4_1_branches_2___1___bn2(x_788);  x_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_789 += shortcut_86;  x_790 = x_789;  x_789 = shortcut_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_87 = self.getattr_L__mod___stage4_1_branches_2___1___act2(x_790);  x_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_792 = self.getattr_L__mod___stage4_1_branches_2___2___conv1(shortcut_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_793 = self.getattr_L__mod___stage4_1_branches_2___2___bn1(x_792);  x_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_794 = self.getattr_L__mod___stage4_1_branches_2___2___drop_block(x_793);  x_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_795 = self.getattr_L__mod___stage4_1_branches_2___2___act1(x_794);  x_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_796 = self.getattr_L__mod___stage4_1_branches_2___2___aa(x_795);  x_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_797 = self.getattr_L__mod___stage4_1_branches_2___2___conv2(x_796);  x_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_798 = self.getattr_L__mod___stage4_1_branches_2___2___bn2(x_797);  x_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_798 += shortcut_87;  x_799 = x_798;  x_798 = shortcut_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_88 = self.getattr_L__mod___stage4_1_branches_2___2___act2(x_799);  x_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_801 = self.getattr_L__mod___stage4_1_branches_2___3___conv1(shortcut_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_802 = self.getattr_L__mod___stage4_1_branches_2___3___bn1(x_801);  x_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_803 = self.getattr_L__mod___stage4_1_branches_2___3___drop_block(x_802);  x_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_804 = self.getattr_L__mod___stage4_1_branches_2___3___act1(x_803);  x_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_805 = self.getattr_L__mod___stage4_1_branches_2___3___aa(x_804);  x_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_806 = self.getattr_L__mod___stage4_1_branches_2___3___conv2(x_805);  x_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_807 = self.getattr_L__mod___stage4_1_branches_2___3___bn2(x_806);  x_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_807 += shortcut_88;  x_808 = x_807;  x_807 = shortcut_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_809 = self.getattr_L__mod___stage4_1_branches_2___3___act2(x_808);  x_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_810 = self.getattr_L__mod___stage4_1_branches_3___0___conv1(shortcut_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_811 = self.getattr_L__mod___stage4_1_branches_3___0___bn1(x_810);  x_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_812 = self.getattr_L__mod___stage4_1_branches_3___0___drop_block(x_811);  x_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_813 = self.getattr_L__mod___stage4_1_branches_3___0___act1(x_812);  x_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_814 = self.getattr_L__mod___stage4_1_branches_3___0___aa(x_813);  x_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_815 = self.getattr_L__mod___stage4_1_branches_3___0___conv2(x_814);  x_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_816 = self.getattr_L__mod___stage4_1_branches_3___0___bn2(x_815);  x_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_816 += shortcut_89;  x_817 = x_816;  x_816 = shortcut_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_90 = self.getattr_L__mod___stage4_1_branches_3___0___act2(x_817);  x_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_819 = self.getattr_L__mod___stage4_1_branches_3___1___conv1(shortcut_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_820 = self.getattr_L__mod___stage4_1_branches_3___1___bn1(x_819);  x_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_821 = self.getattr_L__mod___stage4_1_branches_3___1___drop_block(x_820);  x_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_822 = self.getattr_L__mod___stage4_1_branches_3___1___act1(x_821);  x_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_823 = self.getattr_L__mod___stage4_1_branches_3___1___aa(x_822);  x_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_824 = self.getattr_L__mod___stage4_1_branches_3___1___conv2(x_823);  x_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_825 = self.getattr_L__mod___stage4_1_branches_3___1___bn2(x_824);  x_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_825 += shortcut_90;  x_826 = x_825;  x_825 = shortcut_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_91 = self.getattr_L__mod___stage4_1_branches_3___1___act2(x_826);  x_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_828 = self.getattr_L__mod___stage4_1_branches_3___2___conv1(shortcut_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_829 = self.getattr_L__mod___stage4_1_branches_3___2___bn1(x_828);  x_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_830 = self.getattr_L__mod___stage4_1_branches_3___2___drop_block(x_829);  x_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_831 = self.getattr_L__mod___stage4_1_branches_3___2___act1(x_830);  x_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_832 = self.getattr_L__mod___stage4_1_branches_3___2___aa(x_831);  x_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_833 = self.getattr_L__mod___stage4_1_branches_3___2___conv2(x_832);  x_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_834 = self.getattr_L__mod___stage4_1_branches_3___2___bn2(x_833);  x_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_834 += shortcut_91;  x_835 = x_834;  x_834 = shortcut_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_92 = self.getattr_L__mod___stage4_1_branches_3___2___act2(x_835);  x_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_837 = self.getattr_L__mod___stage4_1_branches_3___3___conv1(shortcut_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_838 = self.getattr_L__mod___stage4_1_branches_3___3___bn1(x_837);  x_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_839 = self.getattr_L__mod___stage4_1_branches_3___3___drop_block(x_838);  x_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_840 = self.getattr_L__mod___stage4_1_branches_3___3___act1(x_839);  x_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_841 = self.getattr_L__mod___stage4_1_branches_3___3___aa(x_840);  x_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_842 = self.getattr_L__mod___stage4_1_branches_3___3___conv2(x_841);  x_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_843 = self.getattr_L__mod___stage4_1_branches_3___3___bn2(x_842);  x_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_843 += shortcut_92;  x_844 = x_843;  x_843 = shortcut_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_845 = self.getattr_L__mod___stage4_1_branches_3___3___act2(x_844);  x_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_56 = self.L__mod___stage4_1_fuse_layers_0_0(x_737)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_1_fuse_layers_0_1_0 = self.L__mod___stage4_1_fuse_layers_0_1_0(x_773)
    l__mod___stage4_1_fuse_layers_0_1_1 = self.L__mod___stage4_1_fuse_layers_0_1_1(l__mod___stage4_1_fuse_layers_0_1_0);  l__mod___stage4_1_fuse_layers_0_1_0 = None
    l__mod___stage4_1_fuse_layers_0_1_2 = self.L__mod___stage4_1_fuse_layers_0_1_2(l__mod___stage4_1_fuse_layers_0_1_1);  l__mod___stage4_1_fuse_layers_0_1_1 = None
    y_57 = y_56 + l__mod___stage4_1_fuse_layers_0_1_2;  y_56 = l__mod___stage4_1_fuse_layers_0_1_2 = None
    l__mod___stage4_1_fuse_layers_0_2_0 = self.L__mod___stage4_1_fuse_layers_0_2_0(x_809)
    l__mod___stage4_1_fuse_layers_0_2_1 = self.L__mod___stage4_1_fuse_layers_0_2_1(l__mod___stage4_1_fuse_layers_0_2_0);  l__mod___stage4_1_fuse_layers_0_2_0 = None
    l__mod___stage4_1_fuse_layers_0_2_2 = self.L__mod___stage4_1_fuse_layers_0_2_2(l__mod___stage4_1_fuse_layers_0_2_1);  l__mod___stage4_1_fuse_layers_0_2_1 = None
    y_58 = y_57 + l__mod___stage4_1_fuse_layers_0_2_2;  y_57 = l__mod___stage4_1_fuse_layers_0_2_2 = None
    l__mod___stage4_1_fuse_layers_0_3_0 = self.L__mod___stage4_1_fuse_layers_0_3_0(x_845)
    l__mod___stage4_1_fuse_layers_0_3_1 = self.L__mod___stage4_1_fuse_layers_0_3_1(l__mod___stage4_1_fuse_layers_0_3_0);  l__mod___stage4_1_fuse_layers_0_3_0 = None
    l__mod___stage4_1_fuse_layers_0_3_2 = self.L__mod___stage4_1_fuse_layers_0_3_2(l__mod___stage4_1_fuse_layers_0_3_1);  l__mod___stage4_1_fuse_layers_0_3_1 = None
    y_59 = y_58 + l__mod___stage4_1_fuse_layers_0_3_2;  y_58 = l__mod___stage4_1_fuse_layers_0_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_93 = self.L__mod___stage4_1_fuse_act(y_59);  y_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_1_fuse_layers_1_0_0_0 = self.L__mod___stage4_1_fuse_layers_1_0_0_0(x_737)
    y_60 = self.L__mod___stage4_1_fuse_layers_1_0_0_1(l__mod___stage4_1_fuse_layers_1_0_0_0);  l__mod___stage4_1_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_1_fuse_layers_1_1 = self.L__mod___stage4_1_fuse_layers_1_1(x_773)
    y_61 = y_60 + l__mod___stage4_1_fuse_layers_1_1;  y_60 = l__mod___stage4_1_fuse_layers_1_1 = None
    l__mod___stage4_1_fuse_layers_1_2_0 = self.L__mod___stage4_1_fuse_layers_1_2_0(x_809)
    l__mod___stage4_1_fuse_layers_1_2_1 = self.L__mod___stage4_1_fuse_layers_1_2_1(l__mod___stage4_1_fuse_layers_1_2_0);  l__mod___stage4_1_fuse_layers_1_2_0 = None
    l__mod___stage4_1_fuse_layers_1_2_2 = self.L__mod___stage4_1_fuse_layers_1_2_2(l__mod___stage4_1_fuse_layers_1_2_1);  l__mod___stage4_1_fuse_layers_1_2_1 = None
    y_62 = y_61 + l__mod___stage4_1_fuse_layers_1_2_2;  y_61 = l__mod___stage4_1_fuse_layers_1_2_2 = None
    l__mod___stage4_1_fuse_layers_1_3_0 = self.L__mod___stage4_1_fuse_layers_1_3_0(x_845)
    l__mod___stage4_1_fuse_layers_1_3_1 = self.L__mod___stage4_1_fuse_layers_1_3_1(l__mod___stage4_1_fuse_layers_1_3_0);  l__mod___stage4_1_fuse_layers_1_3_0 = None
    l__mod___stage4_1_fuse_layers_1_3_2 = self.L__mod___stage4_1_fuse_layers_1_3_2(l__mod___stage4_1_fuse_layers_1_3_1);  l__mod___stage4_1_fuse_layers_1_3_1 = None
    y_63 = y_62 + l__mod___stage4_1_fuse_layers_1_3_2;  y_62 = l__mod___stage4_1_fuse_layers_1_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_97 = self.L__mod___stage4_1_fuse_act(y_63);  y_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_1_fuse_layers_2_0_0_0 = self.L__mod___stage4_1_fuse_layers_2_0_0_0(x_737)
    l__mod___stage4_1_fuse_layers_2_0_0_1 = self.L__mod___stage4_1_fuse_layers_2_0_0_1(l__mod___stage4_1_fuse_layers_2_0_0_0);  l__mod___stage4_1_fuse_layers_2_0_0_0 = None
    l__mod___stage4_1_fuse_layers_2_0_0_2 = self.L__mod___stage4_1_fuse_layers_2_0_0_2(l__mod___stage4_1_fuse_layers_2_0_0_1);  l__mod___stage4_1_fuse_layers_2_0_0_1 = None
    l__mod___stage4_1_fuse_layers_2_0_1_0 = self.L__mod___stage4_1_fuse_layers_2_0_1_0(l__mod___stage4_1_fuse_layers_2_0_0_2);  l__mod___stage4_1_fuse_layers_2_0_0_2 = None
    y_64 = self.L__mod___stage4_1_fuse_layers_2_0_1_1(l__mod___stage4_1_fuse_layers_2_0_1_0);  l__mod___stage4_1_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_1_fuse_layers_2_1_0_0 = self.L__mod___stage4_1_fuse_layers_2_1_0_0(x_773)
    l__mod___stage4_1_fuse_layers_2_1_0_1 = self.L__mod___stage4_1_fuse_layers_2_1_0_1(l__mod___stage4_1_fuse_layers_2_1_0_0);  l__mod___stage4_1_fuse_layers_2_1_0_0 = None
    y_65 = y_64 + l__mod___stage4_1_fuse_layers_2_1_0_1;  y_64 = l__mod___stage4_1_fuse_layers_2_1_0_1 = None
    l__mod___stage4_1_fuse_layers_2_2 = self.L__mod___stage4_1_fuse_layers_2_2(x_809)
    y_66 = y_65 + l__mod___stage4_1_fuse_layers_2_2;  y_65 = l__mod___stage4_1_fuse_layers_2_2 = None
    l__mod___stage4_1_fuse_layers_2_3_0 = self.L__mod___stage4_1_fuse_layers_2_3_0(x_845)
    l__mod___stage4_1_fuse_layers_2_3_1 = self.L__mod___stage4_1_fuse_layers_2_3_1(l__mod___stage4_1_fuse_layers_2_3_0);  l__mod___stage4_1_fuse_layers_2_3_0 = None
    l__mod___stage4_1_fuse_layers_2_3_2 = self.L__mod___stage4_1_fuse_layers_2_3_2(l__mod___stage4_1_fuse_layers_2_3_1);  l__mod___stage4_1_fuse_layers_2_3_1 = None
    y_67 = y_66 + l__mod___stage4_1_fuse_layers_2_3_2;  y_66 = l__mod___stage4_1_fuse_layers_2_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_101 = self.L__mod___stage4_1_fuse_act(y_67);  y_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_1_fuse_layers_3_0_0_0 = self.L__mod___stage4_1_fuse_layers_3_0_0_0(x_737);  x_737 = None
    l__mod___stage4_1_fuse_layers_3_0_0_1 = self.L__mod___stage4_1_fuse_layers_3_0_0_1(l__mod___stage4_1_fuse_layers_3_0_0_0);  l__mod___stage4_1_fuse_layers_3_0_0_0 = None
    l__mod___stage4_1_fuse_layers_3_0_0_2 = self.L__mod___stage4_1_fuse_layers_3_0_0_2(l__mod___stage4_1_fuse_layers_3_0_0_1);  l__mod___stage4_1_fuse_layers_3_0_0_1 = None
    l__mod___stage4_1_fuse_layers_3_0_1_0 = self.L__mod___stage4_1_fuse_layers_3_0_1_0(l__mod___stage4_1_fuse_layers_3_0_0_2);  l__mod___stage4_1_fuse_layers_3_0_0_2 = None
    l__mod___stage4_1_fuse_layers_3_0_1_1 = self.L__mod___stage4_1_fuse_layers_3_0_1_1(l__mod___stage4_1_fuse_layers_3_0_1_0);  l__mod___stage4_1_fuse_layers_3_0_1_0 = None
    l__mod___stage4_1_fuse_layers_3_0_1_2 = self.L__mod___stage4_1_fuse_layers_3_0_1_2(l__mod___stage4_1_fuse_layers_3_0_1_1);  l__mod___stage4_1_fuse_layers_3_0_1_1 = None
    l__mod___stage4_1_fuse_layers_3_0_2_0 = self.L__mod___stage4_1_fuse_layers_3_0_2_0(l__mod___stage4_1_fuse_layers_3_0_1_2);  l__mod___stage4_1_fuse_layers_3_0_1_2 = None
    y_68 = self.L__mod___stage4_1_fuse_layers_3_0_2_1(l__mod___stage4_1_fuse_layers_3_0_2_0);  l__mod___stage4_1_fuse_layers_3_0_2_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_1_fuse_layers_3_1_0_0 = self.L__mod___stage4_1_fuse_layers_3_1_0_0(x_773);  x_773 = None
    l__mod___stage4_1_fuse_layers_3_1_0_1 = self.L__mod___stage4_1_fuse_layers_3_1_0_1(l__mod___stage4_1_fuse_layers_3_1_0_0);  l__mod___stage4_1_fuse_layers_3_1_0_0 = None
    l__mod___stage4_1_fuse_layers_3_1_0_2 = self.L__mod___stage4_1_fuse_layers_3_1_0_2(l__mod___stage4_1_fuse_layers_3_1_0_1);  l__mod___stage4_1_fuse_layers_3_1_0_1 = None
    l__mod___stage4_1_fuse_layers_3_1_1_0 = self.L__mod___stage4_1_fuse_layers_3_1_1_0(l__mod___stage4_1_fuse_layers_3_1_0_2);  l__mod___stage4_1_fuse_layers_3_1_0_2 = None
    l__mod___stage4_1_fuse_layers_3_1_1_1 = self.L__mod___stage4_1_fuse_layers_3_1_1_1(l__mod___stage4_1_fuse_layers_3_1_1_0);  l__mod___stage4_1_fuse_layers_3_1_1_0 = None
    y_69 = y_68 + l__mod___stage4_1_fuse_layers_3_1_1_1;  y_68 = l__mod___stage4_1_fuse_layers_3_1_1_1 = None
    l__mod___stage4_1_fuse_layers_3_2_0_0 = self.L__mod___stage4_1_fuse_layers_3_2_0_0(x_809);  x_809 = None
    l__mod___stage4_1_fuse_layers_3_2_0_1 = self.L__mod___stage4_1_fuse_layers_3_2_0_1(l__mod___stage4_1_fuse_layers_3_2_0_0);  l__mod___stage4_1_fuse_layers_3_2_0_0 = None
    y_70 = y_69 + l__mod___stage4_1_fuse_layers_3_2_0_1;  y_69 = l__mod___stage4_1_fuse_layers_3_2_0_1 = None
    l__mod___stage4_1_fuse_layers_3_3 = self.L__mod___stage4_1_fuse_layers_3_3(x_845);  x_845 = None
    y_71 = y_70 + l__mod___stage4_1_fuse_layers_3_3;  y_70 = l__mod___stage4_1_fuse_layers_3_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_105 = self.L__mod___stage4_1_fuse_act(y_71);  y_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_846 = self.getattr_L__mod___stage4_2_branches_0___0___conv1(shortcut_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_847 = self.getattr_L__mod___stage4_2_branches_0___0___bn1(x_846);  x_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_848 = self.getattr_L__mod___stage4_2_branches_0___0___drop_block(x_847);  x_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_849 = self.getattr_L__mod___stage4_2_branches_0___0___act1(x_848);  x_848 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_850 = self.getattr_L__mod___stage4_2_branches_0___0___aa(x_849);  x_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_851 = self.getattr_L__mod___stage4_2_branches_0___0___conv2(x_850);  x_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_852 = self.getattr_L__mod___stage4_2_branches_0___0___bn2(x_851);  x_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_852 += shortcut_93;  x_853 = x_852;  x_852 = shortcut_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_94 = self.getattr_L__mod___stage4_2_branches_0___0___act2(x_853);  x_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_855 = self.getattr_L__mod___stage4_2_branches_0___1___conv1(shortcut_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_856 = self.getattr_L__mod___stage4_2_branches_0___1___bn1(x_855);  x_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_857 = self.getattr_L__mod___stage4_2_branches_0___1___drop_block(x_856);  x_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_858 = self.getattr_L__mod___stage4_2_branches_0___1___act1(x_857);  x_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_859 = self.getattr_L__mod___stage4_2_branches_0___1___aa(x_858);  x_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_860 = self.getattr_L__mod___stage4_2_branches_0___1___conv2(x_859);  x_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_861 = self.getattr_L__mod___stage4_2_branches_0___1___bn2(x_860);  x_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_861 += shortcut_94;  x_862 = x_861;  x_861 = shortcut_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_95 = self.getattr_L__mod___stage4_2_branches_0___1___act2(x_862);  x_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_864 = self.getattr_L__mod___stage4_2_branches_0___2___conv1(shortcut_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_865 = self.getattr_L__mod___stage4_2_branches_0___2___bn1(x_864);  x_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_866 = self.getattr_L__mod___stage4_2_branches_0___2___drop_block(x_865);  x_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_867 = self.getattr_L__mod___stage4_2_branches_0___2___act1(x_866);  x_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_868 = self.getattr_L__mod___stage4_2_branches_0___2___aa(x_867);  x_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_869 = self.getattr_L__mod___stage4_2_branches_0___2___conv2(x_868);  x_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_870 = self.getattr_L__mod___stage4_2_branches_0___2___bn2(x_869);  x_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_870 += shortcut_95;  x_871 = x_870;  x_870 = shortcut_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_96 = self.getattr_L__mod___stage4_2_branches_0___2___act2(x_871);  x_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_873 = self.getattr_L__mod___stage4_2_branches_0___3___conv1(shortcut_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_874 = self.getattr_L__mod___stage4_2_branches_0___3___bn1(x_873);  x_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_875 = self.getattr_L__mod___stage4_2_branches_0___3___drop_block(x_874);  x_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_876 = self.getattr_L__mod___stage4_2_branches_0___3___act1(x_875);  x_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_877 = self.getattr_L__mod___stage4_2_branches_0___3___aa(x_876);  x_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_878 = self.getattr_L__mod___stage4_2_branches_0___3___conv2(x_877);  x_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_879 = self.getattr_L__mod___stage4_2_branches_0___3___bn2(x_878);  x_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_879 += shortcut_96;  x_880 = x_879;  x_879 = shortcut_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_881 = self.getattr_L__mod___stage4_2_branches_0___3___act2(x_880);  x_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_882 = self.getattr_L__mod___stage4_2_branches_1___0___conv1(shortcut_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_883 = self.getattr_L__mod___stage4_2_branches_1___0___bn1(x_882);  x_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_884 = self.getattr_L__mod___stage4_2_branches_1___0___drop_block(x_883);  x_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_885 = self.getattr_L__mod___stage4_2_branches_1___0___act1(x_884);  x_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_886 = self.getattr_L__mod___stage4_2_branches_1___0___aa(x_885);  x_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_887 = self.getattr_L__mod___stage4_2_branches_1___0___conv2(x_886);  x_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_888 = self.getattr_L__mod___stage4_2_branches_1___0___bn2(x_887);  x_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_888 += shortcut_97;  x_889 = x_888;  x_888 = shortcut_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_98 = self.getattr_L__mod___stage4_2_branches_1___0___act2(x_889);  x_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_891 = self.getattr_L__mod___stage4_2_branches_1___1___conv1(shortcut_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_892 = self.getattr_L__mod___stage4_2_branches_1___1___bn1(x_891);  x_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_893 = self.getattr_L__mod___stage4_2_branches_1___1___drop_block(x_892);  x_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_894 = self.getattr_L__mod___stage4_2_branches_1___1___act1(x_893);  x_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_895 = self.getattr_L__mod___stage4_2_branches_1___1___aa(x_894);  x_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_896 = self.getattr_L__mod___stage4_2_branches_1___1___conv2(x_895);  x_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_897 = self.getattr_L__mod___stage4_2_branches_1___1___bn2(x_896);  x_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_897 += shortcut_98;  x_898 = x_897;  x_897 = shortcut_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_99 = self.getattr_L__mod___stage4_2_branches_1___1___act2(x_898);  x_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_900 = self.getattr_L__mod___stage4_2_branches_1___2___conv1(shortcut_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_901 = self.getattr_L__mod___stage4_2_branches_1___2___bn1(x_900);  x_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_902 = self.getattr_L__mod___stage4_2_branches_1___2___drop_block(x_901);  x_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_903 = self.getattr_L__mod___stage4_2_branches_1___2___act1(x_902);  x_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_904 = self.getattr_L__mod___stage4_2_branches_1___2___aa(x_903);  x_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_905 = self.getattr_L__mod___stage4_2_branches_1___2___conv2(x_904);  x_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_906 = self.getattr_L__mod___stage4_2_branches_1___2___bn2(x_905);  x_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_906 += shortcut_99;  x_907 = x_906;  x_906 = shortcut_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_100 = self.getattr_L__mod___stage4_2_branches_1___2___act2(x_907);  x_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_909 = self.getattr_L__mod___stage4_2_branches_1___3___conv1(shortcut_100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_910 = self.getattr_L__mod___stage4_2_branches_1___3___bn1(x_909);  x_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_911 = self.getattr_L__mod___stage4_2_branches_1___3___drop_block(x_910);  x_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_912 = self.getattr_L__mod___stage4_2_branches_1___3___act1(x_911);  x_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_913 = self.getattr_L__mod___stage4_2_branches_1___3___aa(x_912);  x_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_914 = self.getattr_L__mod___stage4_2_branches_1___3___conv2(x_913);  x_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_915 = self.getattr_L__mod___stage4_2_branches_1___3___bn2(x_914);  x_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_915 += shortcut_100;  x_916 = x_915;  x_915 = shortcut_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_917 = self.getattr_L__mod___stage4_2_branches_1___3___act2(x_916);  x_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_918 = self.getattr_L__mod___stage4_2_branches_2___0___conv1(shortcut_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_919 = self.getattr_L__mod___stage4_2_branches_2___0___bn1(x_918);  x_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_920 = self.getattr_L__mod___stage4_2_branches_2___0___drop_block(x_919);  x_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_921 = self.getattr_L__mod___stage4_2_branches_2___0___act1(x_920);  x_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_922 = self.getattr_L__mod___stage4_2_branches_2___0___aa(x_921);  x_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_923 = self.getattr_L__mod___stage4_2_branches_2___0___conv2(x_922);  x_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_924 = self.getattr_L__mod___stage4_2_branches_2___0___bn2(x_923);  x_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_924 += shortcut_101;  x_925 = x_924;  x_924 = shortcut_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_102 = self.getattr_L__mod___stage4_2_branches_2___0___act2(x_925);  x_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_927 = self.getattr_L__mod___stage4_2_branches_2___1___conv1(shortcut_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_928 = self.getattr_L__mod___stage4_2_branches_2___1___bn1(x_927);  x_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_929 = self.getattr_L__mod___stage4_2_branches_2___1___drop_block(x_928);  x_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_930 = self.getattr_L__mod___stage4_2_branches_2___1___act1(x_929);  x_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_931 = self.getattr_L__mod___stage4_2_branches_2___1___aa(x_930);  x_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_932 = self.getattr_L__mod___stage4_2_branches_2___1___conv2(x_931);  x_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_933 = self.getattr_L__mod___stage4_2_branches_2___1___bn2(x_932);  x_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_933 += shortcut_102;  x_934 = x_933;  x_933 = shortcut_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_103 = self.getattr_L__mod___stage4_2_branches_2___1___act2(x_934);  x_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_936 = self.getattr_L__mod___stage4_2_branches_2___2___conv1(shortcut_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_937 = self.getattr_L__mod___stage4_2_branches_2___2___bn1(x_936);  x_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_938 = self.getattr_L__mod___stage4_2_branches_2___2___drop_block(x_937);  x_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_939 = self.getattr_L__mod___stage4_2_branches_2___2___act1(x_938);  x_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_940 = self.getattr_L__mod___stage4_2_branches_2___2___aa(x_939);  x_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_941 = self.getattr_L__mod___stage4_2_branches_2___2___conv2(x_940);  x_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_942 = self.getattr_L__mod___stage4_2_branches_2___2___bn2(x_941);  x_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_942 += shortcut_103;  x_943 = x_942;  x_942 = shortcut_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_104 = self.getattr_L__mod___stage4_2_branches_2___2___act2(x_943);  x_943 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_945 = self.getattr_L__mod___stage4_2_branches_2___3___conv1(shortcut_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_946 = self.getattr_L__mod___stage4_2_branches_2___3___bn1(x_945);  x_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_947 = self.getattr_L__mod___stage4_2_branches_2___3___drop_block(x_946);  x_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_948 = self.getattr_L__mod___stage4_2_branches_2___3___act1(x_947);  x_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_949 = self.getattr_L__mod___stage4_2_branches_2___3___aa(x_948);  x_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_950 = self.getattr_L__mod___stage4_2_branches_2___3___conv2(x_949);  x_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_951 = self.getattr_L__mod___stage4_2_branches_2___3___bn2(x_950);  x_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_951 += shortcut_104;  x_952 = x_951;  x_951 = shortcut_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_953 = self.getattr_L__mod___stage4_2_branches_2___3___act2(x_952);  x_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_954 = self.getattr_L__mod___stage4_2_branches_3___0___conv1(shortcut_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_955 = self.getattr_L__mod___stage4_2_branches_3___0___bn1(x_954);  x_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_956 = self.getattr_L__mod___stage4_2_branches_3___0___drop_block(x_955);  x_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_957 = self.getattr_L__mod___stage4_2_branches_3___0___act1(x_956);  x_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_958 = self.getattr_L__mod___stage4_2_branches_3___0___aa(x_957);  x_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_959 = self.getattr_L__mod___stage4_2_branches_3___0___conv2(x_958);  x_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_960 = self.getattr_L__mod___stage4_2_branches_3___0___bn2(x_959);  x_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_960 += shortcut_105;  x_961 = x_960;  x_960 = shortcut_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_106 = self.getattr_L__mod___stage4_2_branches_3___0___act2(x_961);  x_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_963 = self.getattr_L__mod___stage4_2_branches_3___1___conv1(shortcut_106)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_964 = self.getattr_L__mod___stage4_2_branches_3___1___bn1(x_963);  x_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_965 = self.getattr_L__mod___stage4_2_branches_3___1___drop_block(x_964);  x_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_966 = self.getattr_L__mod___stage4_2_branches_3___1___act1(x_965);  x_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_967 = self.getattr_L__mod___stage4_2_branches_3___1___aa(x_966);  x_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_968 = self.getattr_L__mod___stage4_2_branches_3___1___conv2(x_967);  x_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_969 = self.getattr_L__mod___stage4_2_branches_3___1___bn2(x_968);  x_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_969 += shortcut_106;  x_970 = x_969;  x_969 = shortcut_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_107 = self.getattr_L__mod___stage4_2_branches_3___1___act2(x_970);  x_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_972 = self.getattr_L__mod___stage4_2_branches_3___2___conv1(shortcut_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_973 = self.getattr_L__mod___stage4_2_branches_3___2___bn1(x_972);  x_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_974 = self.getattr_L__mod___stage4_2_branches_3___2___drop_block(x_973);  x_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_975 = self.getattr_L__mod___stage4_2_branches_3___2___act1(x_974);  x_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_976 = self.getattr_L__mod___stage4_2_branches_3___2___aa(x_975);  x_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_977 = self.getattr_L__mod___stage4_2_branches_3___2___conv2(x_976);  x_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_978 = self.getattr_L__mod___stage4_2_branches_3___2___bn2(x_977);  x_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_978 += shortcut_107;  x_979 = x_978;  x_978 = shortcut_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    shortcut_108 = self.getattr_L__mod___stage4_2_branches_3___2___act2(x_979);  x_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:98, code: x = self.conv1(x)
    x_981 = self.getattr_L__mod___stage4_2_branches_3___3___conv1(shortcut_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:99, code: x = self.bn1(x)
    x_982 = self.getattr_L__mod___stage4_2_branches_3___3___bn1(x_981);  x_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:100, code: x = self.drop_block(x)
    x_983 = self.getattr_L__mod___stage4_2_branches_3___3___drop_block(x_982);  x_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:101, code: x = self.act1(x)
    x_984 = self.getattr_L__mod___stage4_2_branches_3___3___act1(x_983);  x_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:102, code: x = self.aa(x)
    x_985 = self.getattr_L__mod___stage4_2_branches_3___3___aa(x_984);  x_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:104, code: x = self.conv2(x)
    x_986 = self.getattr_L__mod___stage4_2_branches_3___3___conv2(x_985);  x_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:105, code: x = self.bn2(x)
    x_987 = self.getattr_L__mod___stage4_2_branches_3___3___bn2(x_986);  x_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:115, code: x += shortcut
    x_987 += shortcut_108;  x_988 = x_987;  x_987 = shortcut_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:116, code: x = self.act2(x)
    x_989 = self.getattr_L__mod___stage4_2_branches_3___3___act2(x_988);  x_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    y_72 = self.L__mod___stage4_2_fuse_layers_0_0(x_881)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_2_fuse_layers_0_1_0 = self.L__mod___stage4_2_fuse_layers_0_1_0(x_917)
    l__mod___stage4_2_fuse_layers_0_1_1 = self.L__mod___stage4_2_fuse_layers_0_1_1(l__mod___stage4_2_fuse_layers_0_1_0);  l__mod___stage4_2_fuse_layers_0_1_0 = None
    l__mod___stage4_2_fuse_layers_0_1_2 = self.L__mod___stage4_2_fuse_layers_0_1_2(l__mod___stage4_2_fuse_layers_0_1_1);  l__mod___stage4_2_fuse_layers_0_1_1 = None
    y_73 = y_72 + l__mod___stage4_2_fuse_layers_0_1_2;  y_72 = l__mod___stage4_2_fuse_layers_0_1_2 = None
    l__mod___stage4_2_fuse_layers_0_2_0 = self.L__mod___stage4_2_fuse_layers_0_2_0(x_953)
    l__mod___stage4_2_fuse_layers_0_2_1 = self.L__mod___stage4_2_fuse_layers_0_2_1(l__mod___stage4_2_fuse_layers_0_2_0);  l__mod___stage4_2_fuse_layers_0_2_0 = None
    l__mod___stage4_2_fuse_layers_0_2_2 = self.L__mod___stage4_2_fuse_layers_0_2_2(l__mod___stage4_2_fuse_layers_0_2_1);  l__mod___stage4_2_fuse_layers_0_2_1 = None
    y_74 = y_73 + l__mod___stage4_2_fuse_layers_0_2_2;  y_73 = l__mod___stage4_2_fuse_layers_0_2_2 = None
    l__mod___stage4_2_fuse_layers_0_3_0 = self.L__mod___stage4_2_fuse_layers_0_3_0(x_989)
    l__mod___stage4_2_fuse_layers_0_3_1 = self.L__mod___stage4_2_fuse_layers_0_3_1(l__mod___stage4_2_fuse_layers_0_3_0);  l__mod___stage4_2_fuse_layers_0_3_0 = None
    l__mod___stage4_2_fuse_layers_0_3_2 = self.L__mod___stage4_2_fuse_layers_0_3_2(l__mod___stage4_2_fuse_layers_0_3_1);  l__mod___stage4_2_fuse_layers_0_3_1 = None
    y_75 = y_74 + l__mod___stage4_2_fuse_layers_0_3_2;  y_74 = l__mod___stage4_2_fuse_layers_0_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_109 = self.L__mod___stage4_2_fuse_act(y_75);  y_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_2_fuse_layers_1_0_0_0 = self.L__mod___stage4_2_fuse_layers_1_0_0_0(x_881)
    y_76 = self.L__mod___stage4_2_fuse_layers_1_0_0_1(l__mod___stage4_2_fuse_layers_1_0_0_0);  l__mod___stage4_2_fuse_layers_1_0_0_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_2_fuse_layers_1_1 = self.L__mod___stage4_2_fuse_layers_1_1(x_917)
    y_77 = y_76 + l__mod___stage4_2_fuse_layers_1_1;  y_76 = l__mod___stage4_2_fuse_layers_1_1 = None
    l__mod___stage4_2_fuse_layers_1_2_0 = self.L__mod___stage4_2_fuse_layers_1_2_0(x_953)
    l__mod___stage4_2_fuse_layers_1_2_1 = self.L__mod___stage4_2_fuse_layers_1_2_1(l__mod___stage4_2_fuse_layers_1_2_0);  l__mod___stage4_2_fuse_layers_1_2_0 = None
    l__mod___stage4_2_fuse_layers_1_2_2 = self.L__mod___stage4_2_fuse_layers_1_2_2(l__mod___stage4_2_fuse_layers_1_2_1);  l__mod___stage4_2_fuse_layers_1_2_1 = None
    y_78 = y_77 + l__mod___stage4_2_fuse_layers_1_2_2;  y_77 = l__mod___stage4_2_fuse_layers_1_2_2 = None
    l__mod___stage4_2_fuse_layers_1_3_0 = self.L__mod___stage4_2_fuse_layers_1_3_0(x_989)
    l__mod___stage4_2_fuse_layers_1_3_1 = self.L__mod___stage4_2_fuse_layers_1_3_1(l__mod___stage4_2_fuse_layers_1_3_0);  l__mod___stage4_2_fuse_layers_1_3_0 = None
    l__mod___stage4_2_fuse_layers_1_3_2 = self.L__mod___stage4_2_fuse_layers_1_3_2(l__mod___stage4_2_fuse_layers_1_3_1);  l__mod___stage4_2_fuse_layers_1_3_1 = None
    y_79 = y_78 + l__mod___stage4_2_fuse_layers_1_3_2;  y_78 = l__mod___stage4_2_fuse_layers_1_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_111 = self.L__mod___stage4_2_fuse_act(y_79);  y_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_2_fuse_layers_2_0_0_0 = self.L__mod___stage4_2_fuse_layers_2_0_0_0(x_881)
    l__mod___stage4_2_fuse_layers_2_0_0_1 = self.L__mod___stage4_2_fuse_layers_2_0_0_1(l__mod___stage4_2_fuse_layers_2_0_0_0);  l__mod___stage4_2_fuse_layers_2_0_0_0 = None
    l__mod___stage4_2_fuse_layers_2_0_0_2 = self.L__mod___stage4_2_fuse_layers_2_0_0_2(l__mod___stage4_2_fuse_layers_2_0_0_1);  l__mod___stage4_2_fuse_layers_2_0_0_1 = None
    l__mod___stage4_2_fuse_layers_2_0_1_0 = self.L__mod___stage4_2_fuse_layers_2_0_1_0(l__mod___stage4_2_fuse_layers_2_0_0_2);  l__mod___stage4_2_fuse_layers_2_0_0_2 = None
    y_80 = self.L__mod___stage4_2_fuse_layers_2_0_1_1(l__mod___stage4_2_fuse_layers_2_0_1_0);  l__mod___stage4_2_fuse_layers_2_0_1_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_2_fuse_layers_2_1_0_0 = self.L__mod___stage4_2_fuse_layers_2_1_0_0(x_917)
    l__mod___stage4_2_fuse_layers_2_1_0_1 = self.L__mod___stage4_2_fuse_layers_2_1_0_1(l__mod___stage4_2_fuse_layers_2_1_0_0);  l__mod___stage4_2_fuse_layers_2_1_0_0 = None
    y_81 = y_80 + l__mod___stage4_2_fuse_layers_2_1_0_1;  y_80 = l__mod___stage4_2_fuse_layers_2_1_0_1 = None
    l__mod___stage4_2_fuse_layers_2_2 = self.L__mod___stage4_2_fuse_layers_2_2(x_953)
    y_82 = y_81 + l__mod___stage4_2_fuse_layers_2_2;  y_81 = l__mod___stage4_2_fuse_layers_2_2 = None
    l__mod___stage4_2_fuse_layers_2_3_0 = self.L__mod___stage4_2_fuse_layers_2_3_0(x_989)
    l__mod___stage4_2_fuse_layers_2_3_1 = self.L__mod___stage4_2_fuse_layers_2_3_1(l__mod___stage4_2_fuse_layers_2_3_0);  l__mod___stage4_2_fuse_layers_2_3_0 = None
    l__mod___stage4_2_fuse_layers_2_3_2 = self.L__mod___stage4_2_fuse_layers_2_3_2(l__mod___stage4_2_fuse_layers_2_3_1);  l__mod___stage4_2_fuse_layers_2_3_1 = None
    y_83 = y_82 + l__mod___stage4_2_fuse_layers_2_3_2;  y_82 = l__mod___stage4_2_fuse_layers_2_3_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_113 = self.L__mod___stage4_2_fuse_act(y_83);  y_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:482, code: y = f(x[j])
    l__mod___stage4_2_fuse_layers_3_0_0_0 = self.L__mod___stage4_2_fuse_layers_3_0_0_0(x_881);  x_881 = None
    l__mod___stage4_2_fuse_layers_3_0_0_1 = self.L__mod___stage4_2_fuse_layers_3_0_0_1(l__mod___stage4_2_fuse_layers_3_0_0_0);  l__mod___stage4_2_fuse_layers_3_0_0_0 = None
    l__mod___stage4_2_fuse_layers_3_0_0_2 = self.L__mod___stage4_2_fuse_layers_3_0_0_2(l__mod___stage4_2_fuse_layers_3_0_0_1);  l__mod___stage4_2_fuse_layers_3_0_0_1 = None
    l__mod___stage4_2_fuse_layers_3_0_1_0 = self.L__mod___stage4_2_fuse_layers_3_0_1_0(l__mod___stage4_2_fuse_layers_3_0_0_2);  l__mod___stage4_2_fuse_layers_3_0_0_2 = None
    l__mod___stage4_2_fuse_layers_3_0_1_1 = self.L__mod___stage4_2_fuse_layers_3_0_1_1(l__mod___stage4_2_fuse_layers_3_0_1_0);  l__mod___stage4_2_fuse_layers_3_0_1_0 = None
    l__mod___stage4_2_fuse_layers_3_0_1_2 = self.L__mod___stage4_2_fuse_layers_3_0_1_2(l__mod___stage4_2_fuse_layers_3_0_1_1);  l__mod___stage4_2_fuse_layers_3_0_1_1 = None
    l__mod___stage4_2_fuse_layers_3_0_2_0 = self.L__mod___stage4_2_fuse_layers_3_0_2_0(l__mod___stage4_2_fuse_layers_3_0_1_2);  l__mod___stage4_2_fuse_layers_3_0_1_2 = None
    y_84 = self.L__mod___stage4_2_fuse_layers_3_0_2_1(l__mod___stage4_2_fuse_layers_3_0_2_0);  l__mod___stage4_2_fuse_layers_3_0_2_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:484, code: y = y + f(x[j])
    l__mod___stage4_2_fuse_layers_3_1_0_0 = self.L__mod___stage4_2_fuse_layers_3_1_0_0(x_917);  x_917 = None
    l__mod___stage4_2_fuse_layers_3_1_0_1 = self.L__mod___stage4_2_fuse_layers_3_1_0_1(l__mod___stage4_2_fuse_layers_3_1_0_0);  l__mod___stage4_2_fuse_layers_3_1_0_0 = None
    l__mod___stage4_2_fuse_layers_3_1_0_2 = self.L__mod___stage4_2_fuse_layers_3_1_0_2(l__mod___stage4_2_fuse_layers_3_1_0_1);  l__mod___stage4_2_fuse_layers_3_1_0_1 = None
    l__mod___stage4_2_fuse_layers_3_1_1_0 = self.L__mod___stage4_2_fuse_layers_3_1_1_0(l__mod___stage4_2_fuse_layers_3_1_0_2);  l__mod___stage4_2_fuse_layers_3_1_0_2 = None
    l__mod___stage4_2_fuse_layers_3_1_1_1 = self.L__mod___stage4_2_fuse_layers_3_1_1_1(l__mod___stage4_2_fuse_layers_3_1_1_0);  l__mod___stage4_2_fuse_layers_3_1_1_0 = None
    y_85 = y_84 + l__mod___stage4_2_fuse_layers_3_1_1_1;  y_84 = l__mod___stage4_2_fuse_layers_3_1_1_1 = None
    l__mod___stage4_2_fuse_layers_3_2_0_0 = self.L__mod___stage4_2_fuse_layers_3_2_0_0(x_953);  x_953 = None
    l__mod___stage4_2_fuse_layers_3_2_0_1 = self.L__mod___stage4_2_fuse_layers_3_2_0_1(l__mod___stage4_2_fuse_layers_3_2_0_0);  l__mod___stage4_2_fuse_layers_3_2_0_0 = None
    y_86 = y_85 + l__mod___stage4_2_fuse_layers_3_2_0_1;  y_85 = l__mod___stage4_2_fuse_layers_3_2_0_1 = None
    l__mod___stage4_2_fuse_layers_3_3 = self.L__mod___stage4_2_fuse_layers_3_3(x_989);  x_989 = None
    y_87 = y_86 + l__mod___stage4_2_fuse_layers_3_3;  y_86 = l__mod___stage4_2_fuse_layers_3_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:485, code: x_fuse.append(self.fuse_act(y))
    shortcut_115 = self.L__mod___stage4_2_fuse_act(y_87);  y_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_990 = self.getattr_L__mod___incre_modules_0___0___conv1(shortcut_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_991 = self.getattr_L__mod___incre_modules_0___0___bn1(x_990);  x_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_992 = self.getattr_L__mod___incre_modules_0___0___act1(x_991);  x_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_993 = self.getattr_L__mod___incre_modules_0___0___conv2(x_992);  x_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_994 = self.getattr_L__mod___incre_modules_0___0___bn2(x_993);  x_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_995 = self.getattr_L__mod___incre_modules_0___0___drop_block(x_994);  x_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_996 = self.getattr_L__mod___incre_modules_0___0___act2(x_995);  x_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_997 = self.getattr_L__mod___incre_modules_0___0___aa(x_996);  x_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_998 = self.getattr_L__mod___incre_modules_0___0___conv3(x_997);  x_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_999 = self.getattr_L__mod___incre_modules_0___0___bn3(x_998);  x_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___incre_modules_0___0___downsample_0 = self.getattr_L__mod___incre_modules_0___0___downsample_0(shortcut_109);  shortcut_109 = None
    shortcut_110 = self.getattr_L__mod___incre_modules_0___0___downsample_1(getattr_l__mod___incre_modules_0___0___downsample_0);  getattr_l__mod___incre_modules_0___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_999 += shortcut_110;  x_1000 = x_999;  x_999 = shortcut_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    y_88 = self.getattr_L__mod___incre_modules_0___0___act3(x_1000);  x_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_1002 = self.getattr_L__mod___incre_modules_1___0___conv1(shortcut_111)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_1003 = self.getattr_L__mod___incre_modules_1___0___bn1(x_1002);  x_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_1004 = self.getattr_L__mod___incre_modules_1___0___act1(x_1003);  x_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_1005 = self.getattr_L__mod___incre_modules_1___0___conv2(x_1004);  x_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_1006 = self.getattr_L__mod___incre_modules_1___0___bn2(x_1005);  x_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_1007 = self.getattr_L__mod___incre_modules_1___0___drop_block(x_1006);  x_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_1008 = self.getattr_L__mod___incre_modules_1___0___act2(x_1007);  x_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_1009 = self.getattr_L__mod___incre_modules_1___0___aa(x_1008);  x_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_1010 = self.getattr_L__mod___incre_modules_1___0___conv3(x_1009);  x_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_1011 = self.getattr_L__mod___incre_modules_1___0___bn3(x_1010);  x_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___incre_modules_1___0___downsample_0 = self.getattr_L__mod___incre_modules_1___0___downsample_0(shortcut_111);  shortcut_111 = None
    shortcut_112 = self.getattr_L__mod___incre_modules_1___0___downsample_1(getattr_l__mod___incre_modules_1___0___downsample_0);  getattr_l__mod___incre_modules_1___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_1011 += shortcut_112;  x_1012 = x_1011;  x_1011 = shortcut_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    x_1013 = self.getattr_L__mod___incre_modules_1___0___act3(x_1012);  x_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:780, code: y = incre(yl[i]) + down.forward(y)
    l__mod___downsamp_modules_0 = self.L__mod___downsamp_modules_0
    forward = l__mod___downsamp_modules_0.forward(y_88);  l__mod___downsamp_modules_0 = y_88 = None
    y_89 = x_1013 + forward;  x_1013 = forward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_1014 = self.getattr_L__mod___incre_modules_2___0___conv1(shortcut_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_1015 = self.getattr_L__mod___incre_modules_2___0___bn1(x_1014);  x_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_1016 = self.getattr_L__mod___incre_modules_2___0___act1(x_1015);  x_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_1017 = self.getattr_L__mod___incre_modules_2___0___conv2(x_1016);  x_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_1018 = self.getattr_L__mod___incre_modules_2___0___bn2(x_1017);  x_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_1019 = self.getattr_L__mod___incre_modules_2___0___drop_block(x_1018);  x_1018 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_1020 = self.getattr_L__mod___incre_modules_2___0___act2(x_1019);  x_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_1021 = self.getattr_L__mod___incre_modules_2___0___aa(x_1020);  x_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_1022 = self.getattr_L__mod___incre_modules_2___0___conv3(x_1021);  x_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_1023 = self.getattr_L__mod___incre_modules_2___0___bn3(x_1022);  x_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___incre_modules_2___0___downsample_0 = self.getattr_L__mod___incre_modules_2___0___downsample_0(shortcut_113);  shortcut_113 = None
    shortcut_114 = self.getattr_L__mod___incre_modules_2___0___downsample_1(getattr_l__mod___incre_modules_2___0___downsample_0);  getattr_l__mod___incre_modules_2___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_1023 += shortcut_114;  x_1024 = x_1023;  x_1023 = shortcut_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    x_1025 = self.getattr_L__mod___incre_modules_2___0___act3(x_1024);  x_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:780, code: y = incre(yl[i]) + down.forward(y)
    l__mod___downsamp_modules_1 = self.L__mod___downsamp_modules_1
    forward_1 = l__mod___downsamp_modules_1.forward(y_89);  l__mod___downsamp_modules_1 = y_89 = None
    y_90 = x_1025 + forward_1;  x_1025 = forward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_1026 = self.getattr_L__mod___incre_modules_3___0___conv1(shortcut_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_1027 = self.getattr_L__mod___incre_modules_3___0___bn1(x_1026);  x_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_1028 = self.getattr_L__mod___incre_modules_3___0___act1(x_1027);  x_1027 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_1029 = self.getattr_L__mod___incre_modules_3___0___conv2(x_1028);  x_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_1030 = self.getattr_L__mod___incre_modules_3___0___bn2(x_1029);  x_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_1031 = self.getattr_L__mod___incre_modules_3___0___drop_block(x_1030);  x_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_1032 = self.getattr_L__mod___incre_modules_3___0___act2(x_1031);  x_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_1033 = self.getattr_L__mod___incre_modules_3___0___aa(x_1032);  x_1032 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_1034 = self.getattr_L__mod___incre_modules_3___0___conv3(x_1033);  x_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_1035 = self.getattr_L__mod___incre_modules_3___0___bn3(x_1034);  x_1034 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___incre_modules_3___0___downsample_0 = self.getattr_L__mod___incre_modules_3___0___downsample_0(shortcut_115);  shortcut_115 = None
    shortcut_116 = self.getattr_L__mod___incre_modules_3___0___downsample_1(getattr_l__mod___incre_modules_3___0___downsample_0);  getattr_l__mod___incre_modules_3___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_1035 += shortcut_116;  x_1036 = x_1035;  x_1035 = shortcut_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    x_1037 = self.getattr_L__mod___incre_modules_3___0___act3(x_1036);  x_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:780, code: y = incre(yl[i]) + down.forward(y)
    l__mod___downsamp_modules_2 = self.L__mod___downsamp_modules_2
    forward_2 = l__mod___downsamp_modules_2.forward(y_90);  l__mod___downsamp_modules_2 = y_90 = None
    y_91 = x_1037 + forward_2;  x_1037 = forward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:782, code: y = self.final_layer(y)
    l__mod___final_layer_0 = self.L__mod___final_layer_0(y_91);  y_91 = None
    l__mod___final_layer_1 = self.L__mod___final_layer_1(l__mod___final_layer_0);  l__mod___final_layer_0 = None
    y_93 = self.L__mod___final_layer_2(l__mod___final_layer_1);  l__mod___final_layer_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_1038 = self.L__mod___global_pool_pool(y_93);  y_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_1040 = self.L__mod___global_pool_flatten(x_1038);  x_1038 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:788, code: x = self.head_drop(x)
    x_1041 = self.L__mod___head_drop(x_1040);  x_1040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/hrnet.py:789, code: return x if pre_logits else self.classifier(x)
    x_1042 = self.L__mod___classifier(x_1041);  x_1041 = None
    return (x_1042,)
    