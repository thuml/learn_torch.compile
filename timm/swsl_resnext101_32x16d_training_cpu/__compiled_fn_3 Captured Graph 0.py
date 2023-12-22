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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_4 = self.getattr_L__mod___layer1___0___conv1(shortcut)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_5 = self.getattr_L__mod___layer1___0___bn1(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_6 = self.getattr_L__mod___layer1___0___act1(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_7 = self.getattr_L__mod___layer1___0___conv2(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_8 = self.getattr_L__mod___layer1___0___bn2(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_9 = self.getattr_L__mod___layer1___0___drop_block(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_10 = self.getattr_L__mod___layer1___0___act2(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_11 = self.getattr_L__mod___layer1___0___aa(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_12 = self.getattr_L__mod___layer1___0___conv3(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_13 = self.getattr_L__mod___layer1___0___bn3(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___layer1___0___downsample_0 = self.getattr_L__mod___layer1___0___downsample_0(shortcut);  shortcut = None
    shortcut_1 = self.getattr_L__mod___layer1___0___downsample_1(getattr_l__mod___layer1___0___downsample_0);  getattr_l__mod___layer1___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_13 += shortcut_1;  x_14 = x_13;  x_13 = shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_2 = self.getattr_L__mod___layer1___0___act3(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_16 = self.getattr_L__mod___layer1___1___conv1(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_17 = self.getattr_L__mod___layer1___1___bn1(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_18 = self.getattr_L__mod___layer1___1___act1(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_19 = self.getattr_L__mod___layer1___1___conv2(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_20 = self.getattr_L__mod___layer1___1___bn2(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_21 = self.getattr_L__mod___layer1___1___drop_block(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_22 = self.getattr_L__mod___layer1___1___act2(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_23 = self.getattr_L__mod___layer1___1___aa(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_24 = self.getattr_L__mod___layer1___1___conv3(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_25 = self.getattr_L__mod___layer1___1___bn3(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_25 += shortcut_2;  x_26 = x_25;  x_25 = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_3 = self.getattr_L__mod___layer1___1___act3(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_28 = self.getattr_L__mod___layer1___2___conv1(shortcut_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_29 = self.getattr_L__mod___layer1___2___bn1(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_30 = self.getattr_L__mod___layer1___2___act1(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_31 = self.getattr_L__mod___layer1___2___conv2(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_32 = self.getattr_L__mod___layer1___2___bn2(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_33 = self.getattr_L__mod___layer1___2___drop_block(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_34 = self.getattr_L__mod___layer1___2___act2(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_35 = self.getattr_L__mod___layer1___2___aa(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_36 = self.getattr_L__mod___layer1___2___conv3(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_37 = self.getattr_L__mod___layer1___2___bn3(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_37 += shortcut_3;  x_38 = x_37;  x_37 = shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_4 = self.getattr_L__mod___layer1___2___act3(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_41 = self.getattr_L__mod___layer2___0___conv1(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_42 = self.getattr_L__mod___layer2___0___bn1(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_43 = self.getattr_L__mod___layer2___0___act1(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_44 = self.getattr_L__mod___layer2___0___conv2(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_45 = self.getattr_L__mod___layer2___0___bn2(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_46 = self.getattr_L__mod___layer2___0___drop_block(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_47 = self.getattr_L__mod___layer2___0___act2(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_48 = self.getattr_L__mod___layer2___0___aa(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_49 = self.getattr_L__mod___layer2___0___conv3(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_50 = self.getattr_L__mod___layer2___0___bn3(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___layer2___0___downsample_0 = self.getattr_L__mod___layer2___0___downsample_0(shortcut_4);  shortcut_4 = None
    shortcut_5 = self.getattr_L__mod___layer2___0___downsample_1(getattr_l__mod___layer2___0___downsample_0);  getattr_l__mod___layer2___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_50 += shortcut_5;  x_51 = x_50;  x_50 = shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_6 = self.getattr_L__mod___layer2___0___act3(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_53 = self.getattr_L__mod___layer2___1___conv1(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_54 = self.getattr_L__mod___layer2___1___bn1(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_55 = self.getattr_L__mod___layer2___1___act1(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_56 = self.getattr_L__mod___layer2___1___conv2(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_57 = self.getattr_L__mod___layer2___1___bn2(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_58 = self.getattr_L__mod___layer2___1___drop_block(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_59 = self.getattr_L__mod___layer2___1___act2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_60 = self.getattr_L__mod___layer2___1___aa(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_61 = self.getattr_L__mod___layer2___1___conv3(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_62 = self.getattr_L__mod___layer2___1___bn3(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_62 += shortcut_6;  x_63 = x_62;  x_62 = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_7 = self.getattr_L__mod___layer2___1___act3(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_65 = self.getattr_L__mod___layer2___2___conv1(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_66 = self.getattr_L__mod___layer2___2___bn1(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_67 = self.getattr_L__mod___layer2___2___act1(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_68 = self.getattr_L__mod___layer2___2___conv2(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_69 = self.getattr_L__mod___layer2___2___bn2(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_70 = self.getattr_L__mod___layer2___2___drop_block(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_71 = self.getattr_L__mod___layer2___2___act2(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_72 = self.getattr_L__mod___layer2___2___aa(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_73 = self.getattr_L__mod___layer2___2___conv3(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_74 = self.getattr_L__mod___layer2___2___bn3(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_74 += shortcut_7;  x_75 = x_74;  x_74 = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_8 = self.getattr_L__mod___layer2___2___act3(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_77 = self.getattr_L__mod___layer2___3___conv1(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_78 = self.getattr_L__mod___layer2___3___bn1(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_79 = self.getattr_L__mod___layer2___3___act1(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_80 = self.getattr_L__mod___layer2___3___conv2(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_81 = self.getattr_L__mod___layer2___3___bn2(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_82 = self.getattr_L__mod___layer2___3___drop_block(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_83 = self.getattr_L__mod___layer2___3___act2(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_84 = self.getattr_L__mod___layer2___3___aa(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_85 = self.getattr_L__mod___layer2___3___conv3(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_86 = self.getattr_L__mod___layer2___3___bn3(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_86 += shortcut_8;  x_87 = x_86;  x_86 = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_9 = self.getattr_L__mod___layer2___3___act3(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_90 = self.getattr_L__mod___layer3___0___conv1(shortcut_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_91 = self.getattr_L__mod___layer3___0___bn1(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_92 = self.getattr_L__mod___layer3___0___act1(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_93 = self.getattr_L__mod___layer3___0___conv2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_94 = self.getattr_L__mod___layer3___0___bn2(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_95 = self.getattr_L__mod___layer3___0___drop_block(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_96 = self.getattr_L__mod___layer3___0___act2(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_97 = self.getattr_L__mod___layer3___0___aa(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_98 = self.getattr_L__mod___layer3___0___conv3(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_99 = self.getattr_L__mod___layer3___0___bn3(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___layer3___0___downsample_0 = self.getattr_L__mod___layer3___0___downsample_0(shortcut_9);  shortcut_9 = None
    shortcut_10 = self.getattr_L__mod___layer3___0___downsample_1(getattr_l__mod___layer3___0___downsample_0);  getattr_l__mod___layer3___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_99 += shortcut_10;  x_100 = x_99;  x_99 = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_11 = self.getattr_L__mod___layer3___0___act3(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_102 = self.getattr_L__mod___layer3___1___conv1(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_103 = self.getattr_L__mod___layer3___1___bn1(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_104 = self.getattr_L__mod___layer3___1___act1(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_105 = self.getattr_L__mod___layer3___1___conv2(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_106 = self.getattr_L__mod___layer3___1___bn2(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_107 = self.getattr_L__mod___layer3___1___drop_block(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_108 = self.getattr_L__mod___layer3___1___act2(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_109 = self.getattr_L__mod___layer3___1___aa(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_110 = self.getattr_L__mod___layer3___1___conv3(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_111 = self.getattr_L__mod___layer3___1___bn3(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_111 += shortcut_11;  x_112 = x_111;  x_111 = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_12 = self.getattr_L__mod___layer3___1___act3(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_114 = self.getattr_L__mod___layer3___2___conv1(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_115 = self.getattr_L__mod___layer3___2___bn1(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_116 = self.getattr_L__mod___layer3___2___act1(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_117 = self.getattr_L__mod___layer3___2___conv2(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_118 = self.getattr_L__mod___layer3___2___bn2(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_119 = self.getattr_L__mod___layer3___2___drop_block(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_120 = self.getattr_L__mod___layer3___2___act2(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_121 = self.getattr_L__mod___layer3___2___aa(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_122 = self.getattr_L__mod___layer3___2___conv3(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_123 = self.getattr_L__mod___layer3___2___bn3(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_123 += shortcut_12;  x_124 = x_123;  x_123 = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_13 = self.getattr_L__mod___layer3___2___act3(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_126 = self.getattr_L__mod___layer3___3___conv1(shortcut_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_127 = self.getattr_L__mod___layer3___3___bn1(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_128 = self.getattr_L__mod___layer3___3___act1(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_129 = self.getattr_L__mod___layer3___3___conv2(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_130 = self.getattr_L__mod___layer3___3___bn2(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_131 = self.getattr_L__mod___layer3___3___drop_block(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_132 = self.getattr_L__mod___layer3___3___act2(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_133 = self.getattr_L__mod___layer3___3___aa(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_134 = self.getattr_L__mod___layer3___3___conv3(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_135 = self.getattr_L__mod___layer3___3___bn3(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_135 += shortcut_13;  x_136 = x_135;  x_135 = shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_14 = self.getattr_L__mod___layer3___3___act3(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_138 = self.getattr_L__mod___layer3___4___conv1(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_139 = self.getattr_L__mod___layer3___4___bn1(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_140 = self.getattr_L__mod___layer3___4___act1(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_141 = self.getattr_L__mod___layer3___4___conv2(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_142 = self.getattr_L__mod___layer3___4___bn2(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_143 = self.getattr_L__mod___layer3___4___drop_block(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_144 = self.getattr_L__mod___layer3___4___act2(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_145 = self.getattr_L__mod___layer3___4___aa(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_146 = self.getattr_L__mod___layer3___4___conv3(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_147 = self.getattr_L__mod___layer3___4___bn3(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_147 += shortcut_14;  x_148 = x_147;  x_147 = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_15 = self.getattr_L__mod___layer3___4___act3(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_150 = self.getattr_L__mod___layer3___5___conv1(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_151 = self.getattr_L__mod___layer3___5___bn1(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_152 = self.getattr_L__mod___layer3___5___act1(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_153 = self.getattr_L__mod___layer3___5___conv2(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_154 = self.getattr_L__mod___layer3___5___bn2(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_155 = self.getattr_L__mod___layer3___5___drop_block(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_156 = self.getattr_L__mod___layer3___5___act2(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_157 = self.getattr_L__mod___layer3___5___aa(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_158 = self.getattr_L__mod___layer3___5___conv3(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_159 = self.getattr_L__mod___layer3___5___bn3(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_159 += shortcut_15;  x_160 = x_159;  x_159 = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_16 = self.getattr_L__mod___layer3___5___act3(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_162 = self.getattr_L__mod___layer3___6___conv1(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_163 = self.getattr_L__mod___layer3___6___bn1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_164 = self.getattr_L__mod___layer3___6___act1(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_165 = self.getattr_L__mod___layer3___6___conv2(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_166 = self.getattr_L__mod___layer3___6___bn2(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_167 = self.getattr_L__mod___layer3___6___drop_block(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_168 = self.getattr_L__mod___layer3___6___act2(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_169 = self.getattr_L__mod___layer3___6___aa(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_170 = self.getattr_L__mod___layer3___6___conv3(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_171 = self.getattr_L__mod___layer3___6___bn3(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_171 += shortcut_16;  x_172 = x_171;  x_171 = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_17 = self.getattr_L__mod___layer3___6___act3(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_174 = self.getattr_L__mod___layer3___7___conv1(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_175 = self.getattr_L__mod___layer3___7___bn1(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_176 = self.getattr_L__mod___layer3___7___act1(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_177 = self.getattr_L__mod___layer3___7___conv2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_178 = self.getattr_L__mod___layer3___7___bn2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_179 = self.getattr_L__mod___layer3___7___drop_block(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_180 = self.getattr_L__mod___layer3___7___act2(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_181 = self.getattr_L__mod___layer3___7___aa(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_182 = self.getattr_L__mod___layer3___7___conv3(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_183 = self.getattr_L__mod___layer3___7___bn3(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_183 += shortcut_17;  x_184 = x_183;  x_183 = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_18 = self.getattr_L__mod___layer3___7___act3(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_186 = self.getattr_L__mod___layer3___8___conv1(shortcut_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_187 = self.getattr_L__mod___layer3___8___bn1(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_188 = self.getattr_L__mod___layer3___8___act1(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_189 = self.getattr_L__mod___layer3___8___conv2(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_190 = self.getattr_L__mod___layer3___8___bn2(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_191 = self.getattr_L__mod___layer3___8___drop_block(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_192 = self.getattr_L__mod___layer3___8___act2(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_193 = self.getattr_L__mod___layer3___8___aa(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_194 = self.getattr_L__mod___layer3___8___conv3(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_195 = self.getattr_L__mod___layer3___8___bn3(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_195 += shortcut_18;  x_196 = x_195;  x_195 = shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_19 = self.getattr_L__mod___layer3___8___act3(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_198 = self.getattr_L__mod___layer3___9___conv1(shortcut_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_199 = self.getattr_L__mod___layer3___9___bn1(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_200 = self.getattr_L__mod___layer3___9___act1(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_201 = self.getattr_L__mod___layer3___9___conv2(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_202 = self.getattr_L__mod___layer3___9___bn2(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_203 = self.getattr_L__mod___layer3___9___drop_block(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_204 = self.getattr_L__mod___layer3___9___act2(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_205 = self.getattr_L__mod___layer3___9___aa(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_206 = self.getattr_L__mod___layer3___9___conv3(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_207 = self.getattr_L__mod___layer3___9___bn3(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_207 += shortcut_19;  x_208 = x_207;  x_207 = shortcut_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_20 = self.getattr_L__mod___layer3___9___act3(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_210 = self.getattr_L__mod___layer3___10___conv1(shortcut_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_211 = self.getattr_L__mod___layer3___10___bn1(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_212 = self.getattr_L__mod___layer3___10___act1(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_213 = self.getattr_L__mod___layer3___10___conv2(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_214 = self.getattr_L__mod___layer3___10___bn2(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_215 = self.getattr_L__mod___layer3___10___drop_block(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_216 = self.getattr_L__mod___layer3___10___act2(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_217 = self.getattr_L__mod___layer3___10___aa(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_218 = self.getattr_L__mod___layer3___10___conv3(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_219 = self.getattr_L__mod___layer3___10___bn3(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_219 += shortcut_20;  x_220 = x_219;  x_219 = shortcut_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_21 = self.getattr_L__mod___layer3___10___act3(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_222 = self.getattr_L__mod___layer3___11___conv1(shortcut_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_223 = self.getattr_L__mod___layer3___11___bn1(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_224 = self.getattr_L__mod___layer3___11___act1(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_225 = self.getattr_L__mod___layer3___11___conv2(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_226 = self.getattr_L__mod___layer3___11___bn2(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_227 = self.getattr_L__mod___layer3___11___drop_block(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_228 = self.getattr_L__mod___layer3___11___act2(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_229 = self.getattr_L__mod___layer3___11___aa(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_230 = self.getattr_L__mod___layer3___11___conv3(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_231 = self.getattr_L__mod___layer3___11___bn3(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_231 += shortcut_21;  x_232 = x_231;  x_231 = shortcut_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_22 = self.getattr_L__mod___layer3___11___act3(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_234 = self.getattr_L__mod___layer3___12___conv1(shortcut_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_235 = self.getattr_L__mod___layer3___12___bn1(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_236 = self.getattr_L__mod___layer3___12___act1(x_235);  x_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_237 = self.getattr_L__mod___layer3___12___conv2(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_238 = self.getattr_L__mod___layer3___12___bn2(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_239 = self.getattr_L__mod___layer3___12___drop_block(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_240 = self.getattr_L__mod___layer3___12___act2(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_241 = self.getattr_L__mod___layer3___12___aa(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_242 = self.getattr_L__mod___layer3___12___conv3(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_243 = self.getattr_L__mod___layer3___12___bn3(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_243 += shortcut_22;  x_244 = x_243;  x_243 = shortcut_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_23 = self.getattr_L__mod___layer3___12___act3(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_246 = self.getattr_L__mod___layer3___13___conv1(shortcut_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_247 = self.getattr_L__mod___layer3___13___bn1(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_248 = self.getattr_L__mod___layer3___13___act1(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_249 = self.getattr_L__mod___layer3___13___conv2(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_250 = self.getattr_L__mod___layer3___13___bn2(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_251 = self.getattr_L__mod___layer3___13___drop_block(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_252 = self.getattr_L__mod___layer3___13___act2(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_253 = self.getattr_L__mod___layer3___13___aa(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_254 = self.getattr_L__mod___layer3___13___conv3(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_255 = self.getattr_L__mod___layer3___13___bn3(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_255 += shortcut_23;  x_256 = x_255;  x_255 = shortcut_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_24 = self.getattr_L__mod___layer3___13___act3(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_258 = self.getattr_L__mod___layer3___14___conv1(shortcut_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_259 = self.getattr_L__mod___layer3___14___bn1(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_260 = self.getattr_L__mod___layer3___14___act1(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_261 = self.getattr_L__mod___layer3___14___conv2(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_262 = self.getattr_L__mod___layer3___14___bn2(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_263 = self.getattr_L__mod___layer3___14___drop_block(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_264 = self.getattr_L__mod___layer3___14___act2(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_265 = self.getattr_L__mod___layer3___14___aa(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_266 = self.getattr_L__mod___layer3___14___conv3(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_267 = self.getattr_L__mod___layer3___14___bn3(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_267 += shortcut_24;  x_268 = x_267;  x_267 = shortcut_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_25 = self.getattr_L__mod___layer3___14___act3(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_270 = self.getattr_L__mod___layer3___15___conv1(shortcut_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_271 = self.getattr_L__mod___layer3___15___bn1(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_272 = self.getattr_L__mod___layer3___15___act1(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_273 = self.getattr_L__mod___layer3___15___conv2(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_274 = self.getattr_L__mod___layer3___15___bn2(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_275 = self.getattr_L__mod___layer3___15___drop_block(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_276 = self.getattr_L__mod___layer3___15___act2(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_277 = self.getattr_L__mod___layer3___15___aa(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_278 = self.getattr_L__mod___layer3___15___conv3(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_279 = self.getattr_L__mod___layer3___15___bn3(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_279 += shortcut_25;  x_280 = x_279;  x_279 = shortcut_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_26 = self.getattr_L__mod___layer3___15___act3(x_280);  x_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_282 = self.getattr_L__mod___layer3___16___conv1(shortcut_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_283 = self.getattr_L__mod___layer3___16___bn1(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_284 = self.getattr_L__mod___layer3___16___act1(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_285 = self.getattr_L__mod___layer3___16___conv2(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_286 = self.getattr_L__mod___layer3___16___bn2(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_287 = self.getattr_L__mod___layer3___16___drop_block(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_288 = self.getattr_L__mod___layer3___16___act2(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_289 = self.getattr_L__mod___layer3___16___aa(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_290 = self.getattr_L__mod___layer3___16___conv3(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_291 = self.getattr_L__mod___layer3___16___bn3(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_291 += shortcut_26;  x_292 = x_291;  x_291 = shortcut_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_27 = self.getattr_L__mod___layer3___16___act3(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_294 = self.getattr_L__mod___layer3___17___conv1(shortcut_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_295 = self.getattr_L__mod___layer3___17___bn1(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_296 = self.getattr_L__mod___layer3___17___act1(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_297 = self.getattr_L__mod___layer3___17___conv2(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_298 = self.getattr_L__mod___layer3___17___bn2(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_299 = self.getattr_L__mod___layer3___17___drop_block(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_300 = self.getattr_L__mod___layer3___17___act2(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_301 = self.getattr_L__mod___layer3___17___aa(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_302 = self.getattr_L__mod___layer3___17___conv3(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_303 = self.getattr_L__mod___layer3___17___bn3(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_303 += shortcut_27;  x_304 = x_303;  x_303 = shortcut_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_28 = self.getattr_L__mod___layer3___17___act3(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_306 = self.getattr_L__mod___layer3___18___conv1(shortcut_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_307 = self.getattr_L__mod___layer3___18___bn1(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_308 = self.getattr_L__mod___layer3___18___act1(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_309 = self.getattr_L__mod___layer3___18___conv2(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_310 = self.getattr_L__mod___layer3___18___bn2(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_311 = self.getattr_L__mod___layer3___18___drop_block(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_312 = self.getattr_L__mod___layer3___18___act2(x_311);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_313 = self.getattr_L__mod___layer3___18___aa(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_314 = self.getattr_L__mod___layer3___18___conv3(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_315 = self.getattr_L__mod___layer3___18___bn3(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_315 += shortcut_28;  x_316 = x_315;  x_315 = shortcut_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_29 = self.getattr_L__mod___layer3___18___act3(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_318 = self.getattr_L__mod___layer3___19___conv1(shortcut_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_319 = self.getattr_L__mod___layer3___19___bn1(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_320 = self.getattr_L__mod___layer3___19___act1(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_321 = self.getattr_L__mod___layer3___19___conv2(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_322 = self.getattr_L__mod___layer3___19___bn2(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_323 = self.getattr_L__mod___layer3___19___drop_block(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_324 = self.getattr_L__mod___layer3___19___act2(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_325 = self.getattr_L__mod___layer3___19___aa(x_324);  x_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_326 = self.getattr_L__mod___layer3___19___conv3(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_327 = self.getattr_L__mod___layer3___19___bn3(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_327 += shortcut_29;  x_328 = x_327;  x_327 = shortcut_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_30 = self.getattr_L__mod___layer3___19___act3(x_328);  x_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_330 = self.getattr_L__mod___layer3___20___conv1(shortcut_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_331 = self.getattr_L__mod___layer3___20___bn1(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_332 = self.getattr_L__mod___layer3___20___act1(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_333 = self.getattr_L__mod___layer3___20___conv2(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_334 = self.getattr_L__mod___layer3___20___bn2(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_335 = self.getattr_L__mod___layer3___20___drop_block(x_334);  x_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_336 = self.getattr_L__mod___layer3___20___act2(x_335);  x_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_337 = self.getattr_L__mod___layer3___20___aa(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_338 = self.getattr_L__mod___layer3___20___conv3(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_339 = self.getattr_L__mod___layer3___20___bn3(x_338);  x_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_339 += shortcut_30;  x_340 = x_339;  x_339 = shortcut_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_31 = self.getattr_L__mod___layer3___20___act3(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_342 = self.getattr_L__mod___layer3___21___conv1(shortcut_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_343 = self.getattr_L__mod___layer3___21___bn1(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_344 = self.getattr_L__mod___layer3___21___act1(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_345 = self.getattr_L__mod___layer3___21___conv2(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_346 = self.getattr_L__mod___layer3___21___bn2(x_345);  x_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_347 = self.getattr_L__mod___layer3___21___drop_block(x_346);  x_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_348 = self.getattr_L__mod___layer3___21___act2(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_349 = self.getattr_L__mod___layer3___21___aa(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_350 = self.getattr_L__mod___layer3___21___conv3(x_349);  x_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_351 = self.getattr_L__mod___layer3___21___bn3(x_350);  x_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_351 += shortcut_31;  x_352 = x_351;  x_351 = shortcut_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_32 = self.getattr_L__mod___layer3___21___act3(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_354 = self.getattr_L__mod___layer3___22___conv1(shortcut_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_355 = self.getattr_L__mod___layer3___22___bn1(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_356 = self.getattr_L__mod___layer3___22___act1(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_357 = self.getattr_L__mod___layer3___22___conv2(x_356);  x_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_358 = self.getattr_L__mod___layer3___22___bn2(x_357);  x_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_359 = self.getattr_L__mod___layer3___22___drop_block(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_360 = self.getattr_L__mod___layer3___22___act2(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_361 = self.getattr_L__mod___layer3___22___aa(x_360);  x_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_362 = self.getattr_L__mod___layer3___22___conv3(x_361);  x_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_363 = self.getattr_L__mod___layer3___22___bn3(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_363 += shortcut_32;  x_364 = x_363;  x_363 = shortcut_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_33 = self.getattr_L__mod___layer3___22___act3(x_364);  x_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_367 = self.getattr_L__mod___layer4___0___conv1(shortcut_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_368 = self.getattr_L__mod___layer4___0___bn1(x_367);  x_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_369 = self.getattr_L__mod___layer4___0___act1(x_368);  x_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_370 = self.getattr_L__mod___layer4___0___conv2(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_371 = self.getattr_L__mod___layer4___0___bn2(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_372 = self.getattr_L__mod___layer4___0___drop_block(x_371);  x_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_373 = self.getattr_L__mod___layer4___0___act2(x_372);  x_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_374 = self.getattr_L__mod___layer4___0___aa(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_375 = self.getattr_L__mod___layer4___0___conv3(x_374);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_376 = self.getattr_L__mod___layer4___0___bn3(x_375);  x_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(shortcut_33);  shortcut_33 = None
    shortcut_34 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_376 += shortcut_34;  x_377 = x_376;  x_376 = shortcut_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_35 = self.getattr_L__mod___layer4___0___act3(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_379 = self.getattr_L__mod___layer4___1___conv1(shortcut_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_380 = self.getattr_L__mod___layer4___1___bn1(x_379);  x_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_381 = self.getattr_L__mod___layer4___1___act1(x_380);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_382 = self.getattr_L__mod___layer4___1___conv2(x_381);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_383 = self.getattr_L__mod___layer4___1___bn2(x_382);  x_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_384 = self.getattr_L__mod___layer4___1___drop_block(x_383);  x_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_385 = self.getattr_L__mod___layer4___1___act2(x_384);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_386 = self.getattr_L__mod___layer4___1___aa(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_387 = self.getattr_L__mod___layer4___1___conv3(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_388 = self.getattr_L__mod___layer4___1___bn3(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_388 += shortcut_35;  x_389 = x_388;  x_388 = shortcut_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    shortcut_36 = self.getattr_L__mod___layer4___1___act3(x_389);  x_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    x_391 = self.getattr_L__mod___layer4___2___conv1(shortcut_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    x_392 = self.getattr_L__mod___layer4___2___bn1(x_391);  x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    x_393 = self.getattr_L__mod___layer4___2___act1(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    x_394 = self.getattr_L__mod___layer4___2___conv2(x_393);  x_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    x_395 = self.getattr_L__mod___layer4___2___bn2(x_394);  x_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:186, code: x = self.drop_block(x)
    x_396 = self.getattr_L__mod___layer4___2___drop_block(x_395);  x_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    x_397 = self.getattr_L__mod___layer4___2___act2(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:188, code: x = self.aa(x)
    x_398 = self.getattr_L__mod___layer4___2___aa(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    x_399 = self.getattr_L__mod___layer4___2___conv3(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    x_400 = self.getattr_L__mod___layer4___2___bn3(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    x_400 += shortcut_36;  x_401 = x_400;  x_400 = shortcut_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    x_404 = self.getattr_L__mod___layer4___2___act3(x_401);  x_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_405 = self.L__mod___global_pool_pool(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_407 = self.L__mod___global_pool_flatten(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    pred = self.L__mod___fc(x_407);  x_407 = None
    return (pred,)
    