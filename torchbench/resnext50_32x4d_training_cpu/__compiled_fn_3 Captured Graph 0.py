from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    x = self.L__mod___conv1(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    x_1 = self.L__mod___bn1(x);  x = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    x_2 = self.L__mod___relu(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    identity = self.L__mod___maxpool(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out = self.getattr_L__mod___layer1___0___conv1(identity)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_1 = self.getattr_L__mod___layer1___0___bn1(out);  out = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_2 = self.getattr_L__mod___layer1___0___relu(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_3 = self.getattr_L__mod___layer1___0___conv2(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_4 = self.getattr_L__mod___layer1___0___bn2(out_3);  out_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_5 = self.getattr_L__mod___layer1___0___relu(out_4);  out_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_6 = self.getattr_L__mod___layer1___0___conv3(out_5);  out_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_7 = self.getattr_L__mod___layer1___0___bn3(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    getattr_l__mod___layer1___0___downsample_0 = self.getattr_L__mod___layer1___0___downsample_0(identity);  identity = None
    identity_1 = self.getattr_L__mod___layer1___0___downsample_1(getattr_l__mod___layer1___0___downsample_0);  getattr_l__mod___layer1___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_7 += identity_1;  out_8 = out_7;  out_7 = identity_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_2 = self.getattr_L__mod___layer1___0___relu(out_8);  out_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_10 = self.getattr_L__mod___layer1___1___conv1(identity_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_11 = self.getattr_L__mod___layer1___1___bn1(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_12 = self.getattr_L__mod___layer1___1___relu(out_11);  out_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_13 = self.getattr_L__mod___layer1___1___conv2(out_12);  out_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_14 = self.getattr_L__mod___layer1___1___bn2(out_13);  out_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_15 = self.getattr_L__mod___layer1___1___relu(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_16 = self.getattr_L__mod___layer1___1___conv3(out_15);  out_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_17 = self.getattr_L__mod___layer1___1___bn3(out_16);  out_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_17 += identity_2;  out_18 = out_17;  out_17 = identity_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_3 = self.getattr_L__mod___layer1___1___relu(out_18);  out_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_20 = self.getattr_L__mod___layer1___2___conv1(identity_3)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_21 = self.getattr_L__mod___layer1___2___bn1(out_20);  out_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_22 = self.getattr_L__mod___layer1___2___relu(out_21);  out_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_23 = self.getattr_L__mod___layer1___2___conv2(out_22);  out_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_24 = self.getattr_L__mod___layer1___2___bn2(out_23);  out_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_25 = self.getattr_L__mod___layer1___2___relu(out_24);  out_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_26 = self.getattr_L__mod___layer1___2___conv3(out_25);  out_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_27 = self.getattr_L__mod___layer1___2___bn3(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_27 += identity_3;  out_28 = out_27;  out_27 = identity_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_4 = self.getattr_L__mod___layer1___2___relu(out_28);  out_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_30 = self.getattr_L__mod___layer2___0___conv1(identity_4)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_31 = self.getattr_L__mod___layer2___0___bn1(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_32 = self.getattr_L__mod___layer2___0___relu(out_31);  out_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_33 = self.getattr_L__mod___layer2___0___conv2(out_32);  out_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_34 = self.getattr_L__mod___layer2___0___bn2(out_33);  out_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_35 = self.getattr_L__mod___layer2___0___relu(out_34);  out_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_36 = self.getattr_L__mod___layer2___0___conv3(out_35);  out_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_37 = self.getattr_L__mod___layer2___0___bn3(out_36);  out_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    getattr_l__mod___layer2___0___downsample_0 = self.getattr_L__mod___layer2___0___downsample_0(identity_4);  identity_4 = None
    identity_5 = self.getattr_L__mod___layer2___0___downsample_1(getattr_l__mod___layer2___0___downsample_0);  getattr_l__mod___layer2___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_37 += identity_5;  out_38 = out_37;  out_37 = identity_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_6 = self.getattr_L__mod___layer2___0___relu(out_38);  out_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_40 = self.getattr_L__mod___layer2___1___conv1(identity_6)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_41 = self.getattr_L__mod___layer2___1___bn1(out_40);  out_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_42 = self.getattr_L__mod___layer2___1___relu(out_41);  out_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_43 = self.getattr_L__mod___layer2___1___conv2(out_42);  out_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_44 = self.getattr_L__mod___layer2___1___bn2(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_45 = self.getattr_L__mod___layer2___1___relu(out_44);  out_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_46 = self.getattr_L__mod___layer2___1___conv3(out_45);  out_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_47 = self.getattr_L__mod___layer2___1___bn3(out_46);  out_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_47 += identity_6;  out_48 = out_47;  out_47 = identity_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_7 = self.getattr_L__mod___layer2___1___relu(out_48);  out_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_50 = self.getattr_L__mod___layer2___2___conv1(identity_7)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_51 = self.getattr_L__mod___layer2___2___bn1(out_50);  out_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_52 = self.getattr_L__mod___layer2___2___relu(out_51);  out_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_53 = self.getattr_L__mod___layer2___2___conv2(out_52);  out_52 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_54 = self.getattr_L__mod___layer2___2___bn2(out_53);  out_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_55 = self.getattr_L__mod___layer2___2___relu(out_54);  out_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_56 = self.getattr_L__mod___layer2___2___conv3(out_55);  out_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_57 = self.getattr_L__mod___layer2___2___bn3(out_56);  out_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_57 += identity_7;  out_58 = out_57;  out_57 = identity_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_8 = self.getattr_L__mod___layer2___2___relu(out_58);  out_58 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_60 = self.getattr_L__mod___layer2___3___conv1(identity_8)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_61 = self.getattr_L__mod___layer2___3___bn1(out_60);  out_60 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_62 = self.getattr_L__mod___layer2___3___relu(out_61);  out_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_63 = self.getattr_L__mod___layer2___3___conv2(out_62);  out_62 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_64 = self.getattr_L__mod___layer2___3___bn2(out_63);  out_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_65 = self.getattr_L__mod___layer2___3___relu(out_64);  out_64 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_66 = self.getattr_L__mod___layer2___3___conv3(out_65);  out_65 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_67 = self.getattr_L__mod___layer2___3___bn3(out_66);  out_66 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_67 += identity_8;  out_68 = out_67;  out_67 = identity_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_9 = self.getattr_L__mod___layer2___3___relu(out_68);  out_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_70 = self.getattr_L__mod___layer3___0___conv1(identity_9)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_71 = self.getattr_L__mod___layer3___0___bn1(out_70);  out_70 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_72 = self.getattr_L__mod___layer3___0___relu(out_71);  out_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_73 = self.getattr_L__mod___layer3___0___conv2(out_72);  out_72 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_74 = self.getattr_L__mod___layer3___0___bn2(out_73);  out_73 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_75 = self.getattr_L__mod___layer3___0___relu(out_74);  out_74 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_76 = self.getattr_L__mod___layer3___0___conv3(out_75);  out_75 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_77 = self.getattr_L__mod___layer3___0___bn3(out_76);  out_76 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    getattr_l__mod___layer3___0___downsample_0 = self.getattr_L__mod___layer3___0___downsample_0(identity_9);  identity_9 = None
    identity_10 = self.getattr_L__mod___layer3___0___downsample_1(getattr_l__mod___layer3___0___downsample_0);  getattr_l__mod___layer3___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_77 += identity_10;  out_78 = out_77;  out_77 = identity_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_11 = self.getattr_L__mod___layer3___0___relu(out_78);  out_78 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_80 = self.getattr_L__mod___layer3___1___conv1(identity_11)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_81 = self.getattr_L__mod___layer3___1___bn1(out_80);  out_80 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_82 = self.getattr_L__mod___layer3___1___relu(out_81);  out_81 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_83 = self.getattr_L__mod___layer3___1___conv2(out_82);  out_82 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_84 = self.getattr_L__mod___layer3___1___bn2(out_83);  out_83 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_85 = self.getattr_L__mod___layer3___1___relu(out_84);  out_84 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_86 = self.getattr_L__mod___layer3___1___conv3(out_85);  out_85 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_87 = self.getattr_L__mod___layer3___1___bn3(out_86);  out_86 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_87 += identity_11;  out_88 = out_87;  out_87 = identity_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_12 = self.getattr_L__mod___layer3___1___relu(out_88);  out_88 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_90 = self.getattr_L__mod___layer3___2___conv1(identity_12)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_91 = self.getattr_L__mod___layer3___2___bn1(out_90);  out_90 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_92 = self.getattr_L__mod___layer3___2___relu(out_91);  out_91 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_93 = self.getattr_L__mod___layer3___2___conv2(out_92);  out_92 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_94 = self.getattr_L__mod___layer3___2___bn2(out_93);  out_93 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_95 = self.getattr_L__mod___layer3___2___relu(out_94);  out_94 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_96 = self.getattr_L__mod___layer3___2___conv3(out_95);  out_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_97 = self.getattr_L__mod___layer3___2___bn3(out_96);  out_96 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_97 += identity_12;  out_98 = out_97;  out_97 = identity_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_13 = self.getattr_L__mod___layer3___2___relu(out_98);  out_98 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_100 = self.getattr_L__mod___layer3___3___conv1(identity_13)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_101 = self.getattr_L__mod___layer3___3___bn1(out_100);  out_100 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_102 = self.getattr_L__mod___layer3___3___relu(out_101);  out_101 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_103 = self.getattr_L__mod___layer3___3___conv2(out_102);  out_102 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_104 = self.getattr_L__mod___layer3___3___bn2(out_103);  out_103 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_105 = self.getattr_L__mod___layer3___3___relu(out_104);  out_104 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_106 = self.getattr_L__mod___layer3___3___conv3(out_105);  out_105 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_107 = self.getattr_L__mod___layer3___3___bn3(out_106);  out_106 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_107 += identity_13;  out_108 = out_107;  out_107 = identity_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_14 = self.getattr_L__mod___layer3___3___relu(out_108);  out_108 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_110 = self.getattr_L__mod___layer3___4___conv1(identity_14)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_111 = self.getattr_L__mod___layer3___4___bn1(out_110);  out_110 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_112 = self.getattr_L__mod___layer3___4___relu(out_111);  out_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_113 = self.getattr_L__mod___layer3___4___conv2(out_112);  out_112 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_114 = self.getattr_L__mod___layer3___4___bn2(out_113);  out_113 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_115 = self.getattr_L__mod___layer3___4___relu(out_114);  out_114 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_116 = self.getattr_L__mod___layer3___4___conv3(out_115);  out_115 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_117 = self.getattr_L__mod___layer3___4___bn3(out_116);  out_116 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_117 += identity_14;  out_118 = out_117;  out_117 = identity_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_15 = self.getattr_L__mod___layer3___4___relu(out_118);  out_118 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_120 = self.getattr_L__mod___layer3___5___conv1(identity_15)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_121 = self.getattr_L__mod___layer3___5___bn1(out_120);  out_120 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_122 = self.getattr_L__mod___layer3___5___relu(out_121);  out_121 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_123 = self.getattr_L__mod___layer3___5___conv2(out_122);  out_122 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_124 = self.getattr_L__mod___layer3___5___bn2(out_123);  out_123 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_125 = self.getattr_L__mod___layer3___5___relu(out_124);  out_124 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_126 = self.getattr_L__mod___layer3___5___conv3(out_125);  out_125 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_127 = self.getattr_L__mod___layer3___5___bn3(out_126);  out_126 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_127 += identity_15;  out_128 = out_127;  out_127 = identity_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_16 = self.getattr_L__mod___layer3___5___relu(out_128);  out_128 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_130 = self.getattr_L__mod___layer4___0___conv1(identity_16)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_131 = self.getattr_L__mod___layer4___0___bn1(out_130);  out_130 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_132 = self.getattr_L__mod___layer4___0___relu(out_131);  out_131 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_133 = self.getattr_L__mod___layer4___0___conv2(out_132);  out_132 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_134 = self.getattr_L__mod___layer4___0___bn2(out_133);  out_133 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_135 = self.getattr_L__mod___layer4___0___relu(out_134);  out_134 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_136 = self.getattr_L__mod___layer4___0___conv3(out_135);  out_135 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_137 = self.getattr_L__mod___layer4___0___bn3(out_136);  out_136 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:158, code: identity = self.downsample(x)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(identity_16);  identity_16 = None
    identity_17 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_137 += identity_17;  out_138 = out_137;  out_137 = identity_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_18 = self.getattr_L__mod___layer4___0___relu(out_138);  out_138 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_140 = self.getattr_L__mod___layer4___1___conv1(identity_18)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_141 = self.getattr_L__mod___layer4___1___bn1(out_140);  out_140 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_142 = self.getattr_L__mod___layer4___1___relu(out_141);  out_141 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_143 = self.getattr_L__mod___layer4___1___conv2(out_142);  out_142 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_144 = self.getattr_L__mod___layer4___1___bn2(out_143);  out_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_145 = self.getattr_L__mod___layer4___1___relu(out_144);  out_144 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_146 = self.getattr_L__mod___layer4___1___conv3(out_145);  out_145 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_147 = self.getattr_L__mod___layer4___1___bn3(out_146);  out_146 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_147 += identity_18;  out_148 = out_147;  out_147 = identity_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    identity_19 = self.getattr_L__mod___layer4___1___relu(out_148);  out_148 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:146, code: out = self.conv1(x)
    out_150 = self.getattr_L__mod___layer4___2___conv1(identity_19)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:147, code: out = self.bn1(out)
    out_151 = self.getattr_L__mod___layer4___2___bn1(out_150);  out_150 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:148, code: out = self.relu(out)
    out_152 = self.getattr_L__mod___layer4___2___relu(out_151);  out_151 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:150, code: out = self.conv2(out)
    out_153 = self.getattr_L__mod___layer4___2___conv2(out_152);  out_152 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:151, code: out = self.bn2(out)
    out_154 = self.getattr_L__mod___layer4___2___bn2(out_153);  out_153 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:152, code: out = self.relu(out)
    out_155 = self.getattr_L__mod___layer4___2___relu(out_154);  out_154 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:154, code: out = self.conv3(out)
    out_156 = self.getattr_L__mod___layer4___2___conv3(out_155);  out_155 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:155, code: out = self.bn3(out)
    out_157 = self.getattr_L__mod___layer4___2___bn3(out_156);  out_156 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:160, code: out += identity
    out_157 += identity_19;  out_158 = out_157;  out_157 = identity_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:161, code: out = self.relu(out)
    x_7 = self.getattr_L__mod___layer4___2___relu(out_158);  out_158 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    x_8 = self.L__mod___avgpool(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    x_9 = torch.flatten(x_8, 1);  x_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    pred = self.L__mod___fc(x_9);  x_9 = None
    return (pred,)
    