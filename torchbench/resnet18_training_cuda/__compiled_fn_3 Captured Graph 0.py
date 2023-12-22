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
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out = self.getattr_L__mod___layer1___0___conv1(identity)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_1 = self.getattr_L__mod___layer1___0___bn1(out);  out = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_2 = self.getattr_L__mod___layer1___0___relu(out_1);  out_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_3 = self.getattr_L__mod___layer1___0___conv2(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_4 = self.getattr_L__mod___layer1___0___bn2(out_3);  out_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_4 += identity;  out_5 = out_4;  out_4 = identity = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_1 = self.getattr_L__mod___layer1___0___relu(out_5);  out_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_7 = self.getattr_L__mod___layer1___1___conv1(identity_1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_8 = self.getattr_L__mod___layer1___1___bn1(out_7);  out_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_9 = self.getattr_L__mod___layer1___1___relu(out_8);  out_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_10 = self.getattr_L__mod___layer1___1___conv2(out_9);  out_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_11 = self.getattr_L__mod___layer1___1___bn2(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_11 += identity_1;  out_12 = out_11;  out_11 = identity_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_2 = self.getattr_L__mod___layer1___1___relu(out_12);  out_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_14 = self.getattr_L__mod___layer2___0___conv1(identity_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_15 = self.getattr_L__mod___layer2___0___bn1(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_16 = self.getattr_L__mod___layer2___0___relu(out_15);  out_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_17 = self.getattr_L__mod___layer2___0___conv2(out_16);  out_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_18 = self.getattr_L__mod___layer2___0___bn2(out_17);  out_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    getattr_l__mod___layer2___0___downsample_0 = self.getattr_L__mod___layer2___0___downsample_0(identity_2);  identity_2 = None
    identity_3 = self.getattr_L__mod___layer2___0___downsample_1(getattr_l__mod___layer2___0___downsample_0);  getattr_l__mod___layer2___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_18 += identity_3;  out_19 = out_18;  out_18 = identity_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_4 = self.getattr_L__mod___layer2___0___relu(out_19);  out_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_21 = self.getattr_L__mod___layer2___1___conv1(identity_4)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_22 = self.getattr_L__mod___layer2___1___bn1(out_21);  out_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_23 = self.getattr_L__mod___layer2___1___relu(out_22);  out_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_24 = self.getattr_L__mod___layer2___1___conv2(out_23);  out_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_25 = self.getattr_L__mod___layer2___1___bn2(out_24);  out_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_25 += identity_4;  out_26 = out_25;  out_25 = identity_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_5 = self.getattr_L__mod___layer2___1___relu(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_28 = self.getattr_L__mod___layer3___0___conv1(identity_5)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_29 = self.getattr_L__mod___layer3___0___bn1(out_28);  out_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_30 = self.getattr_L__mod___layer3___0___relu(out_29);  out_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_31 = self.getattr_L__mod___layer3___0___conv2(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_32 = self.getattr_L__mod___layer3___0___bn2(out_31);  out_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    getattr_l__mod___layer3___0___downsample_0 = self.getattr_L__mod___layer3___0___downsample_0(identity_5);  identity_5 = None
    identity_6 = self.getattr_L__mod___layer3___0___downsample_1(getattr_l__mod___layer3___0___downsample_0);  getattr_l__mod___layer3___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_32 += identity_6;  out_33 = out_32;  out_32 = identity_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_7 = self.getattr_L__mod___layer3___0___relu(out_33);  out_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_35 = self.getattr_L__mod___layer3___1___conv1(identity_7)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_36 = self.getattr_L__mod___layer3___1___bn1(out_35);  out_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_37 = self.getattr_L__mod___layer3___1___relu(out_36);  out_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_38 = self.getattr_L__mod___layer3___1___conv2(out_37);  out_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_39 = self.getattr_L__mod___layer3___1___bn2(out_38);  out_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_39 += identity_7;  out_40 = out_39;  out_39 = identity_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_8 = self.getattr_L__mod___layer3___1___relu(out_40);  out_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_42 = self.getattr_L__mod___layer4___0___conv1(identity_8)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_43 = self.getattr_L__mod___layer4___0___bn1(out_42);  out_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_44 = self.getattr_L__mod___layer4___0___relu(out_43);  out_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_45 = self.getattr_L__mod___layer4___0___conv2(out_44);  out_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_46 = self.getattr_L__mod___layer4___0___bn2(out_45);  out_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    getattr_l__mod___layer4___0___downsample_0 = self.getattr_L__mod___layer4___0___downsample_0(identity_8);  identity_8 = None
    identity_9 = self.getattr_L__mod___layer4___0___downsample_1(getattr_l__mod___layer4___0___downsample_0);  getattr_l__mod___layer4___0___downsample_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_46 += identity_9;  out_47 = out_46;  out_46 = identity_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    identity_10 = self.getattr_L__mod___layer4___0___relu(out_47);  out_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    out_49 = self.getattr_L__mod___layer4___1___conv1(identity_10)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    out_50 = self.getattr_L__mod___layer4___1___bn1(out_49);  out_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    out_51 = self.getattr_L__mod___layer4___1___relu(out_50);  out_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    out_52 = self.getattr_L__mod___layer4___1___conv2(out_51);  out_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    out_53 = self.getattr_L__mod___layer4___1___bn2(out_52);  out_52 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    out_53 += identity_10;  out_54 = out_53;  out_53 = identity_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    x_7 = self.getattr_L__mod___layer4___1___relu(out_54);  out_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    x_8 = self.L__mod___avgpool(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    x_9 = torch.flatten(x_8, 1);  x_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    pred = self.L__mod___fc(x_9);  x_9 = None
    return (pred,)
    