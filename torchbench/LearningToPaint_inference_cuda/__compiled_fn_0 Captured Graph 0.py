from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:105, code: x = F.relu(self.bn1(self.conv1(x)))
    l__mod___conv1 = self.L__mod___conv1(l_inputs_0_);  l_inputs_0_ = None
    l__mod___bn1 = self.L__mod___bn1(l__mod___conv1);  l__mod___conv1 = None
    x = torch.nn.functional.relu(l__mod___bn1);  l__mod___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer1___0___conv1 = self.getattr_L__mod___layer1___0___conv1(x)
    getattr_l__mod___layer1___0___bn1 = self.getattr_L__mod___layer1___0___bn1(getattr_l__mod___layer1___0___conv1);  getattr_l__mod___layer1___0___conv1 = None
    out = torch.nn.functional.relu(getattr_l__mod___layer1___0___bn1);  getattr_l__mod___layer1___0___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer1___0___conv2 = self.getattr_L__mod___layer1___0___conv2(out);  out = None
    out_1 = self.getattr_L__mod___layer1___0___bn2(getattr_l__mod___layer1___0___conv2);  getattr_l__mod___layer1___0___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    getattr_l__mod___layer1___0___shortcut_0 = self.getattr_L__mod___layer1___0___shortcut_0(x);  x = None
    getattr_l__mod___layer1___0___shortcut_1 = self.getattr_L__mod___layer1___0___shortcut_1(getattr_l__mod___layer1___0___shortcut_0);  getattr_l__mod___layer1___0___shortcut_0 = None
    out_1 += getattr_l__mod___layer1___0___shortcut_1;  out_2 = out_1;  out_1 = getattr_l__mod___layer1___0___shortcut_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    out_3 = torch.nn.functional.relu(out_2);  out_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer1___1___conv1 = self.getattr_L__mod___layer1___1___conv1(out_3)
    getattr_l__mod___layer1___1___bn1 = self.getattr_L__mod___layer1___1___bn1(getattr_l__mod___layer1___1___conv1);  getattr_l__mod___layer1___1___conv1 = None
    out_4 = torch.nn.functional.relu(getattr_l__mod___layer1___1___bn1);  getattr_l__mod___layer1___1___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer1___1___conv2 = self.getattr_L__mod___layer1___1___conv2(out_4);  out_4 = None
    out_5 = self.getattr_L__mod___layer1___1___bn2(getattr_l__mod___layer1___1___conv2);  getattr_l__mod___layer1___1___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    out_5 += out_3;  out_6 = out_5;  out_5 = out_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    x_1 = torch.nn.functional.relu(out_6);  out_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer2___0___conv1 = self.getattr_L__mod___layer2___0___conv1(x_1)
    getattr_l__mod___layer2___0___bn1 = self.getattr_L__mod___layer2___0___bn1(getattr_l__mod___layer2___0___conv1);  getattr_l__mod___layer2___0___conv1 = None
    out_8 = torch.nn.functional.relu(getattr_l__mod___layer2___0___bn1);  getattr_l__mod___layer2___0___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer2___0___conv2 = self.getattr_L__mod___layer2___0___conv2(out_8);  out_8 = None
    out_9 = self.getattr_L__mod___layer2___0___bn2(getattr_l__mod___layer2___0___conv2);  getattr_l__mod___layer2___0___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    getattr_l__mod___layer2___0___shortcut_0 = self.getattr_L__mod___layer2___0___shortcut_0(x_1);  x_1 = None
    getattr_l__mod___layer2___0___shortcut_1 = self.getattr_L__mod___layer2___0___shortcut_1(getattr_l__mod___layer2___0___shortcut_0);  getattr_l__mod___layer2___0___shortcut_0 = None
    out_9 += getattr_l__mod___layer2___0___shortcut_1;  out_10 = out_9;  out_9 = getattr_l__mod___layer2___0___shortcut_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    out_11 = torch.nn.functional.relu(out_10);  out_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer2___1___conv1 = self.getattr_L__mod___layer2___1___conv1(out_11)
    getattr_l__mod___layer2___1___bn1 = self.getattr_L__mod___layer2___1___bn1(getattr_l__mod___layer2___1___conv1);  getattr_l__mod___layer2___1___conv1 = None
    out_12 = torch.nn.functional.relu(getattr_l__mod___layer2___1___bn1);  getattr_l__mod___layer2___1___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer2___1___conv2 = self.getattr_L__mod___layer2___1___conv2(out_12);  out_12 = None
    out_13 = self.getattr_L__mod___layer2___1___bn2(getattr_l__mod___layer2___1___conv2);  getattr_l__mod___layer2___1___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    out_13 += out_11;  out_14 = out_13;  out_13 = out_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    x_2 = torch.nn.functional.relu(out_14);  out_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer3___0___conv1 = self.getattr_L__mod___layer3___0___conv1(x_2)
    getattr_l__mod___layer3___0___bn1 = self.getattr_L__mod___layer3___0___bn1(getattr_l__mod___layer3___0___conv1);  getattr_l__mod___layer3___0___conv1 = None
    out_16 = torch.nn.functional.relu(getattr_l__mod___layer3___0___bn1);  getattr_l__mod___layer3___0___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer3___0___conv2 = self.getattr_L__mod___layer3___0___conv2(out_16);  out_16 = None
    out_17 = self.getattr_L__mod___layer3___0___bn2(getattr_l__mod___layer3___0___conv2);  getattr_l__mod___layer3___0___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    getattr_l__mod___layer3___0___shortcut_0 = self.getattr_L__mod___layer3___0___shortcut_0(x_2);  x_2 = None
    getattr_l__mod___layer3___0___shortcut_1 = self.getattr_L__mod___layer3___0___shortcut_1(getattr_l__mod___layer3___0___shortcut_0);  getattr_l__mod___layer3___0___shortcut_0 = None
    out_17 += getattr_l__mod___layer3___0___shortcut_1;  out_18 = out_17;  out_17 = getattr_l__mod___layer3___0___shortcut_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    out_19 = torch.nn.functional.relu(out_18);  out_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer3___1___conv1 = self.getattr_L__mod___layer3___1___conv1(out_19)
    getattr_l__mod___layer3___1___bn1 = self.getattr_L__mod___layer3___1___bn1(getattr_l__mod___layer3___1___conv1);  getattr_l__mod___layer3___1___conv1 = None
    out_20 = torch.nn.functional.relu(getattr_l__mod___layer3___1___bn1);  getattr_l__mod___layer3___1___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer3___1___conv2 = self.getattr_L__mod___layer3___1___conv2(out_20);  out_20 = None
    out_21 = self.getattr_L__mod___layer3___1___bn2(getattr_l__mod___layer3___1___conv2);  getattr_l__mod___layer3___1___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    out_21 += out_19;  out_22 = out_21;  out_21 = out_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    x_3 = torch.nn.functional.relu(out_22);  out_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer4___0___conv1 = self.getattr_L__mod___layer4___0___conv1(x_3)
    getattr_l__mod___layer4___0___bn1 = self.getattr_L__mod___layer4___0___bn1(getattr_l__mod___layer4___0___conv1);  getattr_l__mod___layer4___0___conv1 = None
    out_24 = torch.nn.functional.relu(getattr_l__mod___layer4___0___bn1);  getattr_l__mod___layer4___0___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer4___0___conv2 = self.getattr_L__mod___layer4___0___conv2(out_24);  out_24 = None
    out_25 = self.getattr_L__mod___layer4___0___bn2(getattr_l__mod___layer4___0___conv2);  getattr_l__mod___layer4___0___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    getattr_l__mod___layer4___0___shortcut_0 = self.getattr_L__mod___layer4___0___shortcut_0(x_3);  x_3 = None
    getattr_l__mod___layer4___0___shortcut_1 = self.getattr_L__mod___layer4___0___shortcut_1(getattr_l__mod___layer4___0___shortcut_0);  getattr_l__mod___layer4___0___shortcut_0 = None
    out_25 += getattr_l__mod___layer4___0___shortcut_1;  out_26 = out_25;  out_25 = getattr_l__mod___layer4___0___shortcut_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    out_27 = torch.nn.functional.relu(out_26);  out_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    getattr_l__mod___layer4___1___conv1 = self.getattr_L__mod___layer4___1___conv1(out_27)
    getattr_l__mod___layer4___1___bn1 = self.getattr_L__mod___layer4___1___bn1(getattr_l__mod___layer4___1___conv1);  getattr_l__mod___layer4___1___conv1 = None
    out_28 = torch.nn.functional.relu(getattr_l__mod___layer4___1___bn1);  getattr_l__mod___layer4___1___bn1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    getattr_l__mod___layer4___1___conv2 = self.getattr_L__mod___layer4___1___conv2(out_28);  out_28 = None
    out_29 = self.getattr_L__mod___layer4___1___bn2(getattr_l__mod___layer4___1___conv2);  getattr_l__mod___layer4___1___conv2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    out_29 += out_27;  out_30 = out_29;  out_29 = out_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    x_4 = torch.nn.functional.relu(out_30);  out_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:110, code: x = F.avg_pool2d(x, 4)
    x_5 = torch._C._nn.avg_pool2d(x_4, 4);  x_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:111, code: x = x.view(x.size(0), -1)
    x_6 = x_5.view(4, -1);  x_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:112, code: x = self.fc(x)
    x_7 = self.L__mod___fc(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:113, code: x = torch.sigmoid(x)
    x_8 = torch.sigmoid(x_7);  x_7 = None
    return (x_8,)
    