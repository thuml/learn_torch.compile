from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    l__mod___features_0_0 = self.L__mod___features_0_0(l_inputs_0_);  l_inputs_0_ = None
    l__mod___features_0_1 = self.L__mod___features_0_1(l__mod___features_0_0);  l__mod___features_0_0 = None
    l__mod___features_0_2 = self.L__mod___features_0_2(l__mod___features_0_1);  l__mod___features_0_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    getattr_l__mod___features___1___conv_0_0 = self.getattr_L__mod___features___1___conv_0_0(l__mod___features_0_2);  l__mod___features_0_2 = None
    getattr_l__mod___features___1___conv_0_1 = self.getattr_L__mod___features___1___conv_0_1(getattr_l__mod___features___1___conv_0_0);  getattr_l__mod___features___1___conv_0_0 = None
    getattr_l__mod___features___1___conv_0_2 = self.getattr_L__mod___features___1___conv_0_2(getattr_l__mod___features___1___conv_0_1);  getattr_l__mod___features___1___conv_0_1 = None
    getattr_l__mod___features___1___conv_1 = self.getattr_L__mod___features___1___conv_1(getattr_l__mod___features___1___conv_0_2);  getattr_l__mod___features___1___conv_0_2 = None
    getattr_l__mod___features___1___conv_2 = self.getattr_L__mod___features___1___conv_2(getattr_l__mod___features___1___conv_1);  getattr_l__mod___features___1___conv_1 = None
    getattr_l__mod___features___2___conv_0_0 = self.getattr_L__mod___features___2___conv_0_0(getattr_l__mod___features___1___conv_2);  getattr_l__mod___features___1___conv_2 = None
    getattr_l__mod___features___2___conv_0_1 = self.getattr_L__mod___features___2___conv_0_1(getattr_l__mod___features___2___conv_0_0);  getattr_l__mod___features___2___conv_0_0 = None
    getattr_l__mod___features___2___conv_0_2 = self.getattr_L__mod___features___2___conv_0_2(getattr_l__mod___features___2___conv_0_1);  getattr_l__mod___features___2___conv_0_1 = None
    getattr_l__mod___features___2___conv_1_0 = self.getattr_L__mod___features___2___conv_1_0(getattr_l__mod___features___2___conv_0_2);  getattr_l__mod___features___2___conv_0_2 = None
    getattr_l__mod___features___2___conv_1_1 = self.getattr_L__mod___features___2___conv_1_1(getattr_l__mod___features___2___conv_1_0);  getattr_l__mod___features___2___conv_1_0 = None
    getattr_l__mod___features___2___conv_1_2 = self.getattr_L__mod___features___2___conv_1_2(getattr_l__mod___features___2___conv_1_1);  getattr_l__mod___features___2___conv_1_1 = None
    getattr_l__mod___features___2___conv_2 = self.getattr_L__mod___features___2___conv_2(getattr_l__mod___features___2___conv_1_2);  getattr_l__mod___features___2___conv_1_2 = None
    getattr_l__mod___features___2___conv_3 = self.getattr_L__mod___features___2___conv_3(getattr_l__mod___features___2___conv_2);  getattr_l__mod___features___2___conv_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    getattr_l__mod___features___3___conv_0_0 = self.getattr_L__mod___features___3___conv_0_0(getattr_l__mod___features___2___conv_3)
    getattr_l__mod___features___3___conv_0_1 = self.getattr_L__mod___features___3___conv_0_1(getattr_l__mod___features___3___conv_0_0);  getattr_l__mod___features___3___conv_0_0 = None
    getattr_l__mod___features___3___conv_0_2 = self.getattr_L__mod___features___3___conv_0_2(getattr_l__mod___features___3___conv_0_1);  getattr_l__mod___features___3___conv_0_1 = None
    getattr_l__mod___features___3___conv_1_0 = self.getattr_L__mod___features___3___conv_1_0(getattr_l__mod___features___3___conv_0_2);  getattr_l__mod___features___3___conv_0_2 = None
    getattr_l__mod___features___3___conv_1_1 = self.getattr_L__mod___features___3___conv_1_1(getattr_l__mod___features___3___conv_1_0);  getattr_l__mod___features___3___conv_1_0 = None
    getattr_l__mod___features___3___conv_1_2 = self.getattr_L__mod___features___3___conv_1_2(getattr_l__mod___features___3___conv_1_1);  getattr_l__mod___features___3___conv_1_1 = None
    getattr_l__mod___features___3___conv_2 = self.getattr_L__mod___features___3___conv_2(getattr_l__mod___features___3___conv_1_2);  getattr_l__mod___features___3___conv_1_2 = None
    getattr_l__mod___features___3___conv_3 = self.getattr_L__mod___features___3___conv_3(getattr_l__mod___features___3___conv_2);  getattr_l__mod___features___3___conv_2 = None
    add = getattr_l__mod___features___2___conv_3 + getattr_l__mod___features___3___conv_3;  getattr_l__mod___features___2___conv_3 = getattr_l__mod___features___3___conv_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    getattr_l__mod___features___4___conv_0_0 = self.getattr_L__mod___features___4___conv_0_0(add);  add = None
    getattr_l__mod___features___4___conv_0_1 = self.getattr_L__mod___features___4___conv_0_1(getattr_l__mod___features___4___conv_0_0);  getattr_l__mod___features___4___conv_0_0 = None
    getattr_l__mod___features___4___conv_0_2 = self.getattr_L__mod___features___4___conv_0_2(getattr_l__mod___features___4___conv_0_1);  getattr_l__mod___features___4___conv_0_1 = None
    getattr_l__mod___features___4___conv_1_0 = self.getattr_L__mod___features___4___conv_1_0(getattr_l__mod___features___4___conv_0_2);  getattr_l__mod___features___4___conv_0_2 = None
    getattr_l__mod___features___4___conv_1_1 = self.getattr_L__mod___features___4___conv_1_1(getattr_l__mod___features___4___conv_1_0);  getattr_l__mod___features___4___conv_1_0 = None
    getattr_l__mod___features___4___conv_1_2 = self.getattr_L__mod___features___4___conv_1_2(getattr_l__mod___features___4___conv_1_1);  getattr_l__mod___features___4___conv_1_1 = None
    getattr_l__mod___features___4___conv_2 = self.getattr_L__mod___features___4___conv_2(getattr_l__mod___features___4___conv_1_2);  getattr_l__mod___features___4___conv_1_2 = None
    getattr_l__mod___features___4___conv_3 = self.getattr_L__mod___features___4___conv_3(getattr_l__mod___features___4___conv_2);  getattr_l__mod___features___4___conv_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    getattr_l__mod___features___5___conv_0_0 = self.getattr_L__mod___features___5___conv_0_0(getattr_l__mod___features___4___conv_3)
    getattr_l__mod___features___5___conv_0_1 = self.getattr_L__mod___features___5___conv_0_1(getattr_l__mod___features___5___conv_0_0);  getattr_l__mod___features___5___conv_0_0 = None
    getattr_l__mod___features___5___conv_0_2 = self.getattr_L__mod___features___5___conv_0_2(getattr_l__mod___features___5___conv_0_1);  getattr_l__mod___features___5___conv_0_1 = None
    getattr_l__mod___features___5___conv_1_0 = self.getattr_L__mod___features___5___conv_1_0(getattr_l__mod___features___5___conv_0_2);  getattr_l__mod___features___5___conv_0_2 = None
    getattr_l__mod___features___5___conv_1_1 = self.getattr_L__mod___features___5___conv_1_1(getattr_l__mod___features___5___conv_1_0);  getattr_l__mod___features___5___conv_1_0 = None
    getattr_l__mod___features___5___conv_1_2 = self.getattr_L__mod___features___5___conv_1_2(getattr_l__mod___features___5___conv_1_1);  getattr_l__mod___features___5___conv_1_1 = None
    getattr_l__mod___features___5___conv_2 = self.getattr_L__mod___features___5___conv_2(getattr_l__mod___features___5___conv_1_2);  getattr_l__mod___features___5___conv_1_2 = None
    getattr_l__mod___features___5___conv_3 = self.getattr_L__mod___features___5___conv_3(getattr_l__mod___features___5___conv_2);  getattr_l__mod___features___5___conv_2 = None
    add_1 = getattr_l__mod___features___4___conv_3 + getattr_l__mod___features___5___conv_3;  getattr_l__mod___features___4___conv_3 = getattr_l__mod___features___5___conv_3 = None
    getattr_l__mod___features___6___conv_0_0 = self.getattr_L__mod___features___6___conv_0_0(add_1)
    getattr_l__mod___features___6___conv_0_1 = self.getattr_L__mod___features___6___conv_0_1(getattr_l__mod___features___6___conv_0_0);  getattr_l__mod___features___6___conv_0_0 = None
    getattr_l__mod___features___6___conv_0_2 = self.getattr_L__mod___features___6___conv_0_2(getattr_l__mod___features___6___conv_0_1);  getattr_l__mod___features___6___conv_0_1 = None
    getattr_l__mod___features___6___conv_1_0 = self.getattr_L__mod___features___6___conv_1_0(getattr_l__mod___features___6___conv_0_2);  getattr_l__mod___features___6___conv_0_2 = None
    getattr_l__mod___features___6___conv_1_1 = self.getattr_L__mod___features___6___conv_1_1(getattr_l__mod___features___6___conv_1_0);  getattr_l__mod___features___6___conv_1_0 = None
    getattr_l__mod___features___6___conv_1_2 = self.getattr_L__mod___features___6___conv_1_2(getattr_l__mod___features___6___conv_1_1);  getattr_l__mod___features___6___conv_1_1 = None
    getattr_l__mod___features___6___conv_2 = self.getattr_L__mod___features___6___conv_2(getattr_l__mod___features___6___conv_1_2);  getattr_l__mod___features___6___conv_1_2 = None
    getattr_l__mod___features___6___conv_3 = self.getattr_L__mod___features___6___conv_3(getattr_l__mod___features___6___conv_2);  getattr_l__mod___features___6___conv_2 = None
    add_2 = add_1 + getattr_l__mod___features___6___conv_3;  add_1 = getattr_l__mod___features___6___conv_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    getattr_l__mod___features___7___conv_0_0 = self.getattr_L__mod___features___7___conv_0_0(add_2);  add_2 = None
    getattr_l__mod___features___7___conv_0_1 = self.getattr_L__mod___features___7___conv_0_1(getattr_l__mod___features___7___conv_0_0);  getattr_l__mod___features___7___conv_0_0 = None
    getattr_l__mod___features___7___conv_0_2 = self.getattr_L__mod___features___7___conv_0_2(getattr_l__mod___features___7___conv_0_1);  getattr_l__mod___features___7___conv_0_1 = None
    getattr_l__mod___features___7___conv_1_0 = self.getattr_L__mod___features___7___conv_1_0(getattr_l__mod___features___7___conv_0_2);  getattr_l__mod___features___7___conv_0_2 = None
    getattr_l__mod___features___7___conv_1_1 = self.getattr_L__mod___features___7___conv_1_1(getattr_l__mod___features___7___conv_1_0);  getattr_l__mod___features___7___conv_1_0 = None
    getattr_l__mod___features___7___conv_1_2 = self.getattr_L__mod___features___7___conv_1_2(getattr_l__mod___features___7___conv_1_1);  getattr_l__mod___features___7___conv_1_1 = None
    getattr_l__mod___features___7___conv_2 = self.getattr_L__mod___features___7___conv_2(getattr_l__mod___features___7___conv_1_2);  getattr_l__mod___features___7___conv_1_2 = None
    getattr_l__mod___features___7___conv_3 = self.getattr_L__mod___features___7___conv_3(getattr_l__mod___features___7___conv_2);  getattr_l__mod___features___7___conv_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    getattr_l__mod___features___8___conv_0_0 = self.getattr_L__mod___features___8___conv_0_0(getattr_l__mod___features___7___conv_3)
    getattr_l__mod___features___8___conv_0_1 = self.getattr_L__mod___features___8___conv_0_1(getattr_l__mod___features___8___conv_0_0);  getattr_l__mod___features___8___conv_0_0 = None
    getattr_l__mod___features___8___conv_0_2 = self.getattr_L__mod___features___8___conv_0_2(getattr_l__mod___features___8___conv_0_1);  getattr_l__mod___features___8___conv_0_1 = None
    getattr_l__mod___features___8___conv_1_0 = self.getattr_L__mod___features___8___conv_1_0(getattr_l__mod___features___8___conv_0_2);  getattr_l__mod___features___8___conv_0_2 = None
    getattr_l__mod___features___8___conv_1_1 = self.getattr_L__mod___features___8___conv_1_1(getattr_l__mod___features___8___conv_1_0);  getattr_l__mod___features___8___conv_1_0 = None
    getattr_l__mod___features___8___conv_1_2 = self.getattr_L__mod___features___8___conv_1_2(getattr_l__mod___features___8___conv_1_1);  getattr_l__mod___features___8___conv_1_1 = None
    getattr_l__mod___features___8___conv_2 = self.getattr_L__mod___features___8___conv_2(getattr_l__mod___features___8___conv_1_2);  getattr_l__mod___features___8___conv_1_2 = None
    getattr_l__mod___features___8___conv_3 = self.getattr_L__mod___features___8___conv_3(getattr_l__mod___features___8___conv_2);  getattr_l__mod___features___8___conv_2 = None
    add_3 = getattr_l__mod___features___7___conv_3 + getattr_l__mod___features___8___conv_3;  getattr_l__mod___features___7___conv_3 = getattr_l__mod___features___8___conv_3 = None
    getattr_l__mod___features___9___conv_0_0 = self.getattr_L__mod___features___9___conv_0_0(add_3)
    getattr_l__mod___features___9___conv_0_1 = self.getattr_L__mod___features___9___conv_0_1(getattr_l__mod___features___9___conv_0_0);  getattr_l__mod___features___9___conv_0_0 = None
    getattr_l__mod___features___9___conv_0_2 = self.getattr_L__mod___features___9___conv_0_2(getattr_l__mod___features___9___conv_0_1);  getattr_l__mod___features___9___conv_0_1 = None
    getattr_l__mod___features___9___conv_1_0 = self.getattr_L__mod___features___9___conv_1_0(getattr_l__mod___features___9___conv_0_2);  getattr_l__mod___features___9___conv_0_2 = None
    getattr_l__mod___features___9___conv_1_1 = self.getattr_L__mod___features___9___conv_1_1(getattr_l__mod___features___9___conv_1_0);  getattr_l__mod___features___9___conv_1_0 = None
    getattr_l__mod___features___9___conv_1_2 = self.getattr_L__mod___features___9___conv_1_2(getattr_l__mod___features___9___conv_1_1);  getattr_l__mod___features___9___conv_1_1 = None
    getattr_l__mod___features___9___conv_2 = self.getattr_L__mod___features___9___conv_2(getattr_l__mod___features___9___conv_1_2);  getattr_l__mod___features___9___conv_1_2 = None
    getattr_l__mod___features___9___conv_3 = self.getattr_L__mod___features___9___conv_3(getattr_l__mod___features___9___conv_2);  getattr_l__mod___features___9___conv_2 = None
    add_4 = add_3 + getattr_l__mod___features___9___conv_3;  add_3 = getattr_l__mod___features___9___conv_3 = None
    getattr_l__mod___features___10___conv_0_0 = self.getattr_L__mod___features___10___conv_0_0(add_4)
    getattr_l__mod___features___10___conv_0_1 = self.getattr_L__mod___features___10___conv_0_1(getattr_l__mod___features___10___conv_0_0);  getattr_l__mod___features___10___conv_0_0 = None
    getattr_l__mod___features___10___conv_0_2 = self.getattr_L__mod___features___10___conv_0_2(getattr_l__mod___features___10___conv_0_1);  getattr_l__mod___features___10___conv_0_1 = None
    getattr_l__mod___features___10___conv_1_0 = self.getattr_L__mod___features___10___conv_1_0(getattr_l__mod___features___10___conv_0_2);  getattr_l__mod___features___10___conv_0_2 = None
    getattr_l__mod___features___10___conv_1_1 = self.getattr_L__mod___features___10___conv_1_1(getattr_l__mod___features___10___conv_1_0);  getattr_l__mod___features___10___conv_1_0 = None
    getattr_l__mod___features___10___conv_1_2 = self.getattr_L__mod___features___10___conv_1_2(getattr_l__mod___features___10___conv_1_1);  getattr_l__mod___features___10___conv_1_1 = None
    getattr_l__mod___features___10___conv_2 = self.getattr_L__mod___features___10___conv_2(getattr_l__mod___features___10___conv_1_2);  getattr_l__mod___features___10___conv_1_2 = None
    getattr_l__mod___features___10___conv_3 = self.getattr_L__mod___features___10___conv_3(getattr_l__mod___features___10___conv_2);  getattr_l__mod___features___10___conv_2 = None
    add_5 = add_4 + getattr_l__mod___features___10___conv_3;  add_4 = getattr_l__mod___features___10___conv_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    getattr_l__mod___features___11___conv_0_0 = self.getattr_L__mod___features___11___conv_0_0(add_5);  add_5 = None
    getattr_l__mod___features___11___conv_0_1 = self.getattr_L__mod___features___11___conv_0_1(getattr_l__mod___features___11___conv_0_0);  getattr_l__mod___features___11___conv_0_0 = None
    getattr_l__mod___features___11___conv_0_2 = self.getattr_L__mod___features___11___conv_0_2(getattr_l__mod___features___11___conv_0_1);  getattr_l__mod___features___11___conv_0_1 = None
    getattr_l__mod___features___11___conv_1_0 = self.getattr_L__mod___features___11___conv_1_0(getattr_l__mod___features___11___conv_0_2);  getattr_l__mod___features___11___conv_0_2 = None
    getattr_l__mod___features___11___conv_1_1 = self.getattr_L__mod___features___11___conv_1_1(getattr_l__mod___features___11___conv_1_0);  getattr_l__mod___features___11___conv_1_0 = None
    getattr_l__mod___features___11___conv_1_2 = self.getattr_L__mod___features___11___conv_1_2(getattr_l__mod___features___11___conv_1_1);  getattr_l__mod___features___11___conv_1_1 = None
    getattr_l__mod___features___11___conv_2 = self.getattr_L__mod___features___11___conv_2(getattr_l__mod___features___11___conv_1_2);  getattr_l__mod___features___11___conv_1_2 = None
    getattr_l__mod___features___11___conv_3 = self.getattr_L__mod___features___11___conv_3(getattr_l__mod___features___11___conv_2);  getattr_l__mod___features___11___conv_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    getattr_l__mod___features___12___conv_0_0 = self.getattr_L__mod___features___12___conv_0_0(getattr_l__mod___features___11___conv_3)
    getattr_l__mod___features___12___conv_0_1 = self.getattr_L__mod___features___12___conv_0_1(getattr_l__mod___features___12___conv_0_0);  getattr_l__mod___features___12___conv_0_0 = None
    getattr_l__mod___features___12___conv_0_2 = self.getattr_L__mod___features___12___conv_0_2(getattr_l__mod___features___12___conv_0_1);  getattr_l__mod___features___12___conv_0_1 = None
    getattr_l__mod___features___12___conv_1_0 = self.getattr_L__mod___features___12___conv_1_0(getattr_l__mod___features___12___conv_0_2);  getattr_l__mod___features___12___conv_0_2 = None
    getattr_l__mod___features___12___conv_1_1 = self.getattr_L__mod___features___12___conv_1_1(getattr_l__mod___features___12___conv_1_0);  getattr_l__mod___features___12___conv_1_0 = None
    getattr_l__mod___features___12___conv_1_2 = self.getattr_L__mod___features___12___conv_1_2(getattr_l__mod___features___12___conv_1_1);  getattr_l__mod___features___12___conv_1_1 = None
    getattr_l__mod___features___12___conv_2 = self.getattr_L__mod___features___12___conv_2(getattr_l__mod___features___12___conv_1_2);  getattr_l__mod___features___12___conv_1_2 = None
    getattr_l__mod___features___12___conv_3 = self.getattr_L__mod___features___12___conv_3(getattr_l__mod___features___12___conv_2);  getattr_l__mod___features___12___conv_2 = None
    add_6 = getattr_l__mod___features___11___conv_3 + getattr_l__mod___features___12___conv_3;  getattr_l__mod___features___11___conv_3 = getattr_l__mod___features___12___conv_3 = None
    getattr_l__mod___features___13___conv_0_0 = self.getattr_L__mod___features___13___conv_0_0(add_6)
    getattr_l__mod___features___13___conv_0_1 = self.getattr_L__mod___features___13___conv_0_1(getattr_l__mod___features___13___conv_0_0);  getattr_l__mod___features___13___conv_0_0 = None
    getattr_l__mod___features___13___conv_0_2 = self.getattr_L__mod___features___13___conv_0_2(getattr_l__mod___features___13___conv_0_1);  getattr_l__mod___features___13___conv_0_1 = None
    getattr_l__mod___features___13___conv_1_0 = self.getattr_L__mod___features___13___conv_1_0(getattr_l__mod___features___13___conv_0_2);  getattr_l__mod___features___13___conv_0_2 = None
    getattr_l__mod___features___13___conv_1_1 = self.getattr_L__mod___features___13___conv_1_1(getattr_l__mod___features___13___conv_1_0);  getattr_l__mod___features___13___conv_1_0 = None
    getattr_l__mod___features___13___conv_1_2 = self.getattr_L__mod___features___13___conv_1_2(getattr_l__mod___features___13___conv_1_1);  getattr_l__mod___features___13___conv_1_1 = None
    getattr_l__mod___features___13___conv_2 = self.getattr_L__mod___features___13___conv_2(getattr_l__mod___features___13___conv_1_2);  getattr_l__mod___features___13___conv_1_2 = None
    getattr_l__mod___features___13___conv_3 = self.getattr_L__mod___features___13___conv_3(getattr_l__mod___features___13___conv_2);  getattr_l__mod___features___13___conv_2 = None
    add_7 = add_6 + getattr_l__mod___features___13___conv_3;  add_6 = getattr_l__mod___features___13___conv_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    getattr_l__mod___features___14___conv_0_0 = self.getattr_L__mod___features___14___conv_0_0(add_7);  add_7 = None
    getattr_l__mod___features___14___conv_0_1 = self.getattr_L__mod___features___14___conv_0_1(getattr_l__mod___features___14___conv_0_0);  getattr_l__mod___features___14___conv_0_0 = None
    getattr_l__mod___features___14___conv_0_2 = self.getattr_L__mod___features___14___conv_0_2(getattr_l__mod___features___14___conv_0_1);  getattr_l__mod___features___14___conv_0_1 = None
    getattr_l__mod___features___14___conv_1_0 = self.getattr_L__mod___features___14___conv_1_0(getattr_l__mod___features___14___conv_0_2);  getattr_l__mod___features___14___conv_0_2 = None
    getattr_l__mod___features___14___conv_1_1 = self.getattr_L__mod___features___14___conv_1_1(getattr_l__mod___features___14___conv_1_0);  getattr_l__mod___features___14___conv_1_0 = None
    getattr_l__mod___features___14___conv_1_2 = self.getattr_L__mod___features___14___conv_1_2(getattr_l__mod___features___14___conv_1_1);  getattr_l__mod___features___14___conv_1_1 = None
    getattr_l__mod___features___14___conv_2 = self.getattr_L__mod___features___14___conv_2(getattr_l__mod___features___14___conv_1_2);  getattr_l__mod___features___14___conv_1_2 = None
    getattr_l__mod___features___14___conv_3 = self.getattr_L__mod___features___14___conv_3(getattr_l__mod___features___14___conv_2);  getattr_l__mod___features___14___conv_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:62, code: return x + self.conv(x)
    getattr_l__mod___features___15___conv_0_0 = self.getattr_L__mod___features___15___conv_0_0(getattr_l__mod___features___14___conv_3)
    getattr_l__mod___features___15___conv_0_1 = self.getattr_L__mod___features___15___conv_0_1(getattr_l__mod___features___15___conv_0_0);  getattr_l__mod___features___15___conv_0_0 = None
    getattr_l__mod___features___15___conv_0_2 = self.getattr_L__mod___features___15___conv_0_2(getattr_l__mod___features___15___conv_0_1);  getattr_l__mod___features___15___conv_0_1 = None
    getattr_l__mod___features___15___conv_1_0 = self.getattr_L__mod___features___15___conv_1_0(getattr_l__mod___features___15___conv_0_2);  getattr_l__mod___features___15___conv_0_2 = None
    getattr_l__mod___features___15___conv_1_1 = self.getattr_L__mod___features___15___conv_1_1(getattr_l__mod___features___15___conv_1_0);  getattr_l__mod___features___15___conv_1_0 = None
    getattr_l__mod___features___15___conv_1_2 = self.getattr_L__mod___features___15___conv_1_2(getattr_l__mod___features___15___conv_1_1);  getattr_l__mod___features___15___conv_1_1 = None
    getattr_l__mod___features___15___conv_2 = self.getattr_L__mod___features___15___conv_2(getattr_l__mod___features___15___conv_1_2);  getattr_l__mod___features___15___conv_1_2 = None
    getattr_l__mod___features___15___conv_3 = self.getattr_L__mod___features___15___conv_3(getattr_l__mod___features___15___conv_2);  getattr_l__mod___features___15___conv_2 = None
    add_8 = getattr_l__mod___features___14___conv_3 + getattr_l__mod___features___15___conv_3;  getattr_l__mod___features___14___conv_3 = getattr_l__mod___features___15___conv_3 = None
    getattr_l__mod___features___16___conv_0_0 = self.getattr_L__mod___features___16___conv_0_0(add_8)
    getattr_l__mod___features___16___conv_0_1 = self.getattr_L__mod___features___16___conv_0_1(getattr_l__mod___features___16___conv_0_0);  getattr_l__mod___features___16___conv_0_0 = None
    getattr_l__mod___features___16___conv_0_2 = self.getattr_L__mod___features___16___conv_0_2(getattr_l__mod___features___16___conv_0_1);  getattr_l__mod___features___16___conv_0_1 = None
    getattr_l__mod___features___16___conv_1_0 = self.getattr_L__mod___features___16___conv_1_0(getattr_l__mod___features___16___conv_0_2);  getattr_l__mod___features___16___conv_0_2 = None
    getattr_l__mod___features___16___conv_1_1 = self.getattr_L__mod___features___16___conv_1_1(getattr_l__mod___features___16___conv_1_0);  getattr_l__mod___features___16___conv_1_0 = None
    getattr_l__mod___features___16___conv_1_2 = self.getattr_L__mod___features___16___conv_1_2(getattr_l__mod___features___16___conv_1_1);  getattr_l__mod___features___16___conv_1_1 = None
    getattr_l__mod___features___16___conv_2 = self.getattr_L__mod___features___16___conv_2(getattr_l__mod___features___16___conv_1_2);  getattr_l__mod___features___16___conv_1_2 = None
    getattr_l__mod___features___16___conv_3 = self.getattr_L__mod___features___16___conv_3(getattr_l__mod___features___16___conv_2);  getattr_l__mod___features___16___conv_2 = None
    add_9 = add_8 + getattr_l__mod___features___16___conv_3;  add_8 = getattr_l__mod___features___16___conv_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:64, code: return self.conv(x)
    getattr_l__mod___features___17___conv_0_0 = self.getattr_L__mod___features___17___conv_0_0(add_9);  add_9 = None
    getattr_l__mod___features___17___conv_0_1 = self.getattr_L__mod___features___17___conv_0_1(getattr_l__mod___features___17___conv_0_0);  getattr_l__mod___features___17___conv_0_0 = None
    getattr_l__mod___features___17___conv_0_2 = self.getattr_L__mod___features___17___conv_0_2(getattr_l__mod___features___17___conv_0_1);  getattr_l__mod___features___17___conv_0_1 = None
    getattr_l__mod___features___17___conv_1_0 = self.getattr_L__mod___features___17___conv_1_0(getattr_l__mod___features___17___conv_0_2);  getattr_l__mod___features___17___conv_0_2 = None
    getattr_l__mod___features___17___conv_1_1 = self.getattr_L__mod___features___17___conv_1_1(getattr_l__mod___features___17___conv_1_0);  getattr_l__mod___features___17___conv_1_0 = None
    getattr_l__mod___features___17___conv_1_2 = self.getattr_L__mod___features___17___conv_1_2(getattr_l__mod___features___17___conv_1_1);  getattr_l__mod___features___17___conv_1_1 = None
    getattr_l__mod___features___17___conv_2 = self.getattr_L__mod___features___17___conv_2(getattr_l__mod___features___17___conv_1_2);  getattr_l__mod___features___17___conv_1_2 = None
    getattr_l__mod___features___17___conv_3 = self.getattr_L__mod___features___17___conv_3(getattr_l__mod___features___17___conv_2);  getattr_l__mod___features___17___conv_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:166, code: x = self.features(x)
    l__mod___features_18_0 = self.L__mod___features_18_0(getattr_l__mod___features___17___conv_3);  getattr_l__mod___features___17___conv_3 = None
    l__mod___features_18_1 = self.L__mod___features_18_1(l__mod___features_18_0);  l__mod___features_18_0 = None
    x = self.L__mod___features_18_2(l__mod___features_18_1);  l__mod___features_18_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:168, code: x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x_1 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1));  x = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:169, code: x = torch.flatten(x, 1)
    x_2 = torch.flatten(x_1, 1);  x_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv2.py:170, code: x = self.classifier(x)
    l__mod___classifier_0 = self.L__mod___classifier_0(x_2);  x_2 = None
    x_3 = self.L__mod___classifier_1(l__mod___classifier_0);  l__mod___classifier_0 = None
    return (x_3,)
    