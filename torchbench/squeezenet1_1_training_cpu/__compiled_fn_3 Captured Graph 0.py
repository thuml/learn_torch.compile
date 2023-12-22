from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    l__mod___features_0 = self.L__mod___features_0(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    l__mod___features_1 = self.L__mod___features_1(l__mod___features_0);  l__mod___features_0 = None
    l__mod___features_2 = self.L__mod___features_2(l__mod___features_1);  l__mod___features_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___3___squeeze = self.getattr_L__mod___features___3___squeeze(l__mod___features_2);  l__mod___features_2 = None
    x = self.getattr_L__mod___features___3___squeeze_activation(getattr_l__mod___features___3___squeeze);  getattr_l__mod___features___3___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___3___expand1x1 = self.getattr_L__mod___features___3___expand1x1(x)
    getattr_l__mod___features___3___expand1x1_activation = self.getattr_L__mod___features___3___expand1x1_activation(getattr_l__mod___features___3___expand1x1);  getattr_l__mod___features___3___expand1x1 = None
    getattr_l__mod___features___3___expand3x3 = self.getattr_L__mod___features___3___expand3x3(x);  x = None
    getattr_l__mod___features___3___expand3x3_activation = self.getattr_L__mod___features___3___expand3x3_activation(getattr_l__mod___features___3___expand3x3);  getattr_l__mod___features___3___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat = torch.cat([getattr_l__mod___features___3___expand1x1_activation, getattr_l__mod___features___3___expand3x3_activation], 1);  getattr_l__mod___features___3___expand1x1_activation = getattr_l__mod___features___3___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___4___squeeze = self.getattr_L__mod___features___4___squeeze(cat);  cat = None
    x_1 = self.getattr_L__mod___features___4___squeeze_activation(getattr_l__mod___features___4___squeeze);  getattr_l__mod___features___4___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___4___expand1x1 = self.getattr_L__mod___features___4___expand1x1(x_1)
    getattr_l__mod___features___4___expand1x1_activation = self.getattr_L__mod___features___4___expand1x1_activation(getattr_l__mod___features___4___expand1x1);  getattr_l__mod___features___4___expand1x1 = None
    getattr_l__mod___features___4___expand3x3 = self.getattr_L__mod___features___4___expand3x3(x_1);  x_1 = None
    getattr_l__mod___features___4___expand3x3_activation = self.getattr_L__mod___features___4___expand3x3_activation(getattr_l__mod___features___4___expand3x3);  getattr_l__mod___features___4___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_1 = torch.cat([getattr_l__mod___features___4___expand1x1_activation, getattr_l__mod___features___4___expand3x3_activation], 1);  getattr_l__mod___features___4___expand1x1_activation = getattr_l__mod___features___4___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    l__mod___features_5 = self.L__mod___features_5(cat_1);  cat_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___6___squeeze = self.getattr_L__mod___features___6___squeeze(l__mod___features_5);  l__mod___features_5 = None
    x_2 = self.getattr_L__mod___features___6___squeeze_activation(getattr_l__mod___features___6___squeeze);  getattr_l__mod___features___6___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___6___expand1x1 = self.getattr_L__mod___features___6___expand1x1(x_2)
    getattr_l__mod___features___6___expand1x1_activation = self.getattr_L__mod___features___6___expand1x1_activation(getattr_l__mod___features___6___expand1x1);  getattr_l__mod___features___6___expand1x1 = None
    getattr_l__mod___features___6___expand3x3 = self.getattr_L__mod___features___6___expand3x3(x_2);  x_2 = None
    getattr_l__mod___features___6___expand3x3_activation = self.getattr_L__mod___features___6___expand3x3_activation(getattr_l__mod___features___6___expand3x3);  getattr_l__mod___features___6___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_2 = torch.cat([getattr_l__mod___features___6___expand1x1_activation, getattr_l__mod___features___6___expand3x3_activation], 1);  getattr_l__mod___features___6___expand1x1_activation = getattr_l__mod___features___6___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___7___squeeze = self.getattr_L__mod___features___7___squeeze(cat_2);  cat_2 = None
    x_3 = self.getattr_L__mod___features___7___squeeze_activation(getattr_l__mod___features___7___squeeze);  getattr_l__mod___features___7___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___7___expand1x1 = self.getattr_L__mod___features___7___expand1x1(x_3)
    getattr_l__mod___features___7___expand1x1_activation = self.getattr_L__mod___features___7___expand1x1_activation(getattr_l__mod___features___7___expand1x1);  getattr_l__mod___features___7___expand1x1 = None
    getattr_l__mod___features___7___expand3x3 = self.getattr_L__mod___features___7___expand3x3(x_3);  x_3 = None
    getattr_l__mod___features___7___expand3x3_activation = self.getattr_L__mod___features___7___expand3x3_activation(getattr_l__mod___features___7___expand3x3);  getattr_l__mod___features___7___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_3 = torch.cat([getattr_l__mod___features___7___expand1x1_activation, getattr_l__mod___features___7___expand3x3_activation], 1);  getattr_l__mod___features___7___expand1x1_activation = getattr_l__mod___features___7___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    l__mod___features_8 = self.L__mod___features_8(cat_3);  cat_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___9___squeeze = self.getattr_L__mod___features___9___squeeze(l__mod___features_8);  l__mod___features_8 = None
    x_4 = self.getattr_L__mod___features___9___squeeze_activation(getattr_l__mod___features___9___squeeze);  getattr_l__mod___features___9___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___9___expand1x1 = self.getattr_L__mod___features___9___expand1x1(x_4)
    getattr_l__mod___features___9___expand1x1_activation = self.getattr_L__mod___features___9___expand1x1_activation(getattr_l__mod___features___9___expand1x1);  getattr_l__mod___features___9___expand1x1 = None
    getattr_l__mod___features___9___expand3x3 = self.getattr_L__mod___features___9___expand3x3(x_4);  x_4 = None
    getattr_l__mod___features___9___expand3x3_activation = self.getattr_L__mod___features___9___expand3x3_activation(getattr_l__mod___features___9___expand3x3);  getattr_l__mod___features___9___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_4 = torch.cat([getattr_l__mod___features___9___expand1x1_activation, getattr_l__mod___features___9___expand3x3_activation], 1);  getattr_l__mod___features___9___expand1x1_activation = getattr_l__mod___features___9___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___10___squeeze = self.getattr_L__mod___features___10___squeeze(cat_4);  cat_4 = None
    x_5 = self.getattr_L__mod___features___10___squeeze_activation(getattr_l__mod___features___10___squeeze);  getattr_l__mod___features___10___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___10___expand1x1 = self.getattr_L__mod___features___10___expand1x1(x_5)
    getattr_l__mod___features___10___expand1x1_activation = self.getattr_L__mod___features___10___expand1x1_activation(getattr_l__mod___features___10___expand1x1);  getattr_l__mod___features___10___expand1x1 = None
    getattr_l__mod___features___10___expand3x3 = self.getattr_L__mod___features___10___expand3x3(x_5);  x_5 = None
    getattr_l__mod___features___10___expand3x3_activation = self.getattr_L__mod___features___10___expand3x3_activation(getattr_l__mod___features___10___expand3x3);  getattr_l__mod___features___10___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_5 = torch.cat([getattr_l__mod___features___10___expand1x1_activation, getattr_l__mod___features___10___expand3x3_activation], 1);  getattr_l__mod___features___10___expand1x1_activation = getattr_l__mod___features___10___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___11___squeeze = self.getattr_L__mod___features___11___squeeze(cat_5);  cat_5 = None
    x_6 = self.getattr_L__mod___features___11___squeeze_activation(getattr_l__mod___features___11___squeeze);  getattr_l__mod___features___11___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___11___expand1x1 = self.getattr_L__mod___features___11___expand1x1(x_6)
    getattr_l__mod___features___11___expand1x1_activation = self.getattr_L__mod___features___11___expand1x1_activation(getattr_l__mod___features___11___expand1x1);  getattr_l__mod___features___11___expand1x1 = None
    getattr_l__mod___features___11___expand3x3 = self.getattr_L__mod___features___11___expand3x3(x_6);  x_6 = None
    getattr_l__mod___features___11___expand3x3_activation = self.getattr_L__mod___features___11___expand3x3_activation(getattr_l__mod___features___11___expand3x3);  getattr_l__mod___features___11___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_6 = torch.cat([getattr_l__mod___features___11___expand1x1_activation, getattr_l__mod___features___11___expand3x3_activation], 1);  getattr_l__mod___features___11___expand1x1_activation = getattr_l__mod___features___11___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    getattr_l__mod___features___12___squeeze = self.getattr_L__mod___features___12___squeeze(cat_6);  cat_6 = None
    x_7 = self.getattr_L__mod___features___12___squeeze_activation(getattr_l__mod___features___12___squeeze);  getattr_l__mod___features___12___squeeze = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    getattr_l__mod___features___12___expand1x1 = self.getattr_L__mod___features___12___expand1x1(x_7)
    getattr_l__mod___features___12___expand1x1_activation = self.getattr_L__mod___features___12___expand1x1_activation(getattr_l__mod___features___12___expand1x1);  getattr_l__mod___features___12___expand1x1 = None
    getattr_l__mod___features___12___expand3x3 = self.getattr_L__mod___features___12___expand3x3(x_7);  x_7 = None
    getattr_l__mod___features___12___expand3x3_activation = self.getattr_L__mod___features___12___expand3x3_activation(getattr_l__mod___features___12___expand3x3);  getattr_l__mod___features___12___expand3x3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    x_8 = torch.cat([getattr_l__mod___features___12___expand1x1_activation, getattr_l__mod___features___12___expand3x3_activation], 1);  getattr_l__mod___features___12___expand1x1_activation = getattr_l__mod___features___12___expand3x3_activation = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    l__mod___classifier_0 = self.L__mod___classifier_0(x_8);  x_8 = None
    l__mod___classifier_1 = self.L__mod___classifier_1(l__mod___classifier_0);  l__mod___classifier_0 = None
    l__mod___classifier_2 = self.L__mod___classifier_2(l__mod___classifier_1);  l__mod___classifier_1 = None
    x_9 = self.L__mod___classifier_3(l__mod___classifier_2);  l__mod___classifier_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:97, code: return torch.flatten(x, 1)
    pred = torch.flatten(x_9, 1);  x_9 = None
    return (pred,)
    