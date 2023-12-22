from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:48, code: x = self.features(x)
    l__mod___features_0 = self.L__mod___features_0(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    l__mod___features_1 = self.L__mod___features_1(l__mod___features_0);  l__mod___features_0 = None
    l__mod___features_2 = self.L__mod___features_2(l__mod___features_1);  l__mod___features_1 = None
    l__mod___features_3 = self.L__mod___features_3(l__mod___features_2);  l__mod___features_2 = None
    l__mod___features_4 = self.L__mod___features_4(l__mod___features_3);  l__mod___features_3 = None
    l__mod___features_5 = self.L__mod___features_5(l__mod___features_4);  l__mod___features_4 = None
    l__mod___features_6 = self.L__mod___features_6(l__mod___features_5);  l__mod___features_5 = None
    l__mod___features_7 = self.L__mod___features_7(l__mod___features_6);  l__mod___features_6 = None
    l__mod___features_8 = self.L__mod___features_8(l__mod___features_7);  l__mod___features_7 = None
    l__mod___features_9 = self.L__mod___features_9(l__mod___features_8);  l__mod___features_8 = None
    l__mod___features_10 = self.L__mod___features_10(l__mod___features_9);  l__mod___features_9 = None
    l__mod___features_11 = self.L__mod___features_11(l__mod___features_10);  l__mod___features_10 = None
    x = self.L__mod___features_12(l__mod___features_11);  l__mod___features_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:49, code: x = self.avgpool(x)
    x_1 = self.L__mod___avgpool(x);  x = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:50, code: x = torch.flatten(x, 1)
    x_2 = torch.flatten(x_1, 1);  x_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/alexnet.py:51, code: x = self.classifier(x)
    l__mod___classifier_0 = self.L__mod___classifier_0(x_2);  x_2 = None
    l__mod___classifier_1 = self.L__mod___classifier_1(l__mod___classifier_0);  l__mod___classifier_0 = None
    l__mod___classifier_2 = self.L__mod___classifier_2(l__mod___classifier_1);  l__mod___classifier_1 = None
    l__mod___classifier_3 = self.L__mod___classifier_3(l__mod___classifier_2);  l__mod___classifier_2 = None
    l__mod___classifier_4 = self.L__mod___classifier_4(l__mod___classifier_3);  l__mod___classifier_3 = None
    l__mod___classifier_5 = self.L__mod___classifier_5(l__mod___classifier_4);  l__mod___classifier_4 = None
    pred = self.L__mod___classifier_6(l__mod___classifier_5);  l__mod___classifier_5 = None
    return (pred,)
    