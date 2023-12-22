from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    l__mod___features_0_0 = self.L__mod___features_0_0(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    l__mod___features_0_1 = self.L__mod___features_0_1(l__mod___features_0_0);  l__mod___features_0_0 = None
    l__mod___features_0_2 = self.L__mod___features_0_2(l__mod___features_0_1);  l__mod___features_0_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___1___block_0_0 = self.getattr_L__mod___features___1___block_0_0(l__mod___features_0_2)
    getattr_l__mod___features___1___block_0_1 = self.getattr_L__mod___features___1___block_0_1(getattr_l__mod___features___1___block_0_0);  getattr_l__mod___features___1___block_0_0 = None
    getattr_l__mod___features___1___block_0_2 = self.getattr_L__mod___features___1___block_0_2(getattr_l__mod___features___1___block_0_1);  getattr_l__mod___features___1___block_0_1 = None
    getattr_l__mod___features___1___block_1_0 = self.getattr_L__mod___features___1___block_1_0(getattr_l__mod___features___1___block_0_2);  getattr_l__mod___features___1___block_0_2 = None
    result = self.getattr_L__mod___features___1___block_1_1(getattr_l__mod___features___1___block_1_0);  getattr_l__mod___features___1___block_1_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result += l__mod___features_0_2;  result_1 = result;  result = l__mod___features_0_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___2___block_0_0 = self.getattr_L__mod___features___2___block_0_0(result_1);  result_1 = None
    getattr_l__mod___features___2___block_0_1 = self.getattr_L__mod___features___2___block_0_1(getattr_l__mod___features___2___block_0_0);  getattr_l__mod___features___2___block_0_0 = None
    getattr_l__mod___features___2___block_0_2 = self.getattr_L__mod___features___2___block_0_2(getattr_l__mod___features___2___block_0_1);  getattr_l__mod___features___2___block_0_1 = None
    getattr_l__mod___features___2___block_1_0 = self.getattr_L__mod___features___2___block_1_0(getattr_l__mod___features___2___block_0_2);  getattr_l__mod___features___2___block_0_2 = None
    getattr_l__mod___features___2___block_1_1 = self.getattr_L__mod___features___2___block_1_1(getattr_l__mod___features___2___block_1_0);  getattr_l__mod___features___2___block_1_0 = None
    getattr_l__mod___features___2___block_1_2 = self.getattr_L__mod___features___2___block_1_2(getattr_l__mod___features___2___block_1_1);  getattr_l__mod___features___2___block_1_1 = None
    getattr_l__mod___features___2___block_2_0 = self.getattr_L__mod___features___2___block_2_0(getattr_l__mod___features___2___block_1_2);  getattr_l__mod___features___2___block_1_2 = None
    result_2 = self.getattr_L__mod___features___2___block_2_1(getattr_l__mod___features___2___block_2_0);  getattr_l__mod___features___2___block_2_0 = None
    getattr_l__mod___features___3___block_0_0 = self.getattr_L__mod___features___3___block_0_0(result_2)
    getattr_l__mod___features___3___block_0_1 = self.getattr_L__mod___features___3___block_0_1(getattr_l__mod___features___3___block_0_0);  getattr_l__mod___features___3___block_0_0 = None
    getattr_l__mod___features___3___block_0_2 = self.getattr_L__mod___features___3___block_0_2(getattr_l__mod___features___3___block_0_1);  getattr_l__mod___features___3___block_0_1 = None
    getattr_l__mod___features___3___block_1_0 = self.getattr_L__mod___features___3___block_1_0(getattr_l__mod___features___3___block_0_2);  getattr_l__mod___features___3___block_0_2 = None
    getattr_l__mod___features___3___block_1_1 = self.getattr_L__mod___features___3___block_1_1(getattr_l__mod___features___3___block_1_0);  getattr_l__mod___features___3___block_1_0 = None
    getattr_l__mod___features___3___block_1_2 = self.getattr_L__mod___features___3___block_1_2(getattr_l__mod___features___3___block_1_1);  getattr_l__mod___features___3___block_1_1 = None
    getattr_l__mod___features___3___block_2_0 = self.getattr_L__mod___features___3___block_2_0(getattr_l__mod___features___3___block_1_2);  getattr_l__mod___features___3___block_1_2 = None
    result_3 = self.getattr_L__mod___features___3___block_2_1(getattr_l__mod___features___3___block_2_0);  getattr_l__mod___features___3___block_2_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_3 += result_2;  result_4 = result_3;  result_3 = result_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___4___block_0_0 = self.getattr_L__mod___features___4___block_0_0(result_4);  result_4 = None
    getattr_l__mod___features___4___block_0_1 = self.getattr_L__mod___features___4___block_0_1(getattr_l__mod___features___4___block_0_0);  getattr_l__mod___features___4___block_0_0 = None
    getattr_l__mod___features___4___block_0_2 = self.getattr_L__mod___features___4___block_0_2(getattr_l__mod___features___4___block_0_1);  getattr_l__mod___features___4___block_0_1 = None
    getattr_l__mod___features___4___block_1_0 = self.getattr_L__mod___features___4___block_1_0(getattr_l__mod___features___4___block_0_2);  getattr_l__mod___features___4___block_0_2 = None
    getattr_l__mod___features___4___block_1_1 = self.getattr_L__mod___features___4___block_1_1(getattr_l__mod___features___4___block_1_0);  getattr_l__mod___features___4___block_1_0 = None
    getattr_l__mod___features___4___block_1_2 = self.getattr_L__mod___features___4___block_1_2(getattr_l__mod___features___4___block_1_1);  getattr_l__mod___features___4___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale = self.getattr_getattr_L__mod___features___4___block___2___avgpool(getattr_l__mod___features___4___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_1 = self.getattr_getattr_L__mod___features___4___block___2___fc1(scale);  scale = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_2 = self.getattr_getattr_L__mod___features___4___block___2___activation(scale_1);  scale_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_3 = self.getattr_getattr_L__mod___features___4___block___2___fc2(scale_2);  scale_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_4 = self.getattr_getattr_L__mod___features___4___block___2___scale_activation(scale_3);  scale_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul = scale_4 * getattr_l__mod___features___4___block_1_2;  scale_4 = getattr_l__mod___features___4___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___4___block_3_0 = self.getattr_L__mod___features___4___block_3_0(mul);  mul = None
    result_5 = self.getattr_L__mod___features___4___block_3_1(getattr_l__mod___features___4___block_3_0);  getattr_l__mod___features___4___block_3_0 = None
    getattr_l__mod___features___5___block_0_0 = self.getattr_L__mod___features___5___block_0_0(result_5)
    getattr_l__mod___features___5___block_0_1 = self.getattr_L__mod___features___5___block_0_1(getattr_l__mod___features___5___block_0_0);  getattr_l__mod___features___5___block_0_0 = None
    getattr_l__mod___features___5___block_0_2 = self.getattr_L__mod___features___5___block_0_2(getattr_l__mod___features___5___block_0_1);  getattr_l__mod___features___5___block_0_1 = None
    getattr_l__mod___features___5___block_1_0 = self.getattr_L__mod___features___5___block_1_0(getattr_l__mod___features___5___block_0_2);  getattr_l__mod___features___5___block_0_2 = None
    getattr_l__mod___features___5___block_1_1 = self.getattr_L__mod___features___5___block_1_1(getattr_l__mod___features___5___block_1_0);  getattr_l__mod___features___5___block_1_0 = None
    getattr_l__mod___features___5___block_1_2 = self.getattr_L__mod___features___5___block_1_2(getattr_l__mod___features___5___block_1_1);  getattr_l__mod___features___5___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_5 = self.getattr_getattr_L__mod___features___5___block___2___avgpool(getattr_l__mod___features___5___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_6 = self.getattr_getattr_L__mod___features___5___block___2___fc1(scale_5);  scale_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_7 = self.getattr_getattr_L__mod___features___5___block___2___activation(scale_6);  scale_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_8 = self.getattr_getattr_L__mod___features___5___block___2___fc2(scale_7);  scale_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_9 = self.getattr_getattr_L__mod___features___5___block___2___scale_activation(scale_8);  scale_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_1 = scale_9 * getattr_l__mod___features___5___block_1_2;  scale_9 = getattr_l__mod___features___5___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___5___block_3_0 = self.getattr_L__mod___features___5___block_3_0(mul_1);  mul_1 = None
    result_6 = self.getattr_L__mod___features___5___block_3_1(getattr_l__mod___features___5___block_3_0);  getattr_l__mod___features___5___block_3_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_6 += result_5;  result_7 = result_6;  result_6 = result_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___6___block_0_0 = self.getattr_L__mod___features___6___block_0_0(result_7)
    getattr_l__mod___features___6___block_0_1 = self.getattr_L__mod___features___6___block_0_1(getattr_l__mod___features___6___block_0_0);  getattr_l__mod___features___6___block_0_0 = None
    getattr_l__mod___features___6___block_0_2 = self.getattr_L__mod___features___6___block_0_2(getattr_l__mod___features___6___block_0_1);  getattr_l__mod___features___6___block_0_1 = None
    getattr_l__mod___features___6___block_1_0 = self.getattr_L__mod___features___6___block_1_0(getattr_l__mod___features___6___block_0_2);  getattr_l__mod___features___6___block_0_2 = None
    getattr_l__mod___features___6___block_1_1 = self.getattr_L__mod___features___6___block_1_1(getattr_l__mod___features___6___block_1_0);  getattr_l__mod___features___6___block_1_0 = None
    getattr_l__mod___features___6___block_1_2 = self.getattr_L__mod___features___6___block_1_2(getattr_l__mod___features___6___block_1_1);  getattr_l__mod___features___6___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_10 = self.getattr_getattr_L__mod___features___6___block___2___avgpool(getattr_l__mod___features___6___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_11 = self.getattr_getattr_L__mod___features___6___block___2___fc1(scale_10);  scale_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_12 = self.getattr_getattr_L__mod___features___6___block___2___activation(scale_11);  scale_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_13 = self.getattr_getattr_L__mod___features___6___block___2___fc2(scale_12);  scale_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_14 = self.getattr_getattr_L__mod___features___6___block___2___scale_activation(scale_13);  scale_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_2 = scale_14 * getattr_l__mod___features___6___block_1_2;  scale_14 = getattr_l__mod___features___6___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___6___block_3_0 = self.getattr_L__mod___features___6___block_3_0(mul_2);  mul_2 = None
    result_8 = self.getattr_L__mod___features___6___block_3_1(getattr_l__mod___features___6___block_3_0);  getattr_l__mod___features___6___block_3_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_8 += result_7;  result_9 = result_8;  result_8 = result_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___7___block_0_0 = self.getattr_L__mod___features___7___block_0_0(result_9);  result_9 = None
    getattr_l__mod___features___7___block_0_1 = self.getattr_L__mod___features___7___block_0_1(getattr_l__mod___features___7___block_0_0);  getattr_l__mod___features___7___block_0_0 = None
    getattr_l__mod___features___7___block_0_2 = self.getattr_L__mod___features___7___block_0_2(getattr_l__mod___features___7___block_0_1);  getattr_l__mod___features___7___block_0_1 = None
    getattr_l__mod___features___7___block_1_0 = self.getattr_L__mod___features___7___block_1_0(getattr_l__mod___features___7___block_0_2);  getattr_l__mod___features___7___block_0_2 = None
    getattr_l__mod___features___7___block_1_1 = self.getattr_L__mod___features___7___block_1_1(getattr_l__mod___features___7___block_1_0);  getattr_l__mod___features___7___block_1_0 = None
    getattr_l__mod___features___7___block_1_2 = self.getattr_L__mod___features___7___block_1_2(getattr_l__mod___features___7___block_1_1);  getattr_l__mod___features___7___block_1_1 = None
    getattr_l__mod___features___7___block_2_0 = self.getattr_L__mod___features___7___block_2_0(getattr_l__mod___features___7___block_1_2);  getattr_l__mod___features___7___block_1_2 = None
    result_10 = self.getattr_L__mod___features___7___block_2_1(getattr_l__mod___features___7___block_2_0);  getattr_l__mod___features___7___block_2_0 = None
    getattr_l__mod___features___8___block_0_0 = self.getattr_L__mod___features___8___block_0_0(result_10)
    getattr_l__mod___features___8___block_0_1 = self.getattr_L__mod___features___8___block_0_1(getattr_l__mod___features___8___block_0_0);  getattr_l__mod___features___8___block_0_0 = None
    getattr_l__mod___features___8___block_0_2 = self.getattr_L__mod___features___8___block_0_2(getattr_l__mod___features___8___block_0_1);  getattr_l__mod___features___8___block_0_1 = None
    getattr_l__mod___features___8___block_1_0 = self.getattr_L__mod___features___8___block_1_0(getattr_l__mod___features___8___block_0_2);  getattr_l__mod___features___8___block_0_2 = None
    getattr_l__mod___features___8___block_1_1 = self.getattr_L__mod___features___8___block_1_1(getattr_l__mod___features___8___block_1_0);  getattr_l__mod___features___8___block_1_0 = None
    getattr_l__mod___features___8___block_1_2 = self.getattr_L__mod___features___8___block_1_2(getattr_l__mod___features___8___block_1_1);  getattr_l__mod___features___8___block_1_1 = None
    getattr_l__mod___features___8___block_2_0 = self.getattr_L__mod___features___8___block_2_0(getattr_l__mod___features___8___block_1_2);  getattr_l__mod___features___8___block_1_2 = None
    result_11 = self.getattr_L__mod___features___8___block_2_1(getattr_l__mod___features___8___block_2_0);  getattr_l__mod___features___8___block_2_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_11 += result_10;  result_12 = result_11;  result_11 = result_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___9___block_0_0 = self.getattr_L__mod___features___9___block_0_0(result_12)
    getattr_l__mod___features___9___block_0_1 = self.getattr_L__mod___features___9___block_0_1(getattr_l__mod___features___9___block_0_0);  getattr_l__mod___features___9___block_0_0 = None
    getattr_l__mod___features___9___block_0_2 = self.getattr_L__mod___features___9___block_0_2(getattr_l__mod___features___9___block_0_1);  getattr_l__mod___features___9___block_0_1 = None
    getattr_l__mod___features___9___block_1_0 = self.getattr_L__mod___features___9___block_1_0(getattr_l__mod___features___9___block_0_2);  getattr_l__mod___features___9___block_0_2 = None
    getattr_l__mod___features___9___block_1_1 = self.getattr_L__mod___features___9___block_1_1(getattr_l__mod___features___9___block_1_0);  getattr_l__mod___features___9___block_1_0 = None
    getattr_l__mod___features___9___block_1_2 = self.getattr_L__mod___features___9___block_1_2(getattr_l__mod___features___9___block_1_1);  getattr_l__mod___features___9___block_1_1 = None
    getattr_l__mod___features___9___block_2_0 = self.getattr_L__mod___features___9___block_2_0(getattr_l__mod___features___9___block_1_2);  getattr_l__mod___features___9___block_1_2 = None
    result_13 = self.getattr_L__mod___features___9___block_2_1(getattr_l__mod___features___9___block_2_0);  getattr_l__mod___features___9___block_2_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_13 += result_12;  result_14 = result_13;  result_13 = result_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___10___block_0_0 = self.getattr_L__mod___features___10___block_0_0(result_14)
    getattr_l__mod___features___10___block_0_1 = self.getattr_L__mod___features___10___block_0_1(getattr_l__mod___features___10___block_0_0);  getattr_l__mod___features___10___block_0_0 = None
    getattr_l__mod___features___10___block_0_2 = self.getattr_L__mod___features___10___block_0_2(getattr_l__mod___features___10___block_0_1);  getattr_l__mod___features___10___block_0_1 = None
    getattr_l__mod___features___10___block_1_0 = self.getattr_L__mod___features___10___block_1_0(getattr_l__mod___features___10___block_0_2);  getattr_l__mod___features___10___block_0_2 = None
    getattr_l__mod___features___10___block_1_1 = self.getattr_L__mod___features___10___block_1_1(getattr_l__mod___features___10___block_1_0);  getattr_l__mod___features___10___block_1_0 = None
    getattr_l__mod___features___10___block_1_2 = self.getattr_L__mod___features___10___block_1_2(getattr_l__mod___features___10___block_1_1);  getattr_l__mod___features___10___block_1_1 = None
    getattr_l__mod___features___10___block_2_0 = self.getattr_L__mod___features___10___block_2_0(getattr_l__mod___features___10___block_1_2);  getattr_l__mod___features___10___block_1_2 = None
    result_15 = self.getattr_L__mod___features___10___block_2_1(getattr_l__mod___features___10___block_2_0);  getattr_l__mod___features___10___block_2_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_15 += result_14;  result_16 = result_15;  result_15 = result_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___11___block_0_0 = self.getattr_L__mod___features___11___block_0_0(result_16);  result_16 = None
    getattr_l__mod___features___11___block_0_1 = self.getattr_L__mod___features___11___block_0_1(getattr_l__mod___features___11___block_0_0);  getattr_l__mod___features___11___block_0_0 = None
    getattr_l__mod___features___11___block_0_2 = self.getattr_L__mod___features___11___block_0_2(getattr_l__mod___features___11___block_0_1);  getattr_l__mod___features___11___block_0_1 = None
    getattr_l__mod___features___11___block_1_0 = self.getattr_L__mod___features___11___block_1_0(getattr_l__mod___features___11___block_0_2);  getattr_l__mod___features___11___block_0_2 = None
    getattr_l__mod___features___11___block_1_1 = self.getattr_L__mod___features___11___block_1_1(getattr_l__mod___features___11___block_1_0);  getattr_l__mod___features___11___block_1_0 = None
    getattr_l__mod___features___11___block_1_2 = self.getattr_L__mod___features___11___block_1_2(getattr_l__mod___features___11___block_1_1);  getattr_l__mod___features___11___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_15 = self.getattr_getattr_L__mod___features___11___block___2___avgpool(getattr_l__mod___features___11___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_16 = self.getattr_getattr_L__mod___features___11___block___2___fc1(scale_15);  scale_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_17 = self.getattr_getattr_L__mod___features___11___block___2___activation(scale_16);  scale_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_18 = self.getattr_getattr_L__mod___features___11___block___2___fc2(scale_17);  scale_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_19 = self.getattr_getattr_L__mod___features___11___block___2___scale_activation(scale_18);  scale_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_3 = scale_19 * getattr_l__mod___features___11___block_1_2;  scale_19 = getattr_l__mod___features___11___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___11___block_3_0 = self.getattr_L__mod___features___11___block_3_0(mul_3);  mul_3 = None
    result_17 = self.getattr_L__mod___features___11___block_3_1(getattr_l__mod___features___11___block_3_0);  getattr_l__mod___features___11___block_3_0 = None
    getattr_l__mod___features___12___block_0_0 = self.getattr_L__mod___features___12___block_0_0(result_17)
    getattr_l__mod___features___12___block_0_1 = self.getattr_L__mod___features___12___block_0_1(getattr_l__mod___features___12___block_0_0);  getattr_l__mod___features___12___block_0_0 = None
    getattr_l__mod___features___12___block_0_2 = self.getattr_L__mod___features___12___block_0_2(getattr_l__mod___features___12___block_0_1);  getattr_l__mod___features___12___block_0_1 = None
    getattr_l__mod___features___12___block_1_0 = self.getattr_L__mod___features___12___block_1_0(getattr_l__mod___features___12___block_0_2);  getattr_l__mod___features___12___block_0_2 = None
    getattr_l__mod___features___12___block_1_1 = self.getattr_L__mod___features___12___block_1_1(getattr_l__mod___features___12___block_1_0);  getattr_l__mod___features___12___block_1_0 = None
    getattr_l__mod___features___12___block_1_2 = self.getattr_L__mod___features___12___block_1_2(getattr_l__mod___features___12___block_1_1);  getattr_l__mod___features___12___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_20 = self.getattr_getattr_L__mod___features___12___block___2___avgpool(getattr_l__mod___features___12___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_21 = self.getattr_getattr_L__mod___features___12___block___2___fc1(scale_20);  scale_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_22 = self.getattr_getattr_L__mod___features___12___block___2___activation(scale_21);  scale_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_23 = self.getattr_getattr_L__mod___features___12___block___2___fc2(scale_22);  scale_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_24 = self.getattr_getattr_L__mod___features___12___block___2___scale_activation(scale_23);  scale_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_4 = scale_24 * getattr_l__mod___features___12___block_1_2;  scale_24 = getattr_l__mod___features___12___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___12___block_3_0 = self.getattr_L__mod___features___12___block_3_0(mul_4);  mul_4 = None
    result_18 = self.getattr_L__mod___features___12___block_3_1(getattr_l__mod___features___12___block_3_0);  getattr_l__mod___features___12___block_3_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_18 += result_17;  result_19 = result_18;  result_18 = result_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___13___block_0_0 = self.getattr_L__mod___features___13___block_0_0(result_19);  result_19 = None
    getattr_l__mod___features___13___block_0_1 = self.getattr_L__mod___features___13___block_0_1(getattr_l__mod___features___13___block_0_0);  getattr_l__mod___features___13___block_0_0 = None
    getattr_l__mod___features___13___block_0_2 = self.getattr_L__mod___features___13___block_0_2(getattr_l__mod___features___13___block_0_1);  getattr_l__mod___features___13___block_0_1 = None
    getattr_l__mod___features___13___block_1_0 = self.getattr_L__mod___features___13___block_1_0(getattr_l__mod___features___13___block_0_2);  getattr_l__mod___features___13___block_0_2 = None
    getattr_l__mod___features___13___block_1_1 = self.getattr_L__mod___features___13___block_1_1(getattr_l__mod___features___13___block_1_0);  getattr_l__mod___features___13___block_1_0 = None
    getattr_l__mod___features___13___block_1_2 = self.getattr_L__mod___features___13___block_1_2(getattr_l__mod___features___13___block_1_1);  getattr_l__mod___features___13___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_25 = self.getattr_getattr_L__mod___features___13___block___2___avgpool(getattr_l__mod___features___13___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_26 = self.getattr_getattr_L__mod___features___13___block___2___fc1(scale_25);  scale_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_27 = self.getattr_getattr_L__mod___features___13___block___2___activation(scale_26);  scale_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_28 = self.getattr_getattr_L__mod___features___13___block___2___fc2(scale_27);  scale_27 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_29 = self.getattr_getattr_L__mod___features___13___block___2___scale_activation(scale_28);  scale_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_5 = scale_29 * getattr_l__mod___features___13___block_1_2;  scale_29 = getattr_l__mod___features___13___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___13___block_3_0 = self.getattr_L__mod___features___13___block_3_0(mul_5);  mul_5 = None
    result_20 = self.getattr_L__mod___features___13___block_3_1(getattr_l__mod___features___13___block_3_0);  getattr_l__mod___features___13___block_3_0 = None
    getattr_l__mod___features___14___block_0_0 = self.getattr_L__mod___features___14___block_0_0(result_20)
    getattr_l__mod___features___14___block_0_1 = self.getattr_L__mod___features___14___block_0_1(getattr_l__mod___features___14___block_0_0);  getattr_l__mod___features___14___block_0_0 = None
    getattr_l__mod___features___14___block_0_2 = self.getattr_L__mod___features___14___block_0_2(getattr_l__mod___features___14___block_0_1);  getattr_l__mod___features___14___block_0_1 = None
    getattr_l__mod___features___14___block_1_0 = self.getattr_L__mod___features___14___block_1_0(getattr_l__mod___features___14___block_0_2);  getattr_l__mod___features___14___block_0_2 = None
    getattr_l__mod___features___14___block_1_1 = self.getattr_L__mod___features___14___block_1_1(getattr_l__mod___features___14___block_1_0);  getattr_l__mod___features___14___block_1_0 = None
    getattr_l__mod___features___14___block_1_2 = self.getattr_L__mod___features___14___block_1_2(getattr_l__mod___features___14___block_1_1);  getattr_l__mod___features___14___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_30 = self.getattr_getattr_L__mod___features___14___block___2___avgpool(getattr_l__mod___features___14___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_31 = self.getattr_getattr_L__mod___features___14___block___2___fc1(scale_30);  scale_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_32 = self.getattr_getattr_L__mod___features___14___block___2___activation(scale_31);  scale_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_33 = self.getattr_getattr_L__mod___features___14___block___2___fc2(scale_32);  scale_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_34 = self.getattr_getattr_L__mod___features___14___block___2___scale_activation(scale_33);  scale_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_6 = scale_34 * getattr_l__mod___features___14___block_1_2;  scale_34 = getattr_l__mod___features___14___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___14___block_3_0 = self.getattr_L__mod___features___14___block_3_0(mul_6);  mul_6 = None
    result_21 = self.getattr_L__mod___features___14___block_3_1(getattr_l__mod___features___14___block_3_0);  getattr_l__mod___features___14___block_3_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_21 += result_20;  result_22 = result_21;  result_21 = result_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___15___block_0_0 = self.getattr_L__mod___features___15___block_0_0(result_22)
    getattr_l__mod___features___15___block_0_1 = self.getattr_L__mod___features___15___block_0_1(getattr_l__mod___features___15___block_0_0);  getattr_l__mod___features___15___block_0_0 = None
    getattr_l__mod___features___15___block_0_2 = self.getattr_L__mod___features___15___block_0_2(getattr_l__mod___features___15___block_0_1);  getattr_l__mod___features___15___block_0_1 = None
    getattr_l__mod___features___15___block_1_0 = self.getattr_L__mod___features___15___block_1_0(getattr_l__mod___features___15___block_0_2);  getattr_l__mod___features___15___block_0_2 = None
    getattr_l__mod___features___15___block_1_1 = self.getattr_L__mod___features___15___block_1_1(getattr_l__mod___features___15___block_1_0);  getattr_l__mod___features___15___block_1_0 = None
    getattr_l__mod___features___15___block_1_2 = self.getattr_L__mod___features___15___block_1_2(getattr_l__mod___features___15___block_1_1);  getattr_l__mod___features___15___block_1_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    scale_35 = self.getattr_getattr_L__mod___features___15___block___2___avgpool(getattr_l__mod___features___15___block_1_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    scale_36 = self.getattr_getattr_L__mod___features___15___block___2___fc1(scale_35);  scale_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    scale_37 = self.getattr_getattr_L__mod___features___15___block___2___activation(scale_36);  scale_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    scale_38 = self.getattr_getattr_L__mod___features___15___block___2___fc2(scale_37);  scale_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    scale_39 = self.getattr_getattr_L__mod___features___15___block___2___scale_activation(scale_38);  scale_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_7 = scale_39 * getattr_l__mod___features___15___block_1_2;  scale_39 = getattr_l__mod___features___15___block_1_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    getattr_l__mod___features___15___block_3_0 = self.getattr_L__mod___features___15___block_3_0(mul_7);  mul_7 = None
    result_23 = self.getattr_L__mod___features___15___block_3_1(getattr_l__mod___features___15___block_3_0);  getattr_l__mod___features___15___block_3_0 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:113, code: result += input
    result_23 += result_22;  result_24 = result_23;  result_23 = result_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    l__mod___features_16_0 = self.L__mod___features_16_0(result_24);  result_24 = None
    l__mod___features_16_1 = self.L__mod___features_16_1(l__mod___features_16_0);  l__mod___features_16_0 = None
    x = self.L__mod___features_16_2(l__mod___features_16_1);  l__mod___features_16_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
    x_1 = self.L__mod___avgpool(x);  x = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
    x_2 = torch.flatten(x_1, 1);  x_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:215, code: x = self.classifier(x)
    l__mod___classifier_0 = self.L__mod___classifier_0(x_2);  x_2 = None
    l__mod___classifier_1 = self.L__mod___classifier_1(l__mod___classifier_0);  l__mod___classifier_0 = None
    l__mod___classifier_2 = self.L__mod___classifier_2(l__mod___classifier_1);  l__mod___classifier_1 = None
    pred = self.L__mod___classifier_3(l__mod___classifier_2);  l__mod___classifier_2 = None
    return (pred,)
    