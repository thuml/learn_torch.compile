from __future__ import annotations



def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[64]", arg2_1: "f32[16, 64, 1, 1]", arg3_1: "f32[16]", arg4_1: "f32[64, 16, 1, 1]", arg5_1: "f32[64]", arg6_1: "f32[64, 16, 3, 3]", arg7_1: "f32[64]", arg8_1: "f32[16, 128, 1, 1]", arg9_1: "f32[16]", arg10_1: "f32[64, 16, 1, 1]", arg11_1: "f32[64]", arg12_1: "f32[64, 16, 3, 3]", arg13_1: "f32[64]", arg14_1: "f32[32, 128, 1, 1]", arg15_1: "f32[32]", arg16_1: "f32[128, 32, 1, 1]", arg17_1: "f32[128]", arg18_1: "f32[128, 32, 3, 3]", arg19_1: "f32[128]", arg20_1: "f32[32, 256, 1, 1]", arg21_1: "f32[32]", arg22_1: "f32[128, 32, 1, 1]", arg23_1: "f32[128]", arg24_1: "f32[128, 32, 3, 3]", arg25_1: "f32[128]", arg26_1: "f32[48, 256, 1, 1]", arg27_1: "f32[48]", arg28_1: "f32[192, 48, 1, 1]", arg29_1: "f32[192]", arg30_1: "f32[192, 48, 3, 3]", arg31_1: "f32[192]", arg32_1: "f32[48, 384, 1, 1]", arg33_1: "f32[48]", arg34_1: "f32[192, 48, 1, 1]", arg35_1: "f32[192]", arg36_1: "f32[192, 48, 3, 3]", arg37_1: "f32[192]", arg38_1: "f32[64, 384, 1, 1]", arg39_1: "f32[64]", arg40_1: "f32[256, 64, 1, 1]", arg41_1: "f32[256]", arg42_1: "f32[256, 64, 3, 3]", arg43_1: "f32[256]", arg44_1: "f32[64, 512, 1, 1]", arg45_1: "f32[64]", arg46_1: "f32[256, 64, 1, 1]", arg47_1: "f32[256]", arg48_1: "f32[256, 64, 3, 3]", arg49_1: "f32[256]", arg50_1: "f32[1000, 512, 1, 1]", arg51_1: "f32[1000]", arg52_1: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    convolution: "f32[4, 64, 111, 111]" = torch.ops.aten.convolution.default(arg52_1, arg0_1, arg1_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg52_1 = arg0_1 = arg1_1 = None
    relu: "f32[4, 64, 111, 111]" = torch.ops.aten.relu.default(convolution);  convolution = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True);  relu = None
    getitem: "f32[4, 64, 55, 55]" = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_1: "f32[4, 16, 55, 55]" = torch.ops.aten.convolution.default(getitem, arg2_1, arg3_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg2_1 = arg3_1 = None
    relu_1: "f32[4, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_2: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, arg4_1, arg5_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg4_1 = arg5_1 = None
    relu_2: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, arg6_1, arg7_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = arg7_1 = None
    relu_3: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat: "f32[4, 128, 55, 55]" = torch.ops.aten.cat.default([relu_2, relu_3], 1);  relu_2 = relu_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_4: "f32[4, 16, 55, 55]" = torch.ops.aten.convolution.default(cat, arg8_1, arg9_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat = arg8_1 = arg9_1 = None
    relu_4: "f32[4, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_5: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, arg10_1, arg11_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg10_1 = arg11_1 = None
    relu_5: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    convolution_6: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg12_1 = arg13_1 = None
    relu_6: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_1: "f32[4, 128, 55, 55]" = torch.ops.aten.cat.default([relu_5, relu_6], 1);  relu_5 = relu_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_1 = None
    getitem_2: "f32[4, 128, 27, 27]" = max_pool2d_with_indices_1[0];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_7: "f32[4, 32, 27, 27]" = torch.ops.aten.convolution.default(getitem_2, arg14_1, arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_2 = arg14_1 = arg15_1 = None
    relu_7: "f32[4, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_8: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, arg16_1, arg17_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = arg17_1 = None
    relu_8: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    convolution_9: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, arg18_1, arg19_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg18_1 = arg19_1 = None
    relu_9: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_2: "f32[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_8, relu_9], 1);  relu_8 = relu_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_10: "f32[4, 32, 27, 27]" = torch.ops.aten.convolution.default(cat_2, arg20_1, arg21_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_2 = arg20_1 = arg21_1 = None
    relu_10: "f32[4, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_11: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, arg22_1, arg23_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg22_1 = arg23_1 = None
    relu_11: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    convolution_12: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, arg24_1, arg25_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg24_1 = arg25_1 = None
    relu_12: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_3: "f32[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_11, relu_12], 1);  relu_11 = relu_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True);  cat_3 = None
    getitem_4: "f32[4, 256, 13, 13]" = max_pool2d_with_indices_2[0];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_13: "f32[4, 48, 13, 13]" = torch.ops.aten.convolution.default(getitem_4, arg26_1, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_4 = arg26_1 = arg27_1 = None
    relu_13: "f32[4, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_14: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg28_1 = arg29_1 = None
    relu_14: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    convolution_15: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, arg30_1, arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg30_1 = arg31_1 = None
    relu_15: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_4: "f32[4, 384, 13, 13]" = torch.ops.aten.cat.default([relu_14, relu_15], 1);  relu_14 = relu_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_16: "f32[4, 48, 13, 13]" = torch.ops.aten.convolution.default(cat_4, arg32_1, arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_4 = arg32_1 = arg33_1 = None
    relu_16: "f32[4, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_17: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, arg34_1, arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg34_1 = arg35_1 = None
    relu_17: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
    convolution_18: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, arg36_1, arg37_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg36_1 = arg37_1 = None
    relu_18: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_5: "f32[4, 384, 13, 13]" = torch.ops.aten.cat.default([relu_17, relu_18], 1);  relu_17 = relu_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_19: "f32[4, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_5, arg38_1, arg39_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_5 = arg38_1 = arg39_1 = None
    relu_19: "f32[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_20: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, arg40_1, arg41_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg40_1 = arg41_1 = None
    relu_20: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    convolution_21: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, arg42_1, arg43_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = arg42_1 = arg43_1 = None
    relu_21: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_6: "f32[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_20, relu_21], 1);  relu_20 = relu_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_22: "f32[4, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_6, arg44_1, arg45_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg44_1 = arg45_1 = None
    relu_22: "f32[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_23: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, arg46_1, arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg46_1 = arg47_1 = None
    relu_23: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
    convolution_24: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, arg48_1, arg49_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg48_1 = arg49_1 = None
    relu_24: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_7: "f32[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_23, relu_24], 1);  relu_23 = relu_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    clone: "f32[4, 512, 13, 13]" = torch.ops.aten.clone.default(cat_7);  cat_7 = None
    convolution_25: "f32[4, 1000, 13, 13]" = torch.ops.aten.convolution.default(clone, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone = arg50_1 = arg51_1 = None
    relu_25: "f32[4, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    mean: "f32[4, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True);  relu_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:97, code: return torch.flatten(x, 1)
    view: "f32[4, 1000]" = torch.ops.aten.view.default(mean, [4, 1000]);  mean = None
    return (view,)
    