from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_3: "f32[16, 64, 1, 1]", primals_4: "f32[16]", primals_5: "f32[64, 16, 1, 1]", primals_6: "f32[64]", primals_7: "f32[64, 16, 3, 3]", primals_8: "f32[64]", primals_9: "f32[16, 128, 1, 1]", primals_10: "f32[16]", primals_11: "f32[64, 16, 1, 1]", primals_12: "f32[64]", primals_13: "f32[64, 16, 3, 3]", primals_14: "f32[64]", primals_15: "f32[32, 128, 1, 1]", primals_16: "f32[32]", primals_17: "f32[128, 32, 1, 1]", primals_18: "f32[128]", primals_19: "f32[128, 32, 3, 3]", primals_20: "f32[128]", primals_21: "f32[32, 256, 1, 1]", primals_22: "f32[32]", primals_23: "f32[128, 32, 1, 1]", primals_24: "f32[128]", primals_25: "f32[128, 32, 3, 3]", primals_26: "f32[128]", primals_27: "f32[48, 256, 1, 1]", primals_28: "f32[48]", primals_29: "f32[192, 48, 1, 1]", primals_30: "f32[192]", primals_31: "f32[192, 48, 3, 3]", primals_32: "f32[192]", primals_33: "f32[48, 384, 1, 1]", primals_34: "f32[48]", primals_35: "f32[192, 48, 1, 1]", primals_36: "f32[192]", primals_37: "f32[192, 48, 3, 3]", primals_38: "f32[192]", primals_39: "f32[64, 384, 1, 1]", primals_40: "f32[64]", primals_41: "f32[256, 64, 1, 1]", primals_42: "f32[256]", primals_43: "f32[256, 64, 3, 3]", primals_44: "f32[256]", primals_45: "f32[64, 512, 1, 1]", primals_46: "f32[64]", primals_47: "f32[256, 64, 1, 1]", primals_48: "f32[256]", primals_49: "f32[256, 64, 3, 3]", primals_50: "f32[256]", primals_51: "f32[1000, 512, 1, 1]", primals_52: "f32[1000]", primals_53: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    convolution: "f32[4, 64, 111, 111]" = torch.ops.aten.convolution.default(primals_53, primals_1, primals_2, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    relu: "f32[4, 64, 111, 111]" = torch.ops.aten.relu.default(convolution);  convolution = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem: "f32[4, 64, 55, 55]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 55, 55]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_1: "f32[4, 16, 55, 55]" = torch.ops.aten.convolution.default(getitem, primals_3, primals_4, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_4 = None
    relu_1: "f32[4, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_2: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, primals_5, primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
    relu_2: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
    convolution_3: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_1, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_8 = None
    relu_3: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat: "f32[4, 128, 55, 55]" = torch.ops.aten.cat.default([relu_2, relu_3], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_4: "f32[4, 16, 55, 55]" = torch.ops.aten.convolution.default(cat, primals_9, primals_10, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_10 = None
    relu_4: "f32[4, 16, 55, 55]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_5: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, primals_11, primals_12, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_12 = None
    relu_5: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    convolution_6: "f32[4, 64, 55, 55]" = torch.ops.aten.convolution.default(relu_4, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_14 = None
    relu_6: "f32[4, 64, 55, 55]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_1: "f32[4, 128, 55, 55]" = torch.ops.aten.cat.default([relu_5, relu_6], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_2: "f32[4, 128, 27, 27]" = max_pool2d_with_indices_1[0]
    getitem_3: "i64[4, 128, 27, 27]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_7: "f32[4, 32, 27, 27]" = torch.ops.aten.convolution.default(getitem_2, primals_15, primals_16, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_16 = None
    relu_7: "f32[4, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_8: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, primals_17, primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_18 = None
    relu_8: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
    convolution_9: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_7, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_20 = None
    relu_9: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_2: "f32[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_8, relu_9], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_10: "f32[4, 32, 27, 27]" = torch.ops.aten.convolution.default(cat_2, primals_21, primals_22, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_22 = None
    relu_10: "f32[4, 32, 27, 27]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_11: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, primals_23, primals_24, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_24 = None
    relu_11: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    convolution_12: "f32[4, 128, 27, 27]" = torch.ops.aten.convolution.default(relu_10, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_26 = None
    relu_12: "f32[4, 128, 27, 27]" = torch.ops.aten.relu.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_3: "f32[4, 256, 27, 27]" = torch.ops.aten.cat.default([relu_11, relu_12], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_4: "f32[4, 256, 13, 13]" = max_pool2d_with_indices_2[0]
    getitem_5: "i64[4, 256, 13, 13]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_13: "f32[4, 48, 13, 13]" = torch.ops.aten.convolution.default(getitem_4, primals_27, primals_28, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_28 = None
    relu_13: "f32[4, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_14: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, primals_29, primals_30, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_30 = None
    relu_14: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
    convolution_15: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_13, primals_31, primals_32, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_32 = None
    relu_15: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_4: "f32[4, 384, 13, 13]" = torch.ops.aten.cat.default([relu_14, relu_15], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_16: "f32[4, 48, 13, 13]" = torch.ops.aten.convolution.default(cat_4, primals_33, primals_34, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_34 = None
    relu_16: "f32[4, 48, 13, 13]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_17: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, primals_35, primals_36, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_36 = None
    relu_17: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
    convolution_18: "f32[4, 192, 13, 13]" = torch.ops.aten.convolution.default(relu_16, primals_37, primals_38, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_38 = None
    relu_18: "f32[4, 192, 13, 13]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_5: "f32[4, 384, 13, 13]" = torch.ops.aten.cat.default([relu_17, relu_18], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_19: "f32[4, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_5, primals_39, primals_40, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_40 = None
    relu_19: "f32[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_20: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, primals_41, primals_42, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_42 = None
    relu_20: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
    convolution_21: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_19, primals_43, primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_44 = None
    relu_21: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_6: "f32[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_20, relu_21], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    convolution_22: "f32[4, 64, 13, 13]" = torch.ops.aten.convolution.default(cat_6, primals_45, primals_46, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_46 = None
    relu_22: "f32[4, 64, 13, 13]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    convolution_23: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, primals_47, primals_48, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_48 = None
    relu_23: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
    convolution_24: "f32[4, 256, 13, 13]" = torch.ops.aten.convolution.default(relu_22, primals_49, primals_50, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_50 = None
    relu_24: "f32[4, 256, 13, 13]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    cat_7: "f32[4, 512, 13, 13]" = torch.ops.aten.cat.default([relu_23, relu_24], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    convolution_25: "f32[4, 1000, 13, 13]" = torch.ops.aten.convolution.default(cat_7, primals_51, primals_52, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_52 = None
    relu_25: "f32[4, 1000, 13, 13]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None
    mean: "f32[4, 1000, 1, 1]" = torch.ops.aten.mean.dim(relu_25, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:97, code: return torch.flatten(x, 1)
    view: "f32[4, 1000]" = torch.ops.aten.reshape.default(mean, [4, 1000]);  mean = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    le: "b8[4, 1000, 13, 13]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    le_1: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    le_2: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    le_4: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    le_5: "b8[4, 256, 13, 13]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    le_7: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    le_8: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    le_10: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    le_11: "b8[4, 192, 13, 13]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    le_13: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    le_14: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    le_16: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    le_17: "b8[4, 128, 27, 27]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    le_19: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    le_20: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    le_22: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    le_23: "b8[4, 64, 55, 55]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    return [view, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, relu, getitem, getitem_1, relu_1, cat, relu_4, cat_1, getitem_2, getitem_3, relu_7, cat_2, relu_10, cat_3, getitem_4, getitem_5, relu_13, cat_4, relu_16, cat_5, relu_19, cat_6, relu_22, cat_7, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23]
    