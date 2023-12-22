from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_3: "f32[16, 64, 1, 1]", primals_5: "f32[64, 16, 1, 1]", primals_7: "f32[64, 16, 3, 3]", primals_9: "f32[16, 128, 1, 1]", primals_11: "f32[64, 16, 1, 1]", primals_13: "f32[64, 16, 3, 3]", primals_15: "f32[32, 128, 1, 1]", primals_17: "f32[128, 32, 1, 1]", primals_19: "f32[128, 32, 3, 3]", primals_21: "f32[32, 256, 1, 1]", primals_23: "f32[128, 32, 1, 1]", primals_25: "f32[128, 32, 3, 3]", primals_27: "f32[48, 256, 1, 1]", primals_29: "f32[192, 48, 1, 1]", primals_31: "f32[192, 48, 3, 3]", primals_33: "f32[48, 384, 1, 1]", primals_35: "f32[192, 48, 1, 1]", primals_37: "f32[192, 48, 3, 3]", primals_39: "f32[64, 384, 1, 1]", primals_41: "f32[256, 64, 1, 1]", primals_43: "f32[256, 64, 3, 3]", primals_45: "f32[64, 512, 1, 1]", primals_47: "f32[256, 64, 1, 1]", primals_49: "f32[256, 64, 3, 3]", primals_51: "f32[1000, 512, 1, 1]", primals_53: "f32[4, 3, 224, 224]", relu: "f32[4, 64, 111, 111]", getitem: "f32[4, 64, 55, 55]", getitem_1: "i64[4, 64, 55, 55]", relu_1: "f32[4, 16, 55, 55]", cat: "f32[4, 128, 55, 55]", relu_4: "f32[4, 16, 55, 55]", cat_1: "f32[4, 128, 55, 55]", getitem_2: "f32[4, 128, 27, 27]", getitem_3: "i64[4, 128, 27, 27]", relu_7: "f32[4, 32, 27, 27]", cat_2: "f32[4, 256, 27, 27]", relu_10: "f32[4, 32, 27, 27]", cat_3: "f32[4, 256, 27, 27]", getitem_4: "f32[4, 256, 13, 13]", getitem_5: "i64[4, 256, 13, 13]", relu_13: "f32[4, 48, 13, 13]", cat_4: "f32[4, 384, 13, 13]", relu_16: "f32[4, 48, 13, 13]", cat_5: "f32[4, 384, 13, 13]", relu_19: "f32[4, 64, 13, 13]", cat_6: "f32[4, 512, 13, 13]", relu_22: "f32[4, 64, 13, 13]", clone: "f32[4, 512, 13, 13]", le: "b8[4, 1000, 13, 13]", le_1: "b8[4, 256, 13, 13]", le_2: "b8[4, 256, 13, 13]", le_4: "b8[4, 256, 13, 13]", le_5: "b8[4, 256, 13, 13]", le_7: "b8[4, 192, 13, 13]", le_8: "b8[4, 192, 13, 13]", le_10: "b8[4, 192, 13, 13]", le_11: "b8[4, 192, 13, 13]", le_13: "b8[4, 128, 27, 27]", le_14: "b8[4, 128, 27, 27]", le_16: "b8[4, 128, 27, 27]", le_17: "b8[4, 128, 27, 27]", le_19: "b8[4, 64, 55, 55]", le_20: "b8[4, 64, 55, 55]", le_22: "b8[4, 64, 55, 55]", le_23: "b8[4, 64, 55, 55]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:97, code: return torch.flatten(x, 1)
    view_1: "f32[4, 1000, 1, 1]" = torch.ops.aten.view.default(tangents_1, [4, 1000, 1, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:96, code: x = self.classifier(x)
    expand: "f32[4, 1000, 13, 13]" = torch.ops.aten.expand.default(view_1, [4, 1000, 13, 13]);  view_1 = None
    div: "f32[4, 1000, 13, 13]" = torch.ops.aten.div.Scalar(expand, 169);  expand = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[4, 1000, 13, 13]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    convolution_backward = torch.ops.aten.convolution_backward.default(where, clone, primals_51, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where = clone = primals_51 = None
    getitem_6: "f32[4, 512, 13, 13]" = convolution_backward[0]
    getitem_7: "f32[1000, 512, 1, 1]" = convolution_backward[1]
    getitem_8: "f32[1000]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_1: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 0, 256)
    slice_2: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 256, 512);  getitem_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_1: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_1, full_default, slice_2);  le_1 = slice_2 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(where_1, relu_22, primals_49, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_1 = primals_49 = None
    getitem_9: "f32[4, 64, 13, 13]" = convolution_backward_1[0]
    getitem_10: "f32[256, 64, 3, 3]" = convolution_backward_1[1]
    getitem_11: "f32[256]" = convolution_backward_1[2];  convolution_backward_1 = None
    where_2: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_2, full_default, slice_1);  le_2 = slice_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_2, relu_22, primals_47, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_2 = primals_47 = None
    getitem_12: "f32[4, 64, 13, 13]" = convolution_backward_2[0]
    getitem_13: "f32[256, 64, 1, 1]" = convolution_backward_2[1]
    getitem_14: "f32[256]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add: "f32[4, 64, 13, 13]" = torch.ops.aten.add.Tensor(getitem_9, getitem_12);  getitem_9 = getitem_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_36: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_37: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_3: "b8[4, 64, 13, 13]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    where_3: "f32[4, 64, 13, 13]" = torch.ops.aten.where.self(le_3, full_default, add);  le_3 = add = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_3, cat_6, primals_45, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_3 = cat_6 = primals_45 = None
    getitem_15: "f32[4, 512, 13, 13]" = convolution_backward_3[0]
    getitem_16: "f32[64, 512, 1, 1]" = convolution_backward_3[1]
    getitem_17: "f32[64]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_3: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_15, 1, 0, 256)
    slice_4: "f32[4, 256, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_15, 1, 256, 512);  getitem_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_4: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_4, full_default, slice_4);  le_4 = slice_4 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_4, relu_19, primals_43, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = primals_43 = None
    getitem_18: "f32[4, 64, 13, 13]" = convolution_backward_4[0]
    getitem_19: "f32[256, 64, 3, 3]" = convolution_backward_4[1]
    getitem_20: "f32[256]" = convolution_backward_4[2];  convolution_backward_4 = None
    where_5: "f32[4, 256, 13, 13]" = torch.ops.aten.where.self(le_5, full_default, slice_3);  le_5 = slice_3 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_5, relu_19, primals_41, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_5 = primals_41 = None
    getitem_21: "f32[4, 64, 13, 13]" = convolution_backward_5[0]
    getitem_22: "f32[256, 64, 1, 1]" = convolution_backward_5[1]
    getitem_23: "f32[256]" = convolution_backward_5[2];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_1: "f32[4, 64, 13, 13]" = torch.ops.aten.add.Tensor(getitem_18, getitem_21);  getitem_18 = getitem_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_45: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_46: "f32[4, 64, 13, 13]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_6: "b8[4, 64, 13, 13]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    where_6: "f32[4, 64, 13, 13]" = torch.ops.aten.where.self(le_6, full_default, add_1);  le_6 = add_1 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(where_6, cat_5, primals_39, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_6 = cat_5 = primals_39 = None
    getitem_24: "f32[4, 384, 13, 13]" = convolution_backward_6[0]
    getitem_25: "f32[64, 384, 1, 1]" = convolution_backward_6[1]
    getitem_26: "f32[64]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_5: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 0, 192)
    slice_6: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 192, 384);  getitem_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_7: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_7, full_default, slice_6);  le_7 = slice_6 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_7, relu_16, primals_37, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_7 = primals_37 = None
    getitem_27: "f32[4, 48, 13, 13]" = convolution_backward_7[0]
    getitem_28: "f32[192, 48, 3, 3]" = convolution_backward_7[1]
    getitem_29: "f32[192]" = convolution_backward_7[2];  convolution_backward_7 = None
    where_8: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_8, full_default, slice_5);  le_8 = slice_5 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_8, relu_16, primals_35, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_8 = primals_35 = None
    getitem_30: "f32[4, 48, 13, 13]" = convolution_backward_8[0]
    getitem_31: "f32[192, 48, 1, 1]" = convolution_backward_8[1]
    getitem_32: "f32[192]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_2: "f32[4, 48, 13, 13]" = torch.ops.aten.add.Tensor(getitem_27, getitem_30);  getitem_27 = getitem_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_54: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_55: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_9: "b8[4, 48, 13, 13]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    where_9: "f32[4, 48, 13, 13]" = torch.ops.aten.where.self(le_9, full_default, add_2);  le_9 = add_2 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_9, cat_4, primals_33, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_9 = cat_4 = primals_33 = None
    getitem_33: "f32[4, 384, 13, 13]" = convolution_backward_9[0]
    getitem_34: "f32[48, 384, 1, 1]" = convolution_backward_9[1]
    getitem_35: "f32[48]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_7: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_33, 1, 0, 192)
    slice_8: "f32[4, 192, 13, 13]" = torch.ops.aten.slice.Tensor(getitem_33, 1, 192, 384);  getitem_33 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_10: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_10, full_default, slice_8);  le_10 = slice_8 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(where_10, relu_13, primals_31, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_10 = primals_31 = None
    getitem_36: "f32[4, 48, 13, 13]" = convolution_backward_10[0]
    getitem_37: "f32[192, 48, 3, 3]" = convolution_backward_10[1]
    getitem_38: "f32[192]" = convolution_backward_10[2];  convolution_backward_10 = None
    where_11: "f32[4, 192, 13, 13]" = torch.ops.aten.where.self(le_11, full_default, slice_7);  le_11 = slice_7 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(where_11, relu_13, primals_29, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_11 = primals_29 = None
    getitem_39: "f32[4, 48, 13, 13]" = convolution_backward_11[0]
    getitem_40: "f32[192, 48, 1, 1]" = convolution_backward_11[1]
    getitem_41: "f32[192]" = convolution_backward_11[2];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_3: "f32[4, 48, 13, 13]" = torch.ops.aten.add.Tensor(getitem_36, getitem_39);  getitem_36 = getitem_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_63: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_64: "f32[4, 48, 13, 13]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_12: "b8[4, 48, 13, 13]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    where_12: "f32[4, 48, 13, 13]" = torch.ops.aten.where.self(le_12, full_default, add_3);  le_12 = add_3 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_12, getitem_4, primals_27, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_12 = getitem_4 = primals_27 = None
    getitem_42: "f32[4, 256, 13, 13]" = convolution_backward_12[0]
    getitem_43: "f32[48, 256, 1, 1]" = convolution_backward_12[1]
    getitem_44: "f32[48]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_backward: "f32[4, 256, 27, 27]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_42, cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5);  getitem_42 = cat_3 = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_9: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward, 1, 0, 128)
    slice_10: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward, 1, 128, 256);  max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_13: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_13, full_default, slice_10);  le_13 = slice_10 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_13, relu_10, primals_25, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_13 = primals_25 = None
    getitem_45: "f32[4, 32, 27, 27]" = convolution_backward_13[0]
    getitem_46: "f32[128, 32, 3, 3]" = convolution_backward_13[1]
    getitem_47: "f32[128]" = convolution_backward_13[2];  convolution_backward_13 = None
    where_14: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_14, full_default, slice_9);  le_14 = slice_9 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_14, relu_10, primals_23, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_14 = primals_23 = None
    getitem_48: "f32[4, 32, 27, 27]" = convolution_backward_14[0]
    getitem_49: "f32[128, 32, 1, 1]" = convolution_backward_14[1]
    getitem_50: "f32[128]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_4: "f32[4, 32, 27, 27]" = torch.ops.aten.add.Tensor(getitem_45, getitem_48);  getitem_45 = getitem_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_72: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_73: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_15: "b8[4, 32, 27, 27]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    where_15: "f32[4, 32, 27, 27]" = torch.ops.aten.where.self(le_15, full_default, add_4);  le_15 = add_4 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_15, cat_2, primals_21, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_15 = cat_2 = primals_21 = None
    getitem_51: "f32[4, 256, 27, 27]" = convolution_backward_15[0]
    getitem_52: "f32[32, 256, 1, 1]" = convolution_backward_15[1]
    getitem_53: "f32[32]" = convolution_backward_15[2];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_11: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(getitem_51, 1, 0, 128)
    slice_12: "f32[4, 128, 27, 27]" = torch.ops.aten.slice.Tensor(getitem_51, 1, 128, 256);  getitem_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_16: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_16, full_default, slice_12);  le_16 = slice_12 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(where_16, relu_7, primals_19, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_16 = primals_19 = None
    getitem_54: "f32[4, 32, 27, 27]" = convolution_backward_16[0]
    getitem_55: "f32[128, 32, 3, 3]" = convolution_backward_16[1]
    getitem_56: "f32[128]" = convolution_backward_16[2];  convolution_backward_16 = None
    where_17: "f32[4, 128, 27, 27]" = torch.ops.aten.where.self(le_17, full_default, slice_11);  le_17 = slice_11 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_17, relu_7, primals_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_17 = primals_17 = None
    getitem_57: "f32[4, 32, 27, 27]" = convolution_backward_17[0]
    getitem_58: "f32[128, 32, 1, 1]" = convolution_backward_17[1]
    getitem_59: "f32[128]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_5: "f32[4, 32, 27, 27]" = torch.ops.aten.add.Tensor(getitem_54, getitem_57);  getitem_54 = getitem_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_81: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_82: "f32[4, 32, 27, 27]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_18: "b8[4, 32, 27, 27]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    where_18: "f32[4, 32, 27, 27]" = torch.ops.aten.where.self(le_18, full_default, add_5);  le_18 = add_5 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_18, getitem_2, primals_15, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_18 = getitem_2 = primals_15 = None
    getitem_60: "f32[4, 128, 27, 27]" = convolution_backward_18[0]
    getitem_61: "f32[32, 128, 1, 1]" = convolution_backward_18[1]
    getitem_62: "f32[32]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_backward_1: "f32[4, 128, 55, 55]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_60, cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3);  getitem_60 = cat_1 = getitem_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_13: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward_1, 1, 0, 64)
    slice_14: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(max_pool2d_with_indices_backward_1, 1, 64, 128);  max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_19: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_19, full_default, slice_14);  le_19 = slice_14 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_19, relu_4, primals_13, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_19 = primals_13 = None
    getitem_63: "f32[4, 16, 55, 55]" = convolution_backward_19[0]
    getitem_64: "f32[64, 16, 3, 3]" = convolution_backward_19[1]
    getitem_65: "f32[64]" = convolution_backward_19[2];  convolution_backward_19 = None
    where_20: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_20, full_default, slice_13);  le_20 = slice_13 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(where_20, relu_4, primals_11, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_20 = primals_11 = None
    getitem_66: "f32[4, 16, 55, 55]" = convolution_backward_20[0]
    getitem_67: "f32[64, 16, 1, 1]" = convolution_backward_20[1]
    getitem_68: "f32[64]" = convolution_backward_20[2];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_6: "f32[4, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_63, getitem_66);  getitem_63 = getitem_66 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_90: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_91: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_21: "b8[4, 16, 55, 55]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    where_21: "f32[4, 16, 55, 55]" = torch.ops.aten.where.self(le_21, full_default, add_6);  le_21 = add_6 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_21, cat, primals_9, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_21 = cat = primals_9 = None
    getitem_69: "f32[4, 128, 55, 55]" = convolution_backward_21[0]
    getitem_70: "f32[16, 128, 1, 1]" = convolution_backward_21[1]
    getitem_71: "f32[16]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:31, code: return torch.cat(
    slice_15: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 0, 64)
    slice_16: "f32[4, 64, 55, 55]" = torch.ops.aten.slice.Tensor(getitem_69, 1, 64, 128);  getitem_69 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    where_22: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_22, full_default, slice_16);  le_22 = slice_16 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_22, relu_1, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]);  where_22 = primals_7 = None
    getitem_72: "f32[4, 16, 55, 55]" = convolution_backward_22[0]
    getitem_73: "f32[64, 16, 3, 3]" = convolution_backward_22[1]
    getitem_74: "f32[64]" = convolution_backward_22[2];  convolution_backward_22 = None
    where_23: "f32[4, 64, 55, 55]" = torch.ops.aten.where.self(le_23, full_default, slice_15);  le_23 = slice_15 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_23, relu_1, primals_5, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_23 = primals_5 = None
    getitem_75: "f32[4, 16, 55, 55]" = convolution_backward_23[0]
    getitem_76: "f32[64, 16, 1, 1]" = convolution_backward_23[1]
    getitem_77: "f32[64]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:32, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
    add_7: "f32[4, 16, 55, 55]" = torch.ops.aten.add.Tensor(getitem_72, getitem_75);  getitem_72 = getitem_75 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:30, code: x = self.squeeze_activation(self.squeeze(x))
    alias_99: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_100: "f32[4, 16, 55, 55]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_24: "b8[4, 16, 55, 55]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    where_24: "f32[4, 16, 55, 55]" = torch.ops.aten.where.self(le_24, full_default, add_7);  le_24 = add_7 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_24, getitem, primals_3, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_24 = getitem = primals_3 = None
    getitem_78: "f32[4, 64, 55, 55]" = convolution_backward_24[0]
    getitem_79: "f32[16, 64, 1, 1]" = convolution_backward_24[1]
    getitem_80: "f32[16]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/squeezenet.py:95, code: x = self.features(x)
    max_pool2d_with_indices_backward_2: "f32[4, 64, 111, 111]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_78, relu, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1);  getitem_78 = getitem_1 = None
    alias_102: "f32[4, 64, 111, 111]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_103: "f32[4, 64, 111, 111]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_25: "b8[4, 64, 111, 111]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    where_25: "f32[4, 64, 111, 111]" = torch.ops.aten.where.self(le_25, full_default, max_pool2d_with_indices_backward_2);  le_25 = full_default = max_pool2d_with_indices_backward_2 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(where_25, primals_53, primals_1, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  where_25 = primals_53 = primals_1 = None
    getitem_82: "f32[64, 3, 3, 3]" = convolution_backward_25[1]
    getitem_83: "f32[64]" = convolution_backward_25[2];  convolution_backward_25 = None
    return [getitem_82, getitem_83, getitem_79, getitem_80, getitem_76, getitem_77, getitem_73, getitem_74, getitem_70, getitem_71, getitem_67, getitem_68, getitem_64, getitem_65, getitem_61, getitem_62, getitem_58, getitem_59, getitem_55, getitem_56, getitem_52, getitem_53, getitem_49, getitem_50, getitem_46, getitem_47, getitem_43, getitem_44, getitem_40, getitem_41, getitem_37, getitem_38, getitem_34, getitem_35, getitem_31, getitem_32, getitem_28, getitem_29, getitem_25, getitem_26, getitem_22, getitem_23, getitem_19, getitem_20, getitem_16, getitem_17, getitem_13, getitem_14, getitem_10, getitem_11, getitem_7, getitem_8, None]
    