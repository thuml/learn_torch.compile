from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 7, 7]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64, 64, 3, 3]"; primals_5: "f32[64]"; primals_6: "f32[64]"; primals_7: "f32[64, 64, 3, 3]"; primals_8: "f32[64]"; primals_9: "f32[64]"; primals_10: "f32[64, 64, 3, 3]"; primals_11: "f32[64]"; primals_12: "f32[64]"; primals_13: "f32[64, 64, 3, 3]"; primals_14: "f32[64]"; primals_15: "f32[64]"; primals_16: "f32[128, 64, 3, 3]"; primals_17: "f32[128]"; primals_18: "f32[128]"; primals_19: "f32[128, 128, 3, 3]"; primals_20: "f32[128]"; primals_21: "f32[128]"; primals_22: "f32[128, 64, 1, 1]"; primals_23: "f32[128]"; primals_24: "f32[128]"; primals_25: "f32[128, 128, 3, 3]"; primals_26: "f32[128]"; primals_27: "f32[128]"; primals_28: "f32[128, 128, 3, 3]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[256, 128, 3, 3]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[256, 256, 3, 3]"; primals_35: "f32[256]"; primals_36: "f32[256]"; primals_37: "f32[256, 128, 1, 1]"; primals_38: "f32[256]"; primals_39: "f32[256]"; primals_40: "f32[256, 256, 3, 3]"; primals_41: "f32[256]"; primals_42: "f32[256]"; primals_43: "f32[256, 256, 3, 3]"; primals_44: "f32[256]"; primals_45: "f32[256]"; primals_46: "f32[512, 256, 3, 3]"; primals_47: "f32[512]"; primals_48: "f32[512]"; primals_49: "f32[512, 512, 3, 3]"; primals_50: "f32[512]"; primals_51: "f32[512]"; primals_52: "f32[512, 256, 1, 1]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[512, 512, 3, 3]"; primals_56: "f32[512]"; primals_57: "f32[512]"; primals_58: "f32[512, 512, 3, 3]"; primals_59: "f32[512]"; primals_60: "f32[512]"; primals_61: "f32[1000, 512]"; primals_62: "f32[1000]"; primals_63: "f32[64]"; primals_64: "f32[64]"; primals_65: "i64[]"; primals_66: "f32[64]"; primals_67: "f32[64]"; primals_68: "i64[]"; primals_69: "f32[64]"; primals_70: "f32[64]"; primals_71: "i64[]"; primals_72: "f32[64]"; primals_73: "f32[64]"; primals_74: "i64[]"; primals_75: "f32[64]"; primals_76: "f32[64]"; primals_77: "i64[]"; primals_78: "f32[128]"; primals_79: "f32[128]"; primals_80: "i64[]"; primals_81: "f32[128]"; primals_82: "f32[128]"; primals_83: "i64[]"; primals_84: "f32[128]"; primals_85: "f32[128]"; primals_86: "i64[]"; primals_87: "f32[128]"; primals_88: "f32[128]"; primals_89: "i64[]"; primals_90: "f32[128]"; primals_91: "f32[128]"; primals_92: "i64[]"; primals_93: "f32[256]"; primals_94: "f32[256]"; primals_95: "i64[]"; primals_96: "f32[256]"; primals_97: "f32[256]"; primals_98: "i64[]"; primals_99: "f32[256]"; primals_100: "f32[256]"; primals_101: "i64[]"; primals_102: "f32[256]"; primals_103: "f32[256]"; primals_104: "i64[]"; primals_105: "f32[256]"; primals_106: "f32[256]"; primals_107: "i64[]"; primals_108: "f32[512]"; primals_109: "f32[512]"; primals_110: "i64[]"; primals_111: "f32[512]"; primals_112: "f32[512]"; primals_113: "i64[]"; primals_114: "f32[512]"; primals_115: "f32[512]"; primals_116: "i64[]"; primals_117: "f32[512]"; primals_118: "f32[512]"; primals_119: "i64[]"; primals_120: "f32[512]"; primals_121: "f32[512]"; primals_122: "i64[]"; primals_123: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_123, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_63, torch.float32)
    convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_64, torch.float32)
    add: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    relu: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
    getitem: "f32[4, 64, 56, 56]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_1: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_66, torch.float32)
    convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_67, torch.float32)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_1: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_2: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_69, torch.float32)
    convert_element_type_5: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_70, torch.float32)
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_6: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_5, getitem);  add_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_2: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_3: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_72, torch.float32)
    convert_element_type_7: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_73, torch.float32)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_7);  add_7 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_8: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_3: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_4: "f32[4, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_8: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_75, torch.float32)
    convert_element_type_9: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_76, torch.float32)
    add_9: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_10: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_11: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_10, relu_2);  add_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_4: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_5: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_4, primals_16, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_10: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_78, torch.float32)
    convert_element_type_11: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_79, torch.float32)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_13: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_5: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_6: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_12: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_81, torch.float32)
    convert_element_type_13: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_82, torch.float32)
    add_14: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[128]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_6: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_15: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    convolution_7: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_4, primals_22, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_14: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_84, torch.float32)
    convert_element_type_15: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_85, torch.float32)
    add_16: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[128]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_7: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_17: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_18: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_15, add_17);  add_15 = add_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_6: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_18);  add_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_8: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_16: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_87, torch.float32)
    convert_element_type_17: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_88, torch.float32)
    add_19: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[128]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    reciprocal_8: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_20: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_7: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_9: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_18: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_90, torch.float32)
    convert_element_type_19: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_91, torch.float32)
    add_21: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[128]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_9: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_22: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_23: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(add_22, relu_6);  add_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_8: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_10: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_8, primals_31, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_20: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_93, torch.float32)
    convert_element_type_21: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_94, torch.float32)
    add_24: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_25: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_9: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_11: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_9, primals_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_22: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_96, torch.float32)
    convert_element_type_23: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_97, torch.float32)
    add_26: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_27: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    convolution_12: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_8, primals_37, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_24: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_99, torch.float32)
    convert_element_type_25: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_100, torch.float32)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[256]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_12: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_29: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_30: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_27, add_29);  add_27 = add_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_10: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_13: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_10, primals_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_26: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_102, torch.float32)
    convert_element_type_27: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_103, torch.float32)
    add_31: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
    reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_32: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_11: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_14: "f32[4, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_11, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_28: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_105, torch.float32)
    convert_element_type_29: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_106, torch.float32)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[256]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_14: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_117: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_119: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_34: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_35: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(add_34, relu_10);  add_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_12: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_15: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_12, primals_46, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_30: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_108, torch.float32)
    convert_element_type_31: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_109, torch.float32)
    add_36: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[512]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_15: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_125: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_127: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_37: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_13: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_16: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_32: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_111, torch.float32)
    convert_element_type_33: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_112, torch.float32)
    add_38: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[512]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_16: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_133: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_135: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_39: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    convolution_17: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_12, primals_52, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_34: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_114, torch.float32)
    convert_element_type_35: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_115, torch.float32)
    add_40: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[512]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_17: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_41: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_42: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_39, add_41);  add_39 = add_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_14: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_42);  add_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_18: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_14, primals_55, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    convert_element_type_36: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_117, torch.float32)
    convert_element_type_37: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_118, torch.float32)
    add_43: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[512]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
    reciprocal_18: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_149: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_151: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_44: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    relu_15: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_19: "f32[4, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_15, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    convert_element_type_38: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_120, torch.float32)
    convert_element_type_39: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_121, torch.float32)
    add_45: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[512]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_19: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_157: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_159: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_46: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:102, code: out += identity
    add_47: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(add_46, relu_14);  add_46 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    relu_16: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    mean: "f32[4, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_16, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view: "f32[4, 512]" = torch.ops.aten.view.default(mean, [4, 512]);  mean = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:280, code: x = self.fc(x)
    permute: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_62, view, permute);  primals_62 = None
    permute_1: "f32[1000, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 512]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:279, code: x = torch.flatten(x, 1)
    view_2: "f32[4, 512, 1, 1]" = torch.ops.aten.view.default(mm, [4, 512, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:278, code: x = self.avgpool(x)
    expand: "f32[4, 512, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 512, 7, 7]);  view_2 = None
    div: "f32[4, 512, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_18: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_19: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    le: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_19, 0);  alias_19 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_48: "f32[512]" = torch.ops.aten.add.Tensor(primals_121, 1e-05);  primals_121 = None
    rsqrt: "f32[512]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    unsqueeze_160: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_120, 0);  primals_120 = None
    unsqueeze_161: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    sum_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_20: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_162);  convolution_19 = unsqueeze_162 = None
    mul_60: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_20);  sub_20 = None
    sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_60, [0, 2, 3]);  mul_60 = None
    mul_65: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt, primals_59);  primals_59 = None
    unsqueeze_169: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_65, 0);  mul_65 = None
    unsqueeze_170: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    mul_66: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_171);  unsqueeze_171 = None
    mul_67: "f32[512]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_66, relu_15, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_66 = primals_58 = None
    getitem_2: "f32[4, 512, 7, 7]" = convolution_backward[0]
    getitem_3: "f32[512, 512, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_21: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_22: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    le_1: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_22, 0);  alias_22 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_2);  le_1 = scalar_tensor_1 = getitem_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_49: "f32[512]" = torch.ops.aten.add.Tensor(primals_118, 1e-05);  primals_118 = None
    rsqrt_1: "f32[512]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    unsqueeze_172: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_117, 0);  primals_117 = None
    unsqueeze_173: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_21: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_174);  convolution_18 = unsqueeze_174 = None
    mul_68: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_21);  sub_21 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_68, [0, 2, 3]);  mul_68 = None
    mul_73: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_56);  primals_56 = None
    unsqueeze_181: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_73, 0);  mul_73 = None
    unsqueeze_182: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    mul_74: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_183);  where_1 = unsqueeze_183 = None
    mul_75: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_74, relu_14, primals_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_74 = primals_55 = None
    getitem_5: "f32[4, 512, 7, 7]" = convolution_backward_1[0]
    getitem_6: "f32[512, 512, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_50: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_5);  where = getitem_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_24: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_25: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    le_2: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_25, 0);  alias_25 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, add_50);  le_2 = scalar_tensor_2 = add_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    add_51: "f32[512]" = torch.ops.aten.add.Tensor(primals_115, 1e-05);  primals_115 = None
    rsqrt_2: "f32[512]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    unsqueeze_184: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_114, 0);  primals_114 = None
    unsqueeze_185: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_22: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_186);  convolution_17 = unsqueeze_186 = None
    mul_76: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_22);  sub_22 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_76, [0, 2, 3]);  mul_76 = None
    mul_81: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_53);  primals_53 = None
    unsqueeze_193: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_81, 0);  mul_81 = None
    unsqueeze_194: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    mul_82: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_195);  unsqueeze_195 = None
    mul_83: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_82, relu_12, primals_52, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_82 = primals_52 = None
    getitem_8: "f32[4, 256, 14, 14]" = convolution_backward_2[0]
    getitem_9: "f32[512, 256, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_52: "f32[512]" = torch.ops.aten.add.Tensor(primals_112, 1e-05);  primals_112 = None
    rsqrt_3: "f32[512]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    unsqueeze_196: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_111, 0);  primals_111 = None
    unsqueeze_197: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_23: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_198);  convolution_16 = unsqueeze_198 = None
    mul_84: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_23);  sub_23 = None
    sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_84, [0, 2, 3]);  mul_84 = None
    mul_89: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_50);  primals_50 = None
    unsqueeze_205: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_89, 0);  mul_89 = None
    unsqueeze_206: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    mul_90: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_207);  where_2 = unsqueeze_207 = None
    mul_91: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_90, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_90 = primals_49 = None
    getitem_11: "f32[4, 512, 7, 7]" = convolution_backward_3[0]
    getitem_12: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_27: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_28: "f32[4, 512, 7, 7]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    le_3: "b8[4, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_28, 0);  alias_28 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 512, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_11);  le_3 = scalar_tensor_3 = getitem_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_53: "f32[512]" = torch.ops.aten.add.Tensor(primals_109, 1e-05);  primals_109 = None
    rsqrt_4: "f32[512]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    unsqueeze_208: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_108, 0);  primals_108 = None
    unsqueeze_209: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_24: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_210);  convolution_15 = unsqueeze_210 = None
    mul_92: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_24);  sub_24 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 2, 3]);  mul_92 = None
    mul_97: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_47);  primals_47 = None
    unsqueeze_217: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_97, 0);  mul_97 = None
    unsqueeze_218: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    mul_98: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_219);  where_3 = unsqueeze_219 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_98, relu_12, primals_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_98 = primals_46 = None
    getitem_14: "f32[4, 256, 14, 14]" = convolution_backward_4[0]
    getitem_15: "f32[512, 256, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_54: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(getitem_8, getitem_14);  getitem_8 = getitem_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_30: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_31: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    le_4: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_31, 0);  alias_31 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, add_54);  le_4 = scalar_tensor_4 = add_54 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_55: "f32[256]" = torch.ops.aten.add.Tensor(primals_106, 1e-05);  primals_106 = None
    rsqrt_5: "f32[256]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    unsqueeze_220: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_105, 0);  primals_105 = None
    unsqueeze_221: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_25: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_222);  convolution_14 = unsqueeze_222 = None
    mul_100: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_25);  sub_25 = None
    sum_13: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 2, 3]);  mul_100 = None
    mul_105: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_44);  primals_44 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_105, 0);  mul_105 = None
    unsqueeze_230: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    mul_106: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_231);  unsqueeze_231 = None
    mul_107: "f32[256]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_106, relu_11, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_106 = primals_43 = None
    getitem_17: "f32[4, 256, 14, 14]" = convolution_backward_5[0]
    getitem_18: "f32[256, 256, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_33: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_34: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    le_5: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_34, 0);  alias_34 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_17);  le_5 = scalar_tensor_5 = getitem_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_56: "f32[256]" = torch.ops.aten.add.Tensor(primals_103, 1e-05);  primals_103 = None
    rsqrt_6: "f32[256]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    unsqueeze_232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_102, 0);  primals_102 = None
    unsqueeze_233: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_26: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_234);  convolution_13 = unsqueeze_234 = None
    mul_108: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_26);  sub_26 = None
    sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_108, [0, 2, 3]);  mul_108 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_41);  primals_41 = None
    unsqueeze_241: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_113, 0);  mul_113 = None
    unsqueeze_242: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_114: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_243);  where_5 = unsqueeze_243 = None
    mul_115: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_114, relu_10, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_114 = primals_40 = None
    getitem_20: "f32[4, 256, 14, 14]" = convolution_backward_6[0]
    getitem_21: "f32[256, 256, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_57: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(where_4, getitem_20);  where_4 = getitem_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_36: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_37: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_6: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_37, 0);  alias_37 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_57);  le_6 = scalar_tensor_6 = add_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    add_58: "f32[256]" = torch.ops.aten.add.Tensor(primals_100, 1e-05);  primals_100 = None
    rsqrt_7: "f32[256]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    unsqueeze_244: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_99, 0);  primals_99 = None
    unsqueeze_245: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_27: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_246);  convolution_12 = unsqueeze_246 = None
    mul_116: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_27);  sub_27 = None
    sum_17: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 2, 3]);  mul_116 = None
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_38);  primals_38 = None
    unsqueeze_253: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_121, 0);  mul_121 = None
    unsqueeze_254: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_122: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_255);  unsqueeze_255 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_122, relu_8, primals_37, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_122 = primals_37 = None
    getitem_23: "f32[4, 128, 28, 28]" = convolution_backward_7[0]
    getitem_24: "f32[256, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_59: "f32[256]" = torch.ops.aten.add.Tensor(primals_97, 1e-05);  primals_97 = None
    rsqrt_8: "f32[256]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    unsqueeze_256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_96, 0);  primals_96 = None
    unsqueeze_257: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_18: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_28: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_258);  convolution_11 = unsqueeze_258 = None
    mul_124: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_28);  sub_28 = None
    sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 2, 3]);  mul_124 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_35);  primals_35 = None
    unsqueeze_265: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_129, 0);  mul_129 = None
    unsqueeze_266: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_130: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_267);  where_6 = unsqueeze_267 = None
    mul_131: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_130, relu_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_130 = primals_34 = None
    getitem_26: "f32[4, 256, 14, 14]" = convolution_backward_8[0]
    getitem_27: "f32[256, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_39: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_40: "f32[4, 256, 14, 14]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    le_7: "b8[4, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_40, 0);  alias_40 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[4, 256, 14, 14]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_26);  le_7 = scalar_tensor_7 = getitem_26 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_60: "f32[256]" = torch.ops.aten.add.Tensor(primals_94, 1e-05);  primals_94 = None
    rsqrt_9: "f32[256]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    unsqueeze_268: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_93, 0);  primals_93 = None
    unsqueeze_269: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_20: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_29: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_270);  convolution_10 = unsqueeze_270 = None
    mul_132: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_29);  sub_29 = None
    sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 2, 3]);  mul_132 = None
    mul_137: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_32);  primals_32 = None
    unsqueeze_277: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_137, 0);  mul_137 = None
    unsqueeze_278: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_138: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_279);  where_7 = unsqueeze_279 = None
    mul_139: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_138, relu_8, primals_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_138 = primals_31 = None
    getitem_29: "f32[4, 128, 28, 28]" = convolution_backward_9[0]
    getitem_30: "f32[256, 128, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_61: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(getitem_23, getitem_29);  getitem_23 = getitem_29 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_42: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_43: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_8: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_43, 0);  alias_43 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, add_61);  le_8 = scalar_tensor_8 = add_61 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_62: "f32[128]" = torch.ops.aten.add.Tensor(primals_91, 1e-05);  primals_91 = None
    rsqrt_10: "f32[128]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    unsqueeze_280: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_90, 0);  primals_90 = None
    unsqueeze_281: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_30: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_282);  convolution_9 = unsqueeze_282 = None
    mul_140: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, sub_30);  sub_30 = None
    sum_23: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 2, 3]);  mul_140 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_29);  primals_29 = None
    unsqueeze_289: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_145, 0);  mul_145 = None
    unsqueeze_290: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_146: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_291);  unsqueeze_291 = None
    mul_147: "f32[128]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_146, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_146 = primals_28 = None
    getitem_32: "f32[4, 128, 28, 28]" = convolution_backward_10[0]
    getitem_33: "f32[128, 128, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_45: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_46: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    le_9: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_46, 0);  alias_46 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_32);  le_9 = scalar_tensor_9 = getitem_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_63: "f32[128]" = torch.ops.aten.add.Tensor(primals_88, 1e-05);  primals_88 = None
    rsqrt_11: "f32[128]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    unsqueeze_292: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_87, 0);  primals_87 = None
    unsqueeze_293: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_24: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_31: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_294);  convolution_8 = unsqueeze_294 = None
    mul_148: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_9, sub_31);  sub_31 = None
    sum_25: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 2, 3]);  mul_148 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_26);  primals_26 = None
    unsqueeze_301: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_153, 0);  mul_153 = None
    unsqueeze_302: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_154: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_303);  where_9 = unsqueeze_303 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_154, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_154 = primals_25 = None
    getitem_35: "f32[4, 128, 28, 28]" = convolution_backward_11[0]
    getitem_36: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_64: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(where_8, getitem_35);  where_8 = getitem_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_48: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_49: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_10: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, add_64);  le_10 = scalar_tensor_10 = add_64 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:100, code: identity = self.downsample(x)
    add_65: "f32[128]" = torch.ops.aten.add.Tensor(primals_85, 1e-05);  primals_85 = None
    rsqrt_12: "f32[128]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    unsqueeze_304: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_84, 0);  primals_84 = None
    unsqueeze_305: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_26: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_32: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_306);  convolution_7 = unsqueeze_306 = None
    mul_156: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, sub_32);  sub_32 = None
    sum_27: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 2, 3]);  mul_156 = None
    mul_161: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_23);  primals_23 = None
    unsqueeze_313: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_161, 0);  mul_161 = None
    unsqueeze_314: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_162: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_315);  unsqueeze_315 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_162, relu_4, primals_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_162 = primals_22 = None
    getitem_38: "f32[4, 64, 56, 56]" = convolution_backward_12[0]
    getitem_39: "f32[128, 64, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(primals_82, 1e-05);  primals_82 = None
    rsqrt_13: "f32[128]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    unsqueeze_316: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_81, 0);  primals_81 = None
    unsqueeze_317: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_28: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_33: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_318);  convolution_6 = unsqueeze_318 = None
    mul_164: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, sub_33);  sub_33 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_164, [0, 2, 3]);  mul_164 = None
    mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_20);  primals_20 = None
    unsqueeze_325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_169, 0);  mul_169 = None
    unsqueeze_326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_170: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_327);  where_10 = unsqueeze_327 = None
    mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_170, relu_5, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_170 = primals_19 = None
    getitem_41: "f32[4, 128, 28, 28]" = convolution_backward_13[0]
    getitem_42: "f32[128, 128, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_51: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_52: "f32[4, 128, 28, 28]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_11: "b8[4, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[4, 128, 28, 28]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_41);  le_11 = scalar_tensor_11 = getitem_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_67: "f32[128]" = torch.ops.aten.add.Tensor(primals_79, 1e-05);  primals_79 = None
    rsqrt_14: "f32[128]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    unsqueeze_328: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_78, 0);  primals_78 = None
    unsqueeze_329: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_34: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_330);  convolution_5 = unsqueeze_330 = None
    mul_172: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_11, sub_34);  sub_34 = None
    sum_31: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 2, 3]);  mul_172 = None
    mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_17);  primals_17 = None
    unsqueeze_337: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
    unsqueeze_338: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_178: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_339);  where_11 = unsqueeze_339 = None
    mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_178, relu_4, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_178 = primals_16 = None
    getitem_44: "f32[4, 64, 56, 56]" = convolution_backward_14[0]
    getitem_45: "f32[128, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_68: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_38, getitem_44);  getitem_38 = getitem_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_54: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_55: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_12: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_68);  le_12 = scalar_tensor_12 = add_68 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_69: "f32[64]" = torch.ops.aten.add.Tensor(primals_76, 1e-05);  primals_76 = None
    rsqrt_15: "f32[64]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    unsqueeze_340: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_75, 0);  primals_75 = None
    unsqueeze_341: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_32: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_35: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_342);  convolution_4 = unsqueeze_342 = None
    mul_180: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, sub_35);  sub_35 = None
    sum_33: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 2, 3]);  mul_180 = None
    mul_185: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_14);  primals_14 = None
    unsqueeze_349: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_185, 0);  mul_185 = None
    unsqueeze_350: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_186: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_351);  unsqueeze_351 = None
    mul_187: "f32[64]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_186, relu_3, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_186 = primals_13 = None
    getitem_47: "f32[4, 64, 56, 56]" = convolution_backward_15[0]
    getitem_48: "f32[64, 64, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_57: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_58: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_13: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_47);  le_13 = scalar_tensor_13 = getitem_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_70: "f32[64]" = torch.ops.aten.add.Tensor(primals_73, 1e-05);  primals_73 = None
    rsqrt_16: "f32[64]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    unsqueeze_352: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_72, 0);  primals_72 = None
    unsqueeze_353: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_34: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_36: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_354);  convolution_3 = unsqueeze_354 = None
    mul_188: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_13, sub_36);  sub_36 = None
    sum_35: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3]);  mul_188 = None
    mul_193: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_11);  primals_11 = None
    unsqueeze_361: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_193, 0);  mul_193 = None
    unsqueeze_362: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_194: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_363);  where_13 = unsqueeze_363 = None
    mul_195: "f32[64]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_194, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_194 = primals_10 = None
    getitem_50: "f32[4, 64, 56, 56]" = convolution_backward_16[0]
    getitem_51: "f32[64, 64, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_71: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(where_12, getitem_50);  where_12 = getitem_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:103, code: out = self.relu(out)
    alias_60: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_61: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_14: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, add_71);  le_14 = scalar_tensor_14 = add_71 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:97, code: out = self.bn2(out)
    add_72: "f32[64]" = torch.ops.aten.add.Tensor(primals_70, 1e-05);  primals_70 = None
    rsqrt_17: "f32[64]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    unsqueeze_364: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_69, 0);  primals_69 = None
    unsqueeze_365: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_36: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_37: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_366);  convolution_2 = unsqueeze_366 = None
    mul_196: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, sub_37);  sub_37 = None
    sum_37: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 2, 3]);  mul_196 = None
    mul_201: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_8);  primals_8 = None
    unsqueeze_373: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_201, 0);  mul_201 = None
    unsqueeze_374: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_202: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_375);  unsqueeze_375 = None
    mul_203: "f32[64]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:96, code: out = self.conv2(out)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_202, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_202 = primals_7 = None
    getitem_53: "f32[4, 64, 56, 56]" = convolution_backward_17[0]
    getitem_54: "f32[64, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:94, code: out = self.relu(out)
    alias_63: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_64: "f32[4, 64, 56, 56]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_15: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, getitem_53);  le_15 = scalar_tensor_15 = getitem_53 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:93, code: out = self.bn1(out)
    add_73: "f32[64]" = torch.ops.aten.add.Tensor(primals_67, 1e-05);  primals_67 = None
    rsqrt_18: "f32[64]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    unsqueeze_376: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_66, 0);  primals_66 = None
    unsqueeze_377: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_38: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_38: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_378);  convolution_1 = unsqueeze_378 = None
    mul_204: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, sub_38);  sub_38 = None
    sum_39: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 2, 3]);  mul_204 = None
    mul_209: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_5);  primals_5 = None
    unsqueeze_385: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_209, 0);  mul_209 = None
    unsqueeze_386: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_210: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_387);  where_15 = unsqueeze_387 = None
    mul_211: "f32[64]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_210, getitem, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = getitem = primals_4 = None
    getitem_56: "f32[4, 64, 56, 56]" = convolution_backward_18[0]
    getitem_57: "f32[64, 64, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:92, code: out = self.conv1(x)
    add_74: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(where_14, getitem_56);  where_14 = getitem_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:271, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[4, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_74, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_1);  add_74 = getitem_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:270, code: x = self.relu(x)
    alias_66: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_67: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_16: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, max_pool2d_with_indices_backward);  le_16 = scalar_tensor_16 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:269, code: x = self.bn1(x)
    add_75: "f32[64]" = torch.ops.aten.add.Tensor(primals_64, 1e-05);  primals_64 = None
    rsqrt_19: "f32[64]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    unsqueeze_388: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_63, 0);  primals_63 = None
    unsqueeze_389: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_40: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_39: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_390);  convolution = unsqueeze_390 = None
    mul_212: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, sub_39);  sub_39 = None
    sum_41: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
    mul_217: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_2);  primals_2 = None
    unsqueeze_397: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
    unsqueeze_398: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_218: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_399);  where_16 = unsqueeze_399 = None
    mul_219: "f32[64]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/resnet.py:268, code: x = self.conv1(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_218, primals_123, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_218 = primals_123 = primals_1 = None
    getitem_60: "f32[64, 3, 7, 7]" = convolution_backward_19[1];  convolution_backward_19 = None
    return pytree.tree_unflatten([addmm, getitem_60, mul_219, sum_40, getitem_57, mul_211, sum_38, getitem_54, mul_203, sum_36, getitem_51, mul_195, sum_34, getitem_48, mul_187, sum_32, getitem_45, mul_179, sum_30, getitem_42, mul_171, sum_28, getitem_39, mul_163, sum_26, getitem_36, mul_155, sum_24, getitem_33, mul_147, sum_22, getitem_30, mul_139, sum_20, getitem_27, mul_131, sum_18, getitem_24, mul_123, sum_16, getitem_21, mul_115, sum_14, getitem_18, mul_107, sum_12, getitem_15, mul_99, sum_10, getitem_12, mul_91, sum_8, getitem_9, mul_83, sum_6, getitem_6, mul_75, sum_4, getitem_3, mul_67, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    