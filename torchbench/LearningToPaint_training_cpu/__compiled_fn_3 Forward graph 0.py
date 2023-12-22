from __future__ import annotations



def forward(self, primals_1: "f32[64, 9, 3, 3]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64, 64, 3, 3]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64, 64, 3, 3]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[64, 64, 1, 1]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64, 64, 3, 3]", primals_14: "f32[64]", primals_15: "f32[64]", primals_16: "f32[64, 64, 3, 3]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[128, 64, 3, 3]", primals_20: "f32[128]", primals_21: "f32[128]", primals_22: "f32[128, 128, 3, 3]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[128, 64, 1, 1]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[128, 128, 3, 3]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128, 128, 3, 3]", primals_32: "f32[128]", primals_33: "f32[128]", primals_34: "f32[256, 128, 3, 3]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[256, 256, 3, 3]", primals_38: "f32[256]", primals_39: "f32[256]", primals_40: "f32[256, 128, 1, 1]", primals_41: "f32[256]", primals_42: "f32[256]", primals_43: "f32[256, 256, 3, 3]", primals_44: "f32[256]", primals_45: "f32[256]", primals_46: "f32[256, 256, 3, 3]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[512, 256, 3, 3]", primals_50: "f32[512]", primals_51: "f32[512]", primals_52: "f32[512, 512, 3, 3]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[512, 256, 1, 1]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512, 512, 3, 3]", primals_59: "f32[512]", primals_60: "f32[512]", primals_61: "f32[512, 512, 3, 3]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[65, 512]", primals_65: "f32[65]", primals_66: "f32[64]", primals_67: "f32[64]", primals_68: "i64[]", primals_69: "f32[64]", primals_70: "f32[64]", primals_71: "i64[]", primals_72: "f32[64]", primals_73: "f32[64]", primals_74: "i64[]", primals_75: "f32[64]", primals_76: "f32[64]", primals_77: "i64[]", primals_78: "f32[64]", primals_79: "f32[64]", primals_80: "i64[]", primals_81: "f32[64]", primals_82: "f32[64]", primals_83: "i64[]", primals_84: "f32[128]", primals_85: "f32[128]", primals_86: "i64[]", primals_87: "f32[128]", primals_88: "f32[128]", primals_89: "i64[]", primals_90: "f32[128]", primals_91: "f32[128]", primals_92: "i64[]", primals_93: "f32[128]", primals_94: "f32[128]", primals_95: "i64[]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "i64[]", primals_99: "f32[256]", primals_100: "f32[256]", primals_101: "i64[]", primals_102: "f32[256]", primals_103: "f32[256]", primals_104: "i64[]", primals_105: "f32[256]", primals_106: "f32[256]", primals_107: "i64[]", primals_108: "f32[256]", primals_109: "f32[256]", primals_110: "i64[]", primals_111: "f32[256]", primals_112: "f32[256]", primals_113: "i64[]", primals_114: "f32[512]", primals_115: "f32[512]", primals_116: "i64[]", primals_117: "f32[512]", primals_118: "f32[512]", primals_119: "i64[]", primals_120: "f32[512]", primals_121: "f32[512]", primals_122: "i64[]", primals_123: "f32[512]", primals_124: "f32[512]", primals_125: "i64[]", primals_126: "f32[512]", primals_127: "f32[512]", primals_128: "i64[]", primals_129: "f32[4, 9, 128, 128]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:105, code: x = F.relu(self.bn1(self.conv1(x)))
    convolution: "f32[4, 64, 64, 64]" = torch.ops.aten.convolution.default(primals_129, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_66, torch.float32)
    convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_67, torch.float32)
    add: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[4, 64, 64, 64]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_1: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu, primals_4, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_69, torch.float32)
    convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_70, torch.float32)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    relu_1: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_2: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_72, torch.float32)
    convert_element_type_5: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_73, torch.float32)
    add_4: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_3: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu, primals_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_75, torch.float32)
    convert_element_type_7: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_76, torch.float32)
    add_6: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[64]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    add_8: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(add_5, add_7);  add_5 = add_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_2: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_4: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_2, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_8: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_78, torch.float32)
    convert_element_type_9: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_79, torch.float32)
    add_9: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[64]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
    reciprocal_4: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_10: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    relu_3: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_5: "f32[4, 64, 32, 32]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_10: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_81, torch.float32)
    convert_element_type_11: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_82, torch.float32)
    add_11: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
    reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 64, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 64, 32, 32]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_12: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_13: "f32[4, 64, 32, 32]" = torch.ops.aten.add.Tensor(add_12, relu_2);  add_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_4: "f32[4, 64, 32, 32]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_6: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_12: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_84, torch.float32)
    convert_element_type_13: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_85, torch.float32)
    add_14: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[128]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_6: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_15: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    relu_5: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_7: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_14: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_87, torch.float32)
    convert_element_type_15: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_88, torch.float32)
    add_16: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[128]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_7: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_17: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_8: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_4, primals_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_16: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_90, torch.float32)
    convert_element_type_17: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_91, torch.float32)
    add_18: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[128]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_8: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_19: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    add_20: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(add_17, add_19);  add_17 = add_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_6: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_9: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_6, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_18: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_93, torch.float32)
    convert_element_type_19: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_94, torch.float32)
    add_21: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[128]" = torch.ops.aten.sqrt.default(add_21);  add_21 = None
    reciprocal_9: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_22: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    relu_7: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_10: "f32[4, 128, 16, 16]" = torch.ops.aten.convolution.default(relu_7, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_20: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_96, torch.float32)
    convert_element_type_21: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_97, torch.float32)
    add_23: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[128]" = torch.ops.aten.sqrt.default(add_23);  add_23 = None
    reciprocal_10: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_24: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_25: "f32[4, 128, 16, 16]" = torch.ops.aten.add.Tensor(add_24, relu_6);  add_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_8: "f32[4, 128, 16, 16]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_11: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_8, primals_34, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_22: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_99, torch.float32)
    convert_element_type_23: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_100, torch.float32)
    add_26: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[256]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_11: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_27: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    relu_9: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_12: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_9, primals_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_24: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_102, torch.float32)
    convert_element_type_25: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_103, torch.float32)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[256]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_12: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_29: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_13: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_8, primals_40, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_26: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_105, torch.float32)
    convert_element_type_27: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_106, torch.float32)
    add_30: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_31: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    add_32: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(add_29, add_31);  add_29 = add_31 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_10: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(add_32);  add_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_14: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_10, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_28: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_108, torch.float32)
    convert_element_type_29: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_109, torch.float32)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[256]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    reciprocal_14: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_117: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_119: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_34: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    relu_11: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_15: "f32[4, 256, 8, 8]" = torch.ops.aten.convolution.default(relu_11, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_30: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_111, torch.float32)
    convert_element_type_31: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_112, torch.float32)
    add_35: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[256]" = torch.ops.aten.sqrt.default(add_35);  add_35 = None
    reciprocal_15: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_125: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_127: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_36: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_37: "f32[4, 256, 8, 8]" = torch.ops.aten.add.Tensor(add_36, relu_10);  add_36 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_12: "f32[4, 256, 8, 8]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_16: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_12, primals_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_32: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_114, torch.float32)
    convert_element_type_33: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_115, torch.float32)
    add_38: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[512]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_16: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_133: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_135: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_39: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    relu_13: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_17: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_13, primals_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_34: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_117, torch.float32)
    convert_element_type_35: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_118, torch.float32)
    add_40: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[512]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_17: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_41: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    convolution_18: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_12, primals_55, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    convert_element_type_36: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_120, torch.float32)
    convert_element_type_37: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_121, torch.float32)
    add_42: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[512]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_18: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_149: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_151: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_43: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    add_44: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(add_41, add_43);  add_41 = add_43 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_14: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:45, code: out = F.relu(self.bn1(self.conv1(x)))
    convolution_19: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_14, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_38: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_123, torch.float32)
    convert_element_type_39: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_124, torch.float32)
    add_45: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[512]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
    reciprocal_19: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_157: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_159: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_46: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    relu_15: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:46, code: out = self.bn2(self.conv2(out))
    convolution_20: "f32[4, 512, 4, 4]" = torch.ops.aten.convolution.default(relu_15, primals_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    convert_element_type_40: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_126, torch.float32)
    convert_element_type_41: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_127, torch.float32)
    add_47: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[512]" = torch.ops.aten.sqrt.default(add_47);  add_47 = None
    reciprocal_20: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_165: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_167: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_48: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:47, code: out += self.shortcut(x)
    add_49: "f32[4, 512, 4, 4]" = torch.ops.aten.add.Tensor(add_48, relu_14);  add_48 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:48, code: out = F.relu(out)
    relu_16: "f32[4, 512, 4, 4]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:110, code: x = F.avg_pool2d(x, 4)
    avg_pool2d: "f32[4, 512, 1, 1]" = torch.ops.aten.avg_pool2d.default(relu_16, [4, 4])
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:111, code: x = x.view(x.size(0), -1)
    view: "f32[4, 512]" = torch.ops.aten.view.default(avg_pool2d, [4, -1]);  avg_pool2d = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:112, code: x = self.fc(x)
    permute: "f32[512, 65]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm: "f32[4, 65]" = torch.ops.aten.addmm.default(primals_65, view, permute);  primals_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:113, code: x = torch.sigmoid(x)
    sigmoid: "f32[4, 65]" = torch.ops.aten.sigmoid.default(addmm);  addmm = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/LearningToPaint/baseline/DRL/actor.py:112, code: x = self.fc(x)
    permute_1: "f32[65, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [sigmoid, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, convolution, relu, convolution_1, relu_1, convolution_2, convolution_3, relu_2, convolution_4, relu_3, convolution_5, relu_4, convolution_6, relu_5, convolution_7, convolution_8, relu_6, convolution_9, relu_7, convolution_10, relu_8, convolution_11, relu_9, convolution_12, convolution_13, relu_10, convolution_14, relu_11, convolution_15, relu_12, convolution_16, relu_13, convolution_17, convolution_18, relu_14, convolution_19, relu_15, convolution_20, relu_16, view, sigmoid, permute_1]
    