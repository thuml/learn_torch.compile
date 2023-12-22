from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[64]"; primals_5: "f32[128]"; primals_6: "f32[128]"; primals_7: "f32[128]"; primals_8: "f32[128]"; primals_9: "f32[128]"; primals_10: "f32[128]"; primals_11: "f32[128]"; primals_12: "f32[128]"; primals_13: "f32[128]"; primals_14: "f32[128]"; primals_15: "f32[128]"; primals_16: "f32[128]"; primals_17: "f32[256]"; primals_18: "f32[256]"; primals_19: "f32[160]"; primals_20: "f32[160]"; primals_21: "f32[160]"; primals_22: "f32[160]"; primals_23: "f32[160]"; primals_24: "f32[160]"; primals_25: "f32[160]"; primals_26: "f32[160]"; primals_27: "f32[160]"; primals_28: "f32[160]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[192]"; primals_32: "f32[192]"; primals_33: "f32[192]"; primals_34: "f32[192]"; primals_35: "f32[192]"; primals_36: "f32[192]"; primals_37: "f32[192]"; primals_38: "f32[192]"; primals_39: "f32[192]"; primals_40: "f32[192]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[192]"; primals_44: "f32[192]"; primals_45: "f32[192]"; primals_46: "f32[192]"; primals_47: "f32[192]"; primals_48: "f32[192]"; primals_49: "f32[192]"; primals_50: "f32[192]"; primals_51: "f32[192]"; primals_52: "f32[192]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[224]"; primals_56: "f32[224]"; primals_57: "f32[224]"; primals_58: "f32[224]"; primals_59: "f32[224]"; primals_60: "f32[224]"; primals_61: "f32[224]"; primals_62: "f32[224]"; primals_63: "f32[224]"; primals_64: "f32[224]"; primals_65: "f32[1024]"; primals_66: "f32[1024]"; primals_67: "f32[224]"; primals_68: "f32[224]"; primals_69: "f32[224]"; primals_70: "f32[224]"; primals_71: "f32[224]"; primals_72: "f32[224]"; primals_73: "f32[224]"; primals_74: "f32[224]"; primals_75: "f32[224]"; primals_76: "f32[224]"; primals_77: "f32[1024]"; primals_78: "f32[1024]"; primals_79: "f32[64, 3, 3, 3]"; primals_80: "f32[64, 64, 3, 3]"; primals_81: "f32[128, 64, 3, 3]"; primals_82: "f32[128, 128, 3, 3]"; primals_83: "f32[128, 128, 3, 3]"; primals_84: "f32[128, 128, 3, 3]"; primals_85: "f32[128, 128, 3, 3]"; primals_86: "f32[128, 128, 3, 3]"; primals_87: "f32[256, 768, 1, 1]"; primals_88: "f32[160, 256, 3, 3]"; primals_89: "f32[160, 160, 3, 3]"; primals_90: "f32[160, 160, 3, 3]"; primals_91: "f32[160, 160, 3, 3]"; primals_92: "f32[160, 160, 3, 3]"; primals_93: "f32[512, 1056, 1, 1]"; primals_94: "f32[192, 512, 3, 3]"; primals_95: "f32[192, 192, 3, 3]"; primals_96: "f32[192, 192, 3, 3]"; primals_97: "f32[192, 192, 3, 3]"; primals_98: "f32[192, 192, 3, 3]"; primals_99: "f32[768, 1472, 1, 1]"; primals_100: "f32[192, 768, 3, 3]"; primals_101: "f32[192, 192, 3, 3]"; primals_102: "f32[192, 192, 3, 3]"; primals_103: "f32[192, 192, 3, 3]"; primals_104: "f32[192, 192, 3, 3]"; primals_105: "f32[768, 1728, 1, 1]"; primals_106: "f32[224, 768, 3, 3]"; primals_107: "f32[224, 224, 3, 3]"; primals_108: "f32[224, 224, 3, 3]"; primals_109: "f32[224, 224, 3, 3]"; primals_110: "f32[224, 224, 3, 3]"; primals_111: "f32[1024, 1888, 1, 1]"; primals_112: "f32[224, 1024, 3, 3]"; primals_113: "f32[224, 224, 3, 3]"; primals_114: "f32[224, 224, 3, 3]"; primals_115: "f32[224, 224, 3, 3]"; primals_116: "f32[224, 224, 3, 3]"; primals_117: "f32[1024, 2144, 1, 1]"; primals_118: "f32[1000, 1024]"; primals_119: "f32[1000]"; primals_120: "f32[64]"; primals_121: "f32[64]"; primals_122: "f32[64]"; primals_123: "f32[64]"; primals_124: "f32[128]"; primals_125: "f32[128]"; primals_126: "f32[128]"; primals_127: "f32[128]"; primals_128: "f32[128]"; primals_129: "f32[128]"; primals_130: "f32[128]"; primals_131: "f32[128]"; primals_132: "f32[128]"; primals_133: "f32[128]"; primals_134: "f32[128]"; primals_135: "f32[128]"; primals_136: "f32[256]"; primals_137: "f32[256]"; primals_138: "f32[160]"; primals_139: "f32[160]"; primals_140: "f32[160]"; primals_141: "f32[160]"; primals_142: "f32[160]"; primals_143: "f32[160]"; primals_144: "f32[160]"; primals_145: "f32[160]"; primals_146: "f32[160]"; primals_147: "f32[160]"; primals_148: "f32[512]"; primals_149: "f32[512]"; primals_150: "f32[192]"; primals_151: "f32[192]"; primals_152: "f32[192]"; primals_153: "f32[192]"; primals_154: "f32[192]"; primals_155: "f32[192]"; primals_156: "f32[192]"; primals_157: "f32[192]"; primals_158: "f32[192]"; primals_159: "f32[192]"; primals_160: "f32[768]"; primals_161: "f32[768]"; primals_162: "f32[192]"; primals_163: "f32[192]"; primals_164: "f32[192]"; primals_165: "f32[192]"; primals_166: "f32[192]"; primals_167: "f32[192]"; primals_168: "f32[192]"; primals_169: "f32[192]"; primals_170: "f32[192]"; primals_171: "f32[192]"; primals_172: "f32[768]"; primals_173: "f32[768]"; primals_174: "f32[224]"; primals_175: "f32[224]"; primals_176: "f32[224]"; primals_177: "f32[224]"; primals_178: "f32[224]"; primals_179: "f32[224]"; primals_180: "f32[224]"; primals_181: "f32[224]"; primals_182: "f32[224]"; primals_183: "f32[224]"; primals_184: "f32[1024]"; primals_185: "f32[1024]"; primals_186: "f32[224]"; primals_187: "f32[224]"; primals_188: "f32[224]"; primals_189: "f32[224]"; primals_190: "f32[224]"; primals_191: "f32[224]"; primals_192: "f32[224]"; primals_193: "f32[224]"; primals_194: "f32[224]"; primals_195: "f32[224]"; primals_196: "f32[1024]"; primals_197: "f32[1024]"; primals_198: "f32[4, 3, 224, 224]"; tangents_1: "f32[4, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_198, primals_79, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_120, torch.float32)
    convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_121, torch.float32)
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
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_80, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_122, torch.float32)
    convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_123, torch.float32)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_81, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_4: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_124, torch.float32)
    convert_element_type_5: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_125, torch.float32)
    add_4: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_82, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_6: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_126, torch.float32)
    convert_element_type_7: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_127, torch.float32)
    add_6: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[128]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_83, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_8: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_128, torch.float32)
    convert_element_type_9: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_129, torch.float32)
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[128]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_84, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_10: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_130, torch.float32)
    convert_element_type_11: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_131, torch.float32)
    add_10: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_12: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_132, torch.float32)
    convert_element_type_13: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_133, torch.float32)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[128]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_86, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_14: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_134, torch.float32)
    convert_element_type_15: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_135, torch.float32)
    add_14: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[128]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat: "f32[4, 768, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_3, relu_4, relu_5, relu_6, relu_7], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[4, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_87, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_16: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_136, torch.float32)
    convert_element_type_17: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_137, torch.float32)
    add_16: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[256]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_8, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem: "f32[4, 256, 28, 28]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 256, 28, 28]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[4, 160, 28, 28]" = torch.ops.aten.convolution.default(getitem, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_18: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_138, torch.float32)
    convert_element_type_19: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_139, torch.float32)
    add_18: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[160]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_9: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_77: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_79: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_19: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[4, 160, 28, 28]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[4, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_20: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_140, torch.float32)
    convert_element_type_21: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_141, torch.float32)
    add_20: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[160]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_10: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_85: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_87: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_21: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[4, 160, 28, 28]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[4, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_90, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_22: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_142, torch.float32)
    convert_element_type_23: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_143, torch.float32)
    add_22: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[160]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_11: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_93: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_95: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_23: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[4, 160, 28, 28]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[4, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_24: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_144, torch.float32)
    convert_element_type_25: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_145, torch.float32)
    add_24: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[160]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_12: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_101: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_103: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_25: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[4, 160, 28, 28]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[4, 160, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_26: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_146, torch.float32)
    convert_element_type_27: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_147, torch.float32)
    add_26: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[160]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_13: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_109: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_111: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_27: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[4, 160, 28, 28]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_1: "f32[4, 1056, 28, 28]" = torch.ops.aten.cat.default([getitem, relu_9, relu_10, relu_11, relu_12, relu_13], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[4, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_1, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_28: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_148, torch.float32)
    convert_element_type_29: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_149, torch.float32)
    add_28: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[512]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_14: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  unsqueeze_113 = None
    mul_43: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_29: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_14, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_2: "f32[4, 512, 14, 14]" = max_pool2d_with_indices_1[0]
    getitem_3: "i64[4, 512, 14, 14]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(getitem_2, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_30: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_150, torch.float32)
    convert_element_type_31: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_151, torch.float32)
    add_30: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[192]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_15: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_125: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_127: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_31: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_95, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_32: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_152, torch.float32)
    convert_element_type_33: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_153, torch.float32)
    add_32: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[192]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_16: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_133: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_135: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_33: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_16, primals_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_34: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_154, torch.float32)
    convert_element_type_35: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_155, torch.float32)
    add_34: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[192]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_17: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_141: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_143: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_35: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_36: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_156, torch.float32)
    convert_element_type_37: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_157, torch.float32)
    add_36: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[192]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_18: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_149: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_151: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_37: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_18, primals_98, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_38: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_158, torch.float32)
    convert_element_type_39: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_159, torch.float32)
    add_38: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[192]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_19: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_157: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_159: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_39: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_2: "f32[4, 1472, 14, 14]" = torch.ops.aten.cat.default([getitem_2, relu_15, relu_16, relu_17, relu_18, relu_19], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[4, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_2, primals_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_40: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_160, torch.float32)
    convert_element_type_41: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_161, torch.float32)
    add_40: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[768]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_20: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_165: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_167: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_41: "f32[4, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[4, 768, 14, 14]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_42: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_162, torch.float32)
    convert_element_type_43: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_163, torch.float32)
    add_42: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[192]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_21: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  unsqueeze_169 = None
    mul_64: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_173: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_175: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_43: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_43);  add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_101, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_44: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_164, torch.float32)
    convert_element_type_45: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_165, torch.float32)
    add_44: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[192]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_22: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  unsqueeze_177 = None
    mul_67: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_181: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_183: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_45: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_46: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_166, torch.float32)
    convert_element_type_47: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_167, torch.float32)
    add_46: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[192]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_23: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  unsqueeze_185 = None
    mul_70: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_189: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_191: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_47: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_23: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_103, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_48: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_168, torch.float32)
    convert_element_type_49: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_169, torch.float32)
    add_48: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[192]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_24: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  unsqueeze_193 = None
    mul_73: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_197: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_199: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_49: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_24: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[4, 192, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_104, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_50: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_170, torch.float32)
    convert_element_type_51: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_171, torch.float32)
    add_50: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[192]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_25: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  unsqueeze_201 = None
    mul_76: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_205: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_207: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_51: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[4, 192, 14, 14]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_3: "f32[4, 1728, 14, 14]" = torch.ops.aten.cat.default([relu_20, relu_21, relu_22, relu_23, relu_24, relu_25], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[4, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_3, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_52: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_172, torch.float32)
    convert_element_type_53: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_173, torch.float32)
    add_52: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[768]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_26: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  unsqueeze_209 = None
    mul_79: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_213: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_215: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_53: "f32[4, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[4, 768, 14, 14]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(relu_26, [3, 3], [2, 2], [0, 0], [1, 1], True)
    getitem_4: "f32[4, 768, 7, 7]" = max_pool2d_with_indices_2[0]
    getitem_5: "i64[4, 768, 7, 7]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(getitem_4, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_54: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_174, torch.float32)
    convert_element_type_55: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_175, torch.float32)
    add_54: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[224]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_27: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  unsqueeze_217 = None
    mul_82: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_221: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_223: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_55: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_27: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_27, primals_107, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_56: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_176, torch.float32)
    convert_element_type_57: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_177, torch.float32)
    add_56: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[224]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_28: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  unsqueeze_225 = None
    mul_85: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_229: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_231: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_57: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_28: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_108, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_58: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_178, torch.float32)
    convert_element_type_59: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_179, torch.float32)
    add_58: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[224]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_29: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  unsqueeze_233 = None
    mul_88: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_237: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_239: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_59: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_59);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_29, primals_109, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_60: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_180, torch.float32)
    convert_element_type_61: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_181, torch.float32)
    add_60: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[224]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_30: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  unsqueeze_241 = None
    mul_91: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_245: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_247: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_61: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_61);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_30, primals_110, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_62: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_182, torch.float32)
    convert_element_type_63: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_183, torch.float32)
    add_62: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[224]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_31: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  unsqueeze_249 = None
    mul_94: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_253: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_255: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_63: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_31: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_4: "f32[4, 1888, 7, 7]" = torch.ops.aten.cat.default([getitem_4, relu_27, relu_28, relu_29, relu_30, relu_31], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[4, 1024, 7, 7]" = torch.ops.aten.convolution.default(cat_4, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_64: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_184, torch.float32)
    convert_element_type_65: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_185, torch.float32)
    add_64: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[1024]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_32: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  unsqueeze_257 = None
    mul_97: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_261: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_263: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_65: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_32: "f32[4, 1024, 7, 7]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_112, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_66: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_186, torch.float32)
    convert_element_type_67: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_187, torch.float32)
    add_66: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[224]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_33: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  unsqueeze_265 = None
    mul_100: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_269: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_271: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_67: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_113, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_68: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_188, torch.float32)
    convert_element_type_69: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_189, torch.float32)
    add_68: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[224]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_34: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  unsqueeze_273 = None
    mul_103: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_277: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_279: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_69: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_34, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_70: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_190, torch.float32)
    convert_element_type_71: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_191, torch.float32)
    add_70: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[224]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_35: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  unsqueeze_281 = None
    mul_106: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_285: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_287: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_71: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_35: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_36: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_35, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_72: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_192, torch.float32)
    convert_element_type_73: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_193, torch.float32)
    add_72: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[224]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_36: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  unsqueeze_289 = None
    mul_109: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_293: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_295: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_73: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_36: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_37: "f32[4, 224, 7, 7]" = torch.ops.aten.convolution.default(relu_36, primals_116, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_74: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_194, torch.float32)
    convert_element_type_75: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_195, torch.float32)
    add_74: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[224]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_37: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  unsqueeze_297 = None
    mul_112: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_301: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_303: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_75: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[4, 224, 7, 7]" = torch.ops.aten.relu.default(add_75);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    cat_5: "f32[4, 2144, 7, 7]" = torch.ops.aten.cat.default([relu_32, relu_33, relu_34, relu_35, relu_36, relu_37], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[4, 1024, 7, 7]" = torch.ops.aten.convolution.default(cat_5, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    convert_element_type_76: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_196, torch.float32)
    convert_element_type_77: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_197, torch.float32)
    add_76: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[1024]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_38: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  unsqueeze_305 = None
    mul_115: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_309: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_311: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_77: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[4, 1024, 7, 7]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[4, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[4, 1024]" = torch.ops.aten.view.default(mean, [4, 1024]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[4, 1024]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_119, clone, permute);  primals_119 = None
    permute_1: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[4, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[4, 1024, 1, 1]" = torch.ops.aten.view.default(mm, [4, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 1024, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 1024, 7, 7]);  view_2 = None
    div: "f32[4, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_40: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_41: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    le: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_41, 0);  alias_41 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_78: "f32[1024]" = torch.ops.aten.add.Tensor(primals_197, 1e-05);  primals_197 = None
    rsqrt: "f32[1024]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    unsqueeze_312: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_196, 0);  primals_196 = None
    unsqueeze_313: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_39: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_314);  convolution_38 = unsqueeze_314 = None
    mul_117: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_39);  sub_39 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_117, [0, 2, 3]);  mul_117 = None
    mul_122: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt, primals_77);  primals_77 = None
    unsqueeze_321: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_122, 0);  mul_122 = None
    unsqueeze_322: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    mul_123: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_323);  where = unsqueeze_323 = None
    mul_124: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_123, cat_5, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_123 = cat_5 = primals_117 = None
    getitem_6: "f32[4, 2144, 7, 7]" = convolution_backward[0]
    getitem_7: "f32[1024, 2144, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_1: "f32[4, 1024, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 0, 1024)
    slice_2: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 1024, 1248)
    slice_3: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 1248, 1472)
    slice_4: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 1472, 1696)
    slice_5: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 1696, 1920)
    slice_6: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_6, 1, 1920, 2144);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_43: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_44: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le_1: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_44, 0);  alias_44 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, slice_6);  le_1 = scalar_tensor_1 = slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_79: "f32[224]" = torch.ops.aten.add.Tensor(primals_195, 1e-05);  primals_195 = None
    rsqrt_1: "f32[224]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    unsqueeze_324: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_194, 0);  primals_194 = None
    unsqueeze_325: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    sum_4: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_40: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_326);  convolution_37 = unsqueeze_326 = None
    mul_125: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_40);  sub_40 = None
    sum_5: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 2, 3]);  mul_125 = None
    mul_130: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_75);  primals_75 = None
    unsqueeze_333: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_130, 0);  mul_130 = None
    unsqueeze_334: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    mul_131: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze_335);  where_1 = unsqueeze_335 = None
    mul_132: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_131, relu_36, primals_116, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_131 = primals_116 = None
    getitem_9: "f32[4, 224, 7, 7]" = convolution_backward_1[0]
    getitem_10: "f32[224, 224, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_80: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_5, getitem_9);  slice_5 = getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_46: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_47: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_2: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, add_80);  le_2 = scalar_tensor_2 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_81: "f32[224]" = torch.ops.aten.add.Tensor(primals_193, 1e-05);  primals_193 = None
    rsqrt_2: "f32[224]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    unsqueeze_336: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_192, 0);  primals_192 = None
    unsqueeze_337: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    sum_6: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_41: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_338);  convolution_36 = unsqueeze_338 = None
    mul_133: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_41);  sub_41 = None
    sum_7: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_133, [0, 2, 3]);  mul_133 = None
    mul_138: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_73);  primals_73 = None
    unsqueeze_345: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_138, 0);  mul_138 = None
    unsqueeze_346: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    mul_139: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_347);  where_2 = unsqueeze_347 = None
    mul_140: "f32[224]" = torch.ops.aten.mul.Tensor(sum_7, rsqrt_2);  sum_7 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_139, relu_35, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_139 = primals_115 = None
    getitem_12: "f32[4, 224, 7, 7]" = convolution_backward_2[0]
    getitem_13: "f32[224, 224, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_82: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_4, getitem_12);  slice_4 = getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_49: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_50: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_3: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_50, 0);  alias_50 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, add_82);  le_3 = scalar_tensor_3 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_83: "f32[224]" = torch.ops.aten.add.Tensor(primals_191, 1e-05);  primals_191 = None
    rsqrt_3: "f32[224]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    unsqueeze_348: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_190, 0);  primals_190 = None
    unsqueeze_349: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    sum_8: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_42: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_350);  convolution_35 = unsqueeze_350 = None
    mul_141: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_42);  sub_42 = None
    sum_9: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 2, 3]);  mul_141 = None
    mul_146: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_71);  primals_71 = None
    unsqueeze_357: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_146, 0);  mul_146 = None
    unsqueeze_358: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_147: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_359);  where_3 = unsqueeze_359 = None
    mul_148: "f32[224]" = torch.ops.aten.mul.Tensor(sum_9, rsqrt_3);  sum_9 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_147, relu_34, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_147 = primals_114 = None
    getitem_15: "f32[4, 224, 7, 7]" = convolution_backward_3[0]
    getitem_16: "f32[224, 224, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_84: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_15);  slice_3 = getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_52: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_53: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    le_4: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_53, 0);  alias_53 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, add_84);  le_4 = scalar_tensor_4 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_85: "f32[224]" = torch.ops.aten.add.Tensor(primals_189, 1e-05);  primals_189 = None
    rsqrt_4: "f32[224]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    unsqueeze_360: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_188, 0);  primals_188 = None
    unsqueeze_361: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    sum_10: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_43: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_362);  convolution_34 = unsqueeze_362 = None
    mul_149: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_43);  sub_43 = None
    sum_11: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_149, [0, 2, 3]);  mul_149 = None
    mul_154: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_69);  primals_69 = None
    unsqueeze_369: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_154, 0);  mul_154 = None
    unsqueeze_370: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_155: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_371);  where_4 = unsqueeze_371 = None
    mul_156: "f32[224]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_4);  sum_11 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_155, relu_33, primals_113, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_155 = primals_113 = None
    getitem_18: "f32[4, 224, 7, 7]" = convolution_backward_4[0]
    getitem_19: "f32[224, 224, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_86: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_18);  slice_2 = getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_55: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_56: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_5: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_56, 0);  alias_56 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, add_86);  le_5 = scalar_tensor_5 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_87: "f32[224]" = torch.ops.aten.add.Tensor(primals_187, 1e-05);  primals_187 = None
    rsqrt_5: "f32[224]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    unsqueeze_372: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_186, 0);  primals_186 = None
    unsqueeze_373: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    sum_12: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_44: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_374);  convolution_33 = unsqueeze_374 = None
    mul_157: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_44);  sub_44 = None
    sum_13: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 2, 3]);  mul_157 = None
    mul_162: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_67);  primals_67 = None
    unsqueeze_381: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_162, 0);  mul_162 = None
    unsqueeze_382: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    mul_163: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, unsqueeze_383);  where_5 = unsqueeze_383 = None
    mul_164: "f32[224]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_5);  sum_13 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_163, relu_32, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_163 = primals_112 = None
    getitem_21: "f32[4, 1024, 7, 7]" = convolution_backward_5[0]
    getitem_22: "f32[224, 1024, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_88: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_21);  slice_1 = getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_58: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_59: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_6: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 1024, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_88);  le_6 = scalar_tensor_6 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_89: "f32[1024]" = torch.ops.aten.add.Tensor(primals_185, 1e-05);  primals_185 = None
    rsqrt_6: "f32[1024]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    unsqueeze_384: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(primals_184, 0);  primals_184 = None
    unsqueeze_385: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    sum_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_45: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_386);  convolution_32 = unsqueeze_386 = None
    mul_165: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_45);  sub_45 = None
    sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_165, [0, 2, 3]);  mul_165 = None
    mul_170: "f32[1024]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_65);  primals_65 = None
    unsqueeze_393: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_170, 0);  mul_170 = None
    unsqueeze_394: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    mul_171: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_395);  where_6 = unsqueeze_395 = None
    mul_172: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_6);  sum_15 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_171, cat_4, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_171 = cat_4 = primals_111 = None
    getitem_24: "f32[4, 1888, 7, 7]" = convolution_backward_6[0]
    getitem_25: "f32[1024, 1888, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_7: "f32[4, 768, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 0, 768)
    slice_8: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 768, 992)
    slice_9: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 992, 1216)
    slice_10: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 1216, 1440)
    slice_11: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 1440, 1664)
    slice_12: "f32[4, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_24, 1, 1664, 1888);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_61: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_62: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    le_7: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_62, 0);  alias_62 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, slice_12);  le_7 = scalar_tensor_7 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_90: "f32[224]" = torch.ops.aten.add.Tensor(primals_183, 1e-05);  primals_183 = None
    rsqrt_7: "f32[224]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    unsqueeze_396: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_182, 0);  primals_182 = None
    unsqueeze_397: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    sum_16: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_46: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_398);  convolution_31 = unsqueeze_398 = None
    mul_173: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_46);  sub_46 = None
    sum_17: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 2, 3]);  mul_173 = None
    mul_178: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_63);  primals_63 = None
    unsqueeze_405: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_178, 0);  mul_178 = None
    unsqueeze_406: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_179: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_407);  where_7 = unsqueeze_407 = None
    mul_180: "f32[224]" = torch.ops.aten.mul.Tensor(sum_17, rsqrt_7);  sum_17 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_179, relu_30, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_179 = primals_110 = None
    getitem_27: "f32[4, 224, 7, 7]" = convolution_backward_7[0]
    getitem_28: "f32[224, 224, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_91: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_11, getitem_27);  slice_11 = getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_64: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_65: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    le_8: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_65, 0);  alias_65 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, add_91);  le_8 = scalar_tensor_8 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_92: "f32[224]" = torch.ops.aten.add.Tensor(primals_181, 1e-05);  primals_181 = None
    rsqrt_8: "f32[224]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    unsqueeze_408: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_180, 0);  primals_180 = None
    unsqueeze_409: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    sum_18: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_47: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_410);  convolution_30 = unsqueeze_410 = None
    mul_181: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_47);  sub_47 = None
    sum_19: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 2, 3]);  mul_181 = None
    mul_186: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_61);  primals_61 = None
    unsqueeze_417: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_186, 0);  mul_186 = None
    unsqueeze_418: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_187: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_419);  where_8 = unsqueeze_419 = None
    mul_188: "f32[224]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_8);  sum_19 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_187, relu_29, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_187 = primals_109 = None
    getitem_30: "f32[4, 224, 7, 7]" = convolution_backward_8[0]
    getitem_31: "f32[224, 224, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_93: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_10, getitem_30);  slice_10 = getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_67: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_68: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    le_9: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_68, 0);  alias_68 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, add_93);  le_9 = scalar_tensor_9 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_94: "f32[224]" = torch.ops.aten.add.Tensor(primals_179, 1e-05);  primals_179 = None
    rsqrt_9: "f32[224]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    unsqueeze_420: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_178, 0);  primals_178 = None
    unsqueeze_421: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    sum_20: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_48: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_422);  convolution_29 = unsqueeze_422 = None
    mul_189: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_48);  sub_48 = None
    sum_21: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3]);  mul_189 = None
    mul_194: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_59);  primals_59 = None
    unsqueeze_429: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_194, 0);  mul_194 = None
    unsqueeze_430: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_195: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_431);  where_9 = unsqueeze_431 = None
    mul_196: "f32[224]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_9);  sum_21 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_195, relu_28, primals_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_195 = primals_108 = None
    getitem_33: "f32[4, 224, 7, 7]" = convolution_backward_9[0]
    getitem_34: "f32[224, 224, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_95: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_9, getitem_33);  slice_9 = getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_70: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_71: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_10: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, add_95);  le_10 = scalar_tensor_10 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_96: "f32[224]" = torch.ops.aten.add.Tensor(primals_177, 1e-05);  primals_177 = None
    rsqrt_10: "f32[224]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    unsqueeze_432: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_176, 0);  primals_176 = None
    unsqueeze_433: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    sum_22: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_49: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_434);  convolution_28 = unsqueeze_434 = None
    mul_197: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_49);  sub_49 = None
    sum_23: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 2, 3]);  mul_197 = None
    mul_202: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_57);  primals_57 = None
    unsqueeze_441: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_202, 0);  mul_202 = None
    unsqueeze_442: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    mul_203: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_443);  where_10 = unsqueeze_443 = None
    mul_204: "f32[224]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_10);  sum_23 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_203, relu_27, primals_107, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_203 = primals_107 = None
    getitem_36: "f32[4, 224, 7, 7]" = convolution_backward_10[0]
    getitem_37: "f32[224, 224, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_97: "f32[4, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_8, getitem_36);  slice_8 = getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_73: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_74: "f32[4, 224, 7, 7]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    le_11: "b8[4, 224, 7, 7]" = torch.ops.aten.le.Scalar(alias_74, 0);  alias_74 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[4, 224, 7, 7]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, add_97);  le_11 = scalar_tensor_11 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_98: "f32[224]" = torch.ops.aten.add.Tensor(primals_175, 1e-05);  primals_175 = None
    rsqrt_11: "f32[224]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    unsqueeze_444: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_174, 0);  primals_174 = None
    unsqueeze_445: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
    unsqueeze_446: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
    sum_24: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_50: "f32[4, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_446);  convolution_27 = unsqueeze_446 = None
    mul_205: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_50);  sub_50 = None
    sum_25: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 2, 3]);  mul_205 = None
    mul_210: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_55);  primals_55 = None
    unsqueeze_453: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_210, 0);  mul_210 = None
    unsqueeze_454: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    mul_211: "f32[4, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_455);  where_11 = unsqueeze_455 = None
    mul_212: "f32[224]" = torch.ops.aten.mul.Tensor(sum_25, rsqrt_11);  sum_25 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_211, getitem_4, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = getitem_4 = primals_106 = None
    getitem_39: "f32[4, 768, 7, 7]" = convolution_backward_11[0]
    getitem_40: "f32[224, 768, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_99: "f32[4, 768, 7, 7]" = torch.ops.aten.add.Tensor(slice_7, getitem_39);  slice_7 = getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward: "f32[4, 768, 14, 14]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_99, relu_26, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5);  add_99 = getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_76: "f32[4, 768, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_77: "f32[4, 768, 14, 14]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    le_12: "b8[4, 768, 14, 14]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[4, 768, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, max_pool2d_with_indices_backward);  le_12 = scalar_tensor_12 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_100: "f32[768]" = torch.ops.aten.add.Tensor(primals_173, 1e-05);  primals_173 = None
    rsqrt_12: "f32[768]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    unsqueeze_456: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_172, 0);  primals_172 = None
    unsqueeze_457: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
    unsqueeze_458: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_51: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_458);  convolution_26 = unsqueeze_458 = None
    mul_213: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_51);  sub_51 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 2, 3]);  mul_213 = None
    mul_218: "f32[768]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_53);  primals_53 = None
    unsqueeze_465: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_218, 0);  mul_218 = None
    unsqueeze_466: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    mul_219: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_467);  where_12 = unsqueeze_467 = None
    mul_220: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, rsqrt_12);  sum_27 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_219, cat_3, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_219 = cat_3 = primals_105 = None
    getitem_42: "f32[4, 1728, 14, 14]" = convolution_backward_12[0]
    getitem_43: "f32[768, 1728, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_13: "f32[4, 768, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_42, 1, 0, 768)
    slice_14: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_42, 1, 768, 960)
    slice_15: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_42, 1, 960, 1152)
    slice_16: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_42, 1, 1152, 1344)
    slice_17: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_42, 1, 1344, 1536)
    slice_18: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_42, 1, 1536, 1728);  getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_79: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_80: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    le_13: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_80, 0);  alias_80 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, slice_18);  le_13 = scalar_tensor_13 = slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_101: "f32[192]" = torch.ops.aten.add.Tensor(primals_171, 1e-05);  primals_171 = None
    rsqrt_13: "f32[192]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    unsqueeze_468: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_170, 0);  primals_170 = None
    unsqueeze_469: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
    unsqueeze_470: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
    sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_52: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_470);  convolution_25 = unsqueeze_470 = None
    mul_221: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_52);  sub_52 = None
    sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 2, 3]);  mul_221 = None
    mul_226: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_51);  primals_51 = None
    unsqueeze_477: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
    unsqueeze_478: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    mul_227: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_479);  where_13 = unsqueeze_479 = None
    mul_228: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_13);  sum_29 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_227, relu_24, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_227 = primals_104 = None
    getitem_45: "f32[4, 192, 14, 14]" = convolution_backward_13[0]
    getitem_46: "f32[192, 192, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_102: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_17, getitem_45);  slice_17 = getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_82: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_83: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    le_14: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, add_102);  le_14 = scalar_tensor_14 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_103: "f32[192]" = torch.ops.aten.add.Tensor(primals_169, 1e-05);  primals_169 = None
    rsqrt_14: "f32[192]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    unsqueeze_480: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_168, 0);  primals_168 = None
    unsqueeze_481: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
    unsqueeze_482: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
    sum_30: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_53: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_482);  convolution_24 = unsqueeze_482 = None
    mul_229: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_53);  sub_53 = None
    sum_31: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 2, 3]);  mul_229 = None
    mul_234: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_49);  primals_49 = None
    unsqueeze_489: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
    unsqueeze_490: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    mul_235: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_491);  where_14 = unsqueeze_491 = None
    mul_236: "f32[192]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_14);  sum_31 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_235, relu_23, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_235 = primals_103 = None
    getitem_48: "f32[4, 192, 14, 14]" = convolution_backward_14[0]
    getitem_49: "f32[192, 192, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_104: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_16, getitem_48);  slice_16 = getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_85: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_86: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    le_15: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_86, 0);  alias_86 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_104);  le_15 = scalar_tensor_15 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_105: "f32[192]" = torch.ops.aten.add.Tensor(primals_167, 1e-05);  primals_167 = None
    rsqrt_15: "f32[192]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    unsqueeze_492: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_166, 0);  primals_166 = None
    unsqueeze_493: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
    unsqueeze_494: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
    sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_54: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_494);  convolution_23 = unsqueeze_494 = None
    mul_237: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_54);  sub_54 = None
    sum_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 2, 3]);  mul_237 = None
    mul_242: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_47);  primals_47 = None
    unsqueeze_501: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_242, 0);  mul_242 = None
    unsqueeze_502: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    mul_243: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_503);  where_15 = unsqueeze_503 = None
    mul_244: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_15);  sum_33 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_243, relu_22, primals_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_243 = primals_102 = None
    getitem_51: "f32[4, 192, 14, 14]" = convolution_backward_15[0]
    getitem_52: "f32[192, 192, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_106: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_15, getitem_51);  slice_15 = getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_88: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_89: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    le_16: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, add_106);  le_16 = scalar_tensor_16 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_107: "f32[192]" = torch.ops.aten.add.Tensor(primals_165, 1e-05);  primals_165 = None
    rsqrt_16: "f32[192]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    unsqueeze_504: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_164, 0);  primals_164 = None
    unsqueeze_505: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
    unsqueeze_506: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
    sum_34: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_55: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_506);  convolution_22 = unsqueeze_506 = None
    mul_245: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_55);  sub_55 = None
    sum_35: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_245, [0, 2, 3]);  mul_245 = None
    mul_250: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_45);  primals_45 = None
    unsqueeze_513: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_250, 0);  mul_250 = None
    unsqueeze_514: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    mul_251: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_515);  where_16 = unsqueeze_515 = None
    mul_252: "f32[192]" = torch.ops.aten.mul.Tensor(sum_35, rsqrt_16);  sum_35 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_251, relu_21, primals_101, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_251 = primals_101 = None
    getitem_54: "f32[4, 192, 14, 14]" = convolution_backward_16[0]
    getitem_55: "f32[192, 192, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_108: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, getitem_54);  slice_14 = getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_91: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_92: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_17: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_92, 0);  alias_92 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, add_108);  le_17 = scalar_tensor_17 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_109: "f32[192]" = torch.ops.aten.add.Tensor(primals_163, 1e-05);  primals_163 = None
    rsqrt_17: "f32[192]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    unsqueeze_516: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_162, 0);  primals_162 = None
    unsqueeze_517: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
    unsqueeze_518: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
    sum_36: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_56: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_518);  convolution_21 = unsqueeze_518 = None
    mul_253: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_56);  sub_56 = None
    sum_37: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 2, 3]);  mul_253 = None
    mul_258: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_43);  primals_43 = None
    unsqueeze_525: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
    unsqueeze_526: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    mul_259: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, unsqueeze_527);  where_17 = unsqueeze_527 = None
    mul_260: "f32[192]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_17);  sum_37 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_259, relu_20, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_259 = primals_100 = None
    getitem_57: "f32[4, 768, 14, 14]" = convolution_backward_17[0]
    getitem_58: "f32[192, 768, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_110: "f32[4, 768, 14, 14]" = torch.ops.aten.add.Tensor(slice_13, getitem_57);  slice_13 = getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_94: "f32[4, 768, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_95: "f32[4, 768, 14, 14]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_18: "b8[4, 768, 14, 14]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[4, 768, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_110);  le_18 = scalar_tensor_18 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_111: "f32[768]" = torch.ops.aten.add.Tensor(primals_161, 1e-05);  primals_161 = None
    rsqrt_18: "f32[768]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    unsqueeze_528: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_160, 0);  primals_160 = None
    unsqueeze_529: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
    unsqueeze_530: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
    sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_57: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_530);  convolution_20 = unsqueeze_530 = None
    mul_261: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_57);  sub_57 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 2, 3]);  mul_261 = None
    mul_266: "f32[768]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_41);  primals_41 = None
    unsqueeze_537: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_266, 0);  mul_266 = None
    unsqueeze_538: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    mul_267: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_539);  where_18 = unsqueeze_539 = None
    mul_268: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_18);  sum_39 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_267, cat_2, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_267 = cat_2 = primals_99 = None
    getitem_60: "f32[4, 1472, 14, 14]" = convolution_backward_18[0]
    getitem_61: "f32[768, 1472, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_19: "f32[4, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_60, 1, 0, 512)
    slice_20: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_60, 1, 512, 704)
    slice_21: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_60, 1, 704, 896)
    slice_22: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_60, 1, 896, 1088)
    slice_23: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_60, 1, 1088, 1280)
    slice_24: "f32[4, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_60, 1, 1280, 1472);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_97: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_98: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_19: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_98, 0);  alias_98 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, slice_24);  le_19 = scalar_tensor_19 = slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_112: "f32[192]" = torch.ops.aten.add.Tensor(primals_159, 1e-05);  primals_159 = None
    rsqrt_19: "f32[192]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    unsqueeze_540: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_158, 0);  primals_158 = None
    unsqueeze_541: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
    unsqueeze_542: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
    sum_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_58: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_542);  convolution_19 = unsqueeze_542 = None
    mul_269: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_58);  sub_58 = None
    sum_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 2, 3]);  mul_269 = None
    mul_274: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_39);  primals_39 = None
    unsqueeze_549: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_550: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    mul_275: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_551);  where_19 = unsqueeze_551 = None
    mul_276: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_19);  sum_41 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_275, relu_18, primals_98, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_275 = primals_98 = None
    getitem_63: "f32[4, 192, 14, 14]" = convolution_backward_19[0]
    getitem_64: "f32[192, 192, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_113: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_23, getitem_63);  slice_23 = getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_100: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_101: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    le_20: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_101, 0);  alias_101 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, add_113);  le_20 = scalar_tensor_20 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_114: "f32[192]" = torch.ops.aten.add.Tensor(primals_157, 1e-05);  primals_157 = None
    rsqrt_20: "f32[192]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    unsqueeze_552: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_156, 0);  primals_156 = None
    unsqueeze_553: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
    unsqueeze_554: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
    sum_42: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_59: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_554);  convolution_18 = unsqueeze_554 = None
    mul_277: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_59);  sub_59 = None
    sum_43: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2, 3]);  mul_277 = None
    mul_282: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_37);  primals_37 = None
    unsqueeze_561: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
    unsqueeze_562: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    mul_283: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_563);  where_20 = unsqueeze_563 = None
    mul_284: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, rsqrt_20);  sum_43 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_283, relu_17, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_283 = primals_97 = None
    getitem_66: "f32[4, 192, 14, 14]" = convolution_backward_20[0]
    getitem_67: "f32[192, 192, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_115: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, getitem_66);  slice_22 = getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_103: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_104: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    le_21: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_104, 0);  alias_104 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_115);  le_21 = scalar_tensor_21 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_116: "f32[192]" = torch.ops.aten.add.Tensor(primals_155, 1e-05);  primals_155 = None
    rsqrt_21: "f32[192]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    unsqueeze_564: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_154, 0);  primals_154 = None
    unsqueeze_565: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
    unsqueeze_566: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
    sum_44: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_60: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_566);  convolution_17 = unsqueeze_566 = None
    mul_285: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_60);  sub_60 = None
    sum_45: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 2, 3]);  mul_285 = None
    mul_290: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_35);  primals_35 = None
    unsqueeze_573: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_574: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    mul_291: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_575);  where_21 = unsqueeze_575 = None
    mul_292: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, rsqrt_21);  sum_45 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_291, relu_16, primals_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_291 = primals_96 = None
    getitem_69: "f32[4, 192, 14, 14]" = convolution_backward_21[0]
    getitem_70: "f32[192, 192, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_117: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_21, getitem_69);  slice_21 = getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_106: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_107: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    le_22: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_107, 0);  alias_107 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, add_117);  le_22 = scalar_tensor_22 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_118: "f32[192]" = torch.ops.aten.add.Tensor(primals_153, 1e-05);  primals_153 = None
    rsqrt_22: "f32[192]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    unsqueeze_576: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_152, 0);  primals_152 = None
    unsqueeze_577: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
    unsqueeze_578: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
    sum_46: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_61: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_578);  convolution_16 = unsqueeze_578 = None
    mul_293: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_61);  sub_61 = None
    sum_47: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
    mul_298: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_33);  primals_33 = None
    unsqueeze_585: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_586: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    mul_299: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_587);  where_22 = unsqueeze_587 = None
    mul_300: "f32[192]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_22);  sum_47 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_299, relu_15, primals_95, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_299 = primals_95 = None
    getitem_72: "f32[4, 192, 14, 14]" = convolution_backward_22[0]
    getitem_73: "f32[192, 192, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_119: "f32[4, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_20, getitem_72);  slice_20 = getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_109: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_110: "f32[4, 192, 14, 14]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    le_23: "b8[4, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[4, 192, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, add_119);  le_23 = scalar_tensor_23 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_120: "f32[192]" = torch.ops.aten.add.Tensor(primals_151, 1e-05);  primals_151 = None
    rsqrt_23: "f32[192]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    unsqueeze_588: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(primals_150, 0);  primals_150 = None
    unsqueeze_589: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
    unsqueeze_590: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
    sum_48: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_62: "f32[4, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_590);  convolution_15 = unsqueeze_590 = None
    mul_301: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_62);  sub_62 = None
    sum_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 2, 3]);  mul_301 = None
    mul_306: "f32[192]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_31);  primals_31 = None
    unsqueeze_597: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
    unsqueeze_598: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    mul_307: "f32[4, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_599);  where_23 = unsqueeze_599 = None
    mul_308: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_23);  sum_49 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_307, getitem_2, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_307 = getitem_2 = primals_94 = None
    getitem_75: "f32[4, 512, 14, 14]" = convolution_backward_23[0]
    getitem_76: "f32[192, 512, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_121: "f32[4, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_19, getitem_75);  slice_19 = getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward_1: "f32[4, 512, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_121, relu_14, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3);  add_121 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_112: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_113: "f32[4, 512, 28, 28]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    le_24: "b8[4, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_113, 0);  alias_113 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[4, 512, 28, 28]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, max_pool2d_with_indices_backward_1);  le_24 = scalar_tensor_24 = max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_122: "f32[512]" = torch.ops.aten.add.Tensor(primals_149, 1e-05);  primals_149 = None
    rsqrt_24: "f32[512]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    unsqueeze_600: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_148, 0);  primals_148 = None
    unsqueeze_601: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
    unsqueeze_602: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_63: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_602);  convolution_14 = unsqueeze_602 = None
    mul_309: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_63);  sub_63 = None
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3]);  mul_309 = None
    mul_314: "f32[512]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_29);  primals_29 = None
    unsqueeze_609: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_610: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    mul_315: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_611);  where_24 = unsqueeze_611 = None
    mul_316: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_24);  sum_51 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_315, cat_1, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_315 = cat_1 = primals_93 = None
    getitem_78: "f32[4, 1056, 28, 28]" = convolution_backward_24[0]
    getitem_79: "f32[512, 1056, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_25: "f32[4, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_78, 1, 0, 256)
    slice_26: "f32[4, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_78, 1, 256, 416)
    slice_27: "f32[4, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_78, 1, 416, 576)
    slice_28: "f32[4, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_78, 1, 576, 736)
    slice_29: "f32[4, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_78, 1, 736, 896)
    slice_30: "f32[4, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_78, 1, 896, 1056);  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_115: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_116: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(alias_115);  alias_115 = None
    le_25: "b8[4, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_116, 0);  alias_116 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[4, 160, 28, 28]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, slice_30);  le_25 = scalar_tensor_25 = slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_123: "f32[160]" = torch.ops.aten.add.Tensor(primals_147, 1e-05);  primals_147 = None
    rsqrt_25: "f32[160]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    unsqueeze_612: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_146, 0);  primals_146 = None
    unsqueeze_613: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
    unsqueeze_614: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
    sum_52: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_64: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_614);  convolution_13 = unsqueeze_614 = None
    mul_317: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_64);  sub_64 = None
    sum_53: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 2, 3]);  mul_317 = None
    mul_322: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_27);  primals_27 = None
    unsqueeze_621: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_622: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    mul_323: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_623);  where_25 = unsqueeze_623 = None
    mul_324: "f32[160]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_25);  sum_53 = rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_323, relu_12, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_323 = primals_92 = None
    getitem_81: "f32[4, 160, 28, 28]" = convolution_backward_25[0]
    getitem_82: "f32[160, 160, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_124: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_29, getitem_81);  slice_29 = getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_118: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_119: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(alias_118);  alias_118 = None
    le_26: "b8[4, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_119, 0);  alias_119 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[4, 160, 28, 28]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, add_124);  le_26 = scalar_tensor_26 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_125: "f32[160]" = torch.ops.aten.add.Tensor(primals_145, 1e-05);  primals_145 = None
    rsqrt_26: "f32[160]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    unsqueeze_624: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_144, 0);  primals_144 = None
    unsqueeze_625: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
    unsqueeze_626: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
    sum_54: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_65: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_626);  convolution_12 = unsqueeze_626 = None
    mul_325: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_65);  sub_65 = None
    sum_55: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3]);  mul_325 = None
    mul_330: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_25);  primals_25 = None
    unsqueeze_633: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_634: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    mul_331: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_635);  where_26 = unsqueeze_635 = None
    mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_26);  sum_55 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_331, relu_11, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_331 = primals_91 = None
    getitem_84: "f32[4, 160, 28, 28]" = convolution_backward_26[0]
    getitem_85: "f32[160, 160, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_126: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_28, getitem_84);  slice_28 = getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_121: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_122: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    le_27: "b8[4, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_122, 0);  alias_122 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[4, 160, 28, 28]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, add_126);  le_27 = scalar_tensor_27 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_127: "f32[160]" = torch.ops.aten.add.Tensor(primals_143, 1e-05);  primals_143 = None
    rsqrt_27: "f32[160]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    unsqueeze_636: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_142, 0);  primals_142 = None
    unsqueeze_637: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
    unsqueeze_638: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
    sum_56: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_66: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_638);  convolution_11 = unsqueeze_638 = None
    mul_333: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_66);  sub_66 = None
    sum_57: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 2, 3]);  mul_333 = None
    mul_338: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_23);  primals_23 = None
    unsqueeze_645: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_646: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    mul_339: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_647);  where_27 = unsqueeze_647 = None
    mul_340: "f32[160]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_27);  sum_57 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_339, relu_10, primals_90, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_339 = primals_90 = None
    getitem_87: "f32[4, 160, 28, 28]" = convolution_backward_27[0]
    getitem_88: "f32[160, 160, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_128: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_27, getitem_87);  slice_27 = getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_124: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_125: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    le_28: "b8[4, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_125, 0);  alias_125 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[4, 160, 28, 28]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, add_128);  le_28 = scalar_tensor_28 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_129: "f32[160]" = torch.ops.aten.add.Tensor(primals_141, 1e-05);  primals_141 = None
    rsqrt_28: "f32[160]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    unsqueeze_648: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_140, 0);  primals_140 = None
    unsqueeze_649: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
    unsqueeze_650: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
    sum_58: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_67: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_650);  convolution_10 = unsqueeze_650 = None
    mul_341: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_67);  sub_67 = None
    sum_59: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3]);  mul_341 = None
    mul_346: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_21);  primals_21 = None
    unsqueeze_657: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_658: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    mul_347: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_659);  where_28 = unsqueeze_659 = None
    mul_348: "f32[160]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_28);  sum_59 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_347, relu_9, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_347 = primals_89 = None
    getitem_90: "f32[4, 160, 28, 28]" = convolution_backward_28[0]
    getitem_91: "f32[160, 160, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_130: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_26, getitem_90);  slice_26 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_127: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_128: "f32[4, 160, 28, 28]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    le_29: "b8[4, 160, 28, 28]" = torch.ops.aten.le.Scalar(alias_128, 0);  alias_128 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[4, 160, 28, 28]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, add_130);  le_29 = scalar_tensor_29 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_131: "f32[160]" = torch.ops.aten.add.Tensor(primals_139, 1e-05);  primals_139 = None
    rsqrt_29: "f32[160]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    unsqueeze_660: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_138, 0);  primals_138 = None
    unsqueeze_661: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
    unsqueeze_662: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
    sum_60: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_68: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_662);  convolution_9 = unsqueeze_662 = None
    mul_349: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_68);  sub_68 = None
    sum_61: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 2, 3]);  mul_349 = None
    mul_354: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_19);  primals_19 = None
    unsqueeze_669: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_354, 0);  mul_354 = None
    unsqueeze_670: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    mul_355: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, unsqueeze_671);  where_29 = unsqueeze_671 = None
    mul_356: "f32[160]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_29);  sum_61 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_355, getitem, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_355 = getitem = primals_88 = None
    getitem_93: "f32[4, 256, 28, 28]" = convolution_backward_29[0]
    getitem_94: "f32[160, 256, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_132: "f32[4, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_25, getitem_93);  slice_25 = getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:145, code: x = self.pool(x)
    max_pool2d_with_indices_backward_2: "f32[4, 256, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_132, relu_8, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1);  add_132 = getitem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_130: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_131: "f32[4, 256, 56, 56]" = torch.ops.aten.alias.default(alias_130);  alias_130 = None
    le_30: "b8[4, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_131, 0);  alias_131 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[4, 256, 56, 56]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, max_pool2d_with_indices_backward_2);  le_30 = scalar_tensor_30 = max_pool2d_with_indices_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_133: "f32[256]" = torch.ops.aten.add.Tensor(primals_137, 1e-05);  primals_137 = None
    rsqrt_30: "f32[256]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_672: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_136, 0);  primals_136 = None
    unsqueeze_673: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
    unsqueeze_674: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
    sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_69: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_674);  convolution_8 = unsqueeze_674 = None
    mul_357: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_69);  sub_69 = None
    sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_17);  primals_17 = None
    unsqueeze_681: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_682: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    mul_363: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_683);  where_30 = unsqueeze_683 = None
    mul_364: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_30);  sum_63 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_363, cat, primals_87, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_363 = cat = primals_87 = None
    getitem_96: "f32[4, 768, 56, 56]" = convolution_backward_30[0]
    getitem_97: "f32[256, 768, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vovnet.py:39, code: x = torch.cat(concat_list, dim=1)
    slice_31: "f32[4, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_96, 1, 0, 128)
    slice_32: "f32[4, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_96, 1, 128, 256)
    slice_33: "f32[4, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_96, 1, 256, 384)
    slice_34: "f32[4, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_96, 1, 384, 512)
    slice_35: "f32[4, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_96, 1, 512, 640)
    slice_36: "f32[4, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_96, 1, 640, 768);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_133: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_134: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_133);  alias_133 = None
    le_31: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_134, 0);  alias_134 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, slice_36);  le_31 = scalar_tensor_31 = slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_134: "f32[128]" = torch.ops.aten.add.Tensor(primals_135, 1e-05);  primals_135 = None
    rsqrt_31: "f32[128]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    unsqueeze_684: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_134, 0);  primals_134 = None
    unsqueeze_685: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
    unsqueeze_686: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
    sum_64: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_70: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_686);  convolution_7 = unsqueeze_686 = None
    mul_365: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_70);  sub_70 = None
    sum_65: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 2, 3]);  mul_365 = None
    mul_370: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_15);  primals_15 = None
    unsqueeze_693: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_694: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    mul_371: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_695);  where_31 = unsqueeze_695 = None
    mul_372: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_31);  sum_65 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_371, relu_6, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = primals_86 = None
    getitem_99: "f32[4, 128, 56, 56]" = convolution_backward_31[0]
    getitem_100: "f32[128, 128, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_135: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_35, getitem_99);  slice_35 = getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_136: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_137: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_136);  alias_136 = None
    le_32: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_137, 0);  alias_137 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, add_135);  le_32 = scalar_tensor_32 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_136: "f32[128]" = torch.ops.aten.add.Tensor(primals_133, 1e-05);  primals_133 = None
    rsqrt_32: "f32[128]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    unsqueeze_696: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_132, 0);  primals_132 = None
    unsqueeze_697: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
    unsqueeze_698: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
    sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_71: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_698);  convolution_6 = unsqueeze_698 = None
    mul_373: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_32, sub_71);  sub_71 = None
    sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_378: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_13);  primals_13 = None
    unsqueeze_705: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_706: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    mul_379: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_707);  where_32 = unsqueeze_707 = None
    mul_380: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_32);  sum_67 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_379, relu_5, primals_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_379 = primals_85 = None
    getitem_102: "f32[4, 128, 56, 56]" = convolution_backward_32[0]
    getitem_103: "f32[128, 128, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_137: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_34, getitem_102);  slice_34 = getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_139: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_140: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_139);  alias_139 = None
    le_33: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_140, 0);  alias_140 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, add_137);  le_33 = scalar_tensor_33 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_138: "f32[128]" = torch.ops.aten.add.Tensor(primals_131, 1e-05);  primals_131 = None
    rsqrt_33: "f32[128]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    unsqueeze_708: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_130, 0);  primals_130 = None
    unsqueeze_709: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
    unsqueeze_710: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_72: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_710);  convolution_5 = unsqueeze_710 = None
    mul_381: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_33, sub_72);  sub_72 = None
    sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
    mul_386: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_11);  primals_11 = None
    unsqueeze_717: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_718: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    mul_387: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_719);  where_33 = unsqueeze_719 = None
    mul_388: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_33);  sum_69 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_387, relu_4, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_387 = primals_84 = None
    getitem_105: "f32[4, 128, 56, 56]" = convolution_backward_33[0]
    getitem_106: "f32[128, 128, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_139: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_33, getitem_105);  slice_33 = getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_142: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_143: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_142);  alias_142 = None
    le_34: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_143, 0);  alias_143 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, add_139);  le_34 = scalar_tensor_34 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_140: "f32[128]" = torch.ops.aten.add.Tensor(primals_129, 1e-05);  primals_129 = None
    rsqrt_34: "f32[128]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    unsqueeze_720: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_128, 0);  primals_128 = None
    unsqueeze_721: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
    unsqueeze_722: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_73: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_722);  convolution_4 = unsqueeze_722 = None
    mul_389: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_73);  sub_73 = None
    sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_389, [0, 2, 3]);  mul_389 = None
    mul_394: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_9);  primals_9 = None
    unsqueeze_729: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
    unsqueeze_730: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    mul_395: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_731);  where_34 = unsqueeze_731 = None
    mul_396: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_34);  sum_71 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_395, relu_3, primals_83, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_395 = primals_83 = None
    getitem_108: "f32[4, 128, 56, 56]" = convolution_backward_34[0]
    getitem_109: "f32[128, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_141: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_32, getitem_108);  slice_32 = getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_145: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_146: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_145);  alias_145 = None
    le_35: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_146, 0);  alias_146 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, add_141);  le_35 = scalar_tensor_35 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_142: "f32[128]" = torch.ops.aten.add.Tensor(primals_127, 1e-05);  primals_127 = None
    rsqrt_35: "f32[128]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    unsqueeze_732: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_126, 0);  primals_126 = None
    unsqueeze_733: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
    unsqueeze_734: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
    sum_72: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_74: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_734);  convolution_3 = unsqueeze_734 = None
    mul_397: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_74);  sub_74 = None
    sum_73: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 2, 3]);  mul_397 = None
    mul_402: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_7);  primals_7 = None
    unsqueeze_741: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
    unsqueeze_742: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    mul_403: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_743);  where_35 = unsqueeze_743 = None
    mul_404: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_35);  sum_73 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_403, relu_2, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_403 = primals_82 = None
    getitem_111: "f32[4, 128, 56, 56]" = convolution_backward_35[0]
    getitem_112: "f32[128, 128, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_143: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_31, getitem_111);  slice_31 = getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_148: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_149: "f32[4, 128, 56, 56]" = torch.ops.aten.alias.default(alias_148);  alias_148 = None
    le_36: "b8[4, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_149, 0);  alias_149 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[4, 128, 56, 56]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, add_143);  le_36 = scalar_tensor_36 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_144: "f32[128]" = torch.ops.aten.add.Tensor(primals_125, 1e-05);  primals_125 = None
    rsqrt_36: "f32[128]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_744: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_124, 0);  primals_124 = None
    unsqueeze_745: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
    unsqueeze_746: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_75: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_746);  convolution_2 = unsqueeze_746 = None
    mul_405: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_75);  sub_75 = None
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3]);  mul_405 = None
    mul_410: "f32[128]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_5);  primals_5 = None
    unsqueeze_753: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_754: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    mul_411: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_755);  where_36 = unsqueeze_755 = None
    mul_412: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_36);  sum_75 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_411, relu_1, primals_81, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_411 = primals_81 = None
    getitem_114: "f32[4, 64, 112, 112]" = convolution_backward_36[0]
    getitem_115: "f32[128, 64, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_151: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_152: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(alias_151);  alias_151 = None
    le_37: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_152, 0);  alias_152 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_114);  le_37 = scalar_tensor_37 = getitem_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_145: "f32[64]" = torch.ops.aten.add.Tensor(primals_123, 1e-05);  primals_123 = None
    rsqrt_37: "f32[64]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    unsqueeze_756: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_122, 0);  primals_122 = None
    unsqueeze_757: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
    unsqueeze_758: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
    sum_76: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_76: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_758);  convolution_1 = unsqueeze_758 = None
    mul_413: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_37, sub_76);  sub_76 = None
    sum_77: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 2, 3]);  mul_413 = None
    mul_418: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_3);  primals_3 = None
    unsqueeze_765: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_766: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    mul_419: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_37, unsqueeze_767);  where_37 = unsqueeze_767 = None
    mul_420: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_37);  sum_77 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_419, relu, primals_80, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_419 = primals_80 = None
    getitem_117: "f32[4, 64, 112, 112]" = convolution_backward_37[0]
    getitem_118: "f32[64, 64, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_154: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_155: "f32[4, 64, 112, 112]" = torch.ops.aten.alias.default(alias_154);  alias_154 = None
    le_38: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_155, 0);  alias_155 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, getitem_117);  le_38 = scalar_tensor_38 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_146: "f32[64]" = torch.ops.aten.add.Tensor(primals_121, 1e-05);  primals_121 = None
    rsqrt_38: "f32[64]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_768: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_120, 0);  primals_120 = None
    unsqueeze_769: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
    unsqueeze_770: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
    sum_78: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_77: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_770);  convolution = unsqueeze_770 = None
    mul_421: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_38, sub_77);  sub_77 = None
    sum_79: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 2, 3]);  mul_421 = None
    mul_426: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_1);  primals_1 = None
    unsqueeze_777: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_778: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    mul_427: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_38, unsqueeze_779);  where_38 = unsqueeze_779 = None
    mul_428: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, rsqrt_38);  sum_79 = rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_427, primals_198, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_427 = primals_198 = primals_79 = None
    getitem_121: "f32[64, 3, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    return pytree.tree_unflatten([addmm, mul_428, sum_78, mul_420, sum_76, mul_412, sum_74, mul_404, sum_72, mul_396, sum_70, mul_388, sum_68, mul_380, sum_66, mul_372, sum_64, mul_364, sum_62, mul_356, sum_60, mul_348, sum_58, mul_340, sum_56, mul_332, sum_54, mul_324, sum_52, mul_316, sum_50, mul_308, sum_48, mul_300, sum_46, mul_292, sum_44, mul_284, sum_42, mul_276, sum_40, mul_268, sum_38, mul_260, sum_36, mul_252, sum_34, mul_244, sum_32, mul_236, sum_30, mul_228, sum_28, mul_220, sum_26, mul_212, sum_24, mul_204, sum_22, mul_196, sum_20, mul_188, sum_18, mul_180, sum_16, mul_172, sum_14, mul_164, sum_12, mul_156, sum_10, mul_148, sum_8, mul_140, sum_6, mul_132, sum_4, mul_124, sum_2, getitem_121, getitem_118, getitem_115, getitem_112, getitem_109, getitem_106, getitem_103, getitem_100, getitem_97, getitem_94, getitem_91, getitem_88, getitem_85, getitem_82, getitem_79, getitem_76, getitem_73, getitem_70, getitem_67, getitem_64, getitem_61, getitem_58, getitem_55, getitem_52, getitem_49, getitem_46, getitem_43, getitem_40, getitem_37, getitem_34, getitem_31, getitem_28, getitem_25, getitem_22, getitem_19, getitem_16, getitem_13, getitem_10, getitem_7, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    