from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 192, 28, 28]"; primals_2: "f32[1, 384, 14, 14]"; primals_3: "f32[1, 768, 7, 7]"; primals_4: "f32[32, 3, 7, 7]"; primals_5: "f32[32]"; primals_6: "f32[32]"; primals_7: "f32[192, 32, 4, 4]"; primals_8: "f32[192]"; primals_9: "f32[192]"; primals_10: "f32[192]"; primals_11: "f32[192]"; primals_12: "f32[192]"; primals_13: "f32[384, 192, 1, 1]"; primals_14: "f32[384, 48, 3, 3]"; primals_15: "f32[192, 384, 1, 1]"; primals_16: "f32[192]"; primals_17: "f32[192]"; primals_18: "f32[384, 192, 1, 1]"; primals_19: "f32[384, 48, 3, 3]"; primals_20: "f32[192, 384, 1, 1]"; primals_21: "f32[192]"; primals_22: "f32[192]"; primals_23: "f32[384, 192, 1, 1]"; primals_24: "f32[384, 48, 3, 3]"; primals_25: "f32[192, 384, 1, 1]"; primals_26: "f32[192]"; primals_27: "f32[192]"; primals_28: "f32[384, 192, 1, 1]"; primals_29: "f32[384, 48, 3, 3]"; primals_30: "f32[192, 384, 1, 1]"; primals_31: "f32[192]"; primals_32: "f32[192]"; primals_33: "f32[384, 192, 1, 1]"; primals_34: "f32[384, 48, 3, 3]"; primals_35: "f32[192, 384, 1, 1]"; primals_36: "f32[192]"; primals_37: "f32[192]"; primals_38: "f32[384, 192, 1, 1]"; primals_39: "f32[384, 48, 3, 3]"; primals_40: "f32[192, 384, 1, 1]"; primals_41: "f32[192]"; primals_42: "f32[192]"; primals_43: "f32[384, 192, 1, 1]"; primals_44: "f32[384, 48, 3, 3]"; primals_45: "f32[192, 384, 1, 1]"; primals_46: "f32[384, 192, 2, 2]"; primals_47: "f32[384]"; primals_48: "f32[384]"; primals_49: "f32[384]"; primals_50: "f32[384]"; primals_51: "f32[384]"; primals_52: "f32[1152, 384, 1, 1]"; primals_53: "f32[384, 384, 1, 1]"; primals_54: "f32[384]"; primals_55: "f32[384]"; primals_56: "f32[1536, 384, 1, 1]"; primals_57: "f32[384, 1536, 1, 1]"; primals_58: "f32[384]"; primals_59: "f32[384]"; primals_60: "f32[1152, 384, 1, 1]"; primals_61: "f32[384, 384, 1, 1]"; primals_62: "f32[384]"; primals_63: "f32[384]"; primals_64: "f32[1536, 384, 1, 1]"; primals_65: "f32[384, 1536, 1, 1]"; primals_66: "f32[384]"; primals_67: "f32[384]"; primals_68: "f32[1152, 384, 1, 1]"; primals_69: "f32[384, 384, 1, 1]"; primals_70: "f32[384]"; primals_71: "f32[384]"; primals_72: "f32[1536, 384, 1, 1]"; primals_73: "f32[384, 1536, 1, 1]"; primals_74: "f32[384]"; primals_75: "f32[384]"; primals_76: "f32[1152, 384, 1, 1]"; primals_77: "f32[384, 384, 1, 1]"; primals_78: "f32[384]"; primals_79: "f32[384]"; primals_80: "f32[1536, 384, 1, 1]"; primals_81: "f32[384, 1536, 1, 1]"; primals_82: "f32[768, 384, 2, 2]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[768]"; primals_87: "f32[768]"; primals_88: "f32[2304, 768, 1, 1]"; primals_89: "f32[768, 768, 1, 1]"; primals_90: "f32[768]"; primals_91: "f32[768]"; primals_92: "f32[3072, 768, 1, 1]"; primals_93: "f32[768, 3072, 1, 1]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[2304, 768, 1, 1]"; primals_97: "f32[768, 768, 1, 1]"; primals_98: "f32[768]"; primals_99: "f32[768]"; primals_100: "f32[3072, 768, 1, 1]"; primals_101: "f32[768, 3072, 1, 1]"; primals_102: "f32[768]"; primals_103: "f32[768]"; primals_104: "f32[2304, 768, 1, 1]"; primals_105: "f32[768, 768, 1, 1]"; primals_106: "f32[768]"; primals_107: "f32[768]"; primals_108: "f32[3072, 768, 1, 1]"; primals_109: "f32[768, 3072, 1, 1]"; primals_110: "f32[768]"; primals_111: "f32[768]"; primals_112: "f32[2304, 768, 1, 1]"; primals_113: "f32[768, 768, 1, 1]"; primals_114: "f32[768]"; primals_115: "f32[768]"; primals_116: "f32[3072, 768, 1, 1]"; primals_117: "f32[768, 3072, 1, 1]"; primals_118: "f32[768]"; primals_119: "f32[768]"; primals_120: "f32[1000, 768]"; primals_121: "f32[1000]"; primals_122: "f32[32]"; primals_123: "f32[32]"; primals_124: "i64[]"; primals_125: "f32[192]"; primals_126: "f32[192]"; primals_127: "i64[]"; primals_128: "f32[192]"; primals_129: "f32[192]"; primals_130: "i64[]"; primals_131: "f32[192]"; primals_132: "f32[192]"; primals_133: "i64[]"; primals_134: "f32[192]"; primals_135: "f32[192]"; primals_136: "i64[]"; primals_137: "f32[192]"; primals_138: "f32[192]"; primals_139: "i64[]"; primals_140: "f32[192]"; primals_141: "f32[192]"; primals_142: "i64[]"; primals_143: "f32[192]"; primals_144: "f32[192]"; primals_145: "i64[]"; primals_146: "f32[192]"; primals_147: "f32[192]"; primals_148: "i64[]"; primals_149: "f32[384]"; primals_150: "f32[384]"; primals_151: "i64[]"; primals_152: "f32[384]"; primals_153: "f32[384]"; primals_154: "i64[]"; primals_155: "f32[384]"; primals_156: "f32[384]"; primals_157: "i64[]"; primals_158: "f32[384]"; primals_159: "f32[384]"; primals_160: "i64[]"; primals_161: "f32[384]"; primals_162: "f32[384]"; primals_163: "i64[]"; primals_164: "f32[384]"; primals_165: "f32[384]"; primals_166: "i64[]"; primals_167: "f32[384]"; primals_168: "f32[384]"; primals_169: "i64[]"; primals_170: "f32[384]"; primals_171: "f32[384]"; primals_172: "i64[]"; primals_173: "f32[384]"; primals_174: "f32[384]"; primals_175: "i64[]"; primals_176: "f32[768]"; primals_177: "f32[768]"; primals_178: "i64[]"; primals_179: "f32[768]"; primals_180: "f32[768]"; primals_181: "i64[]"; primals_182: "f32[768]"; primals_183: "f32[768]"; primals_184: "i64[]"; primals_185: "f32[768]"; primals_186: "f32[768]"; primals_187: "i64[]"; primals_188: "f32[768]"; primals_189: "f32[768]"; primals_190: "i64[]"; primals_191: "f32[768]"; primals_192: "f32[768]"; primals_193: "i64[]"; primals_194: "f32[768]"; primals_195: "f32[768]"; primals_196: "i64[]"; primals_197: "f32[768]"; primals_198: "f32[768]"; primals_199: "i64[]"; primals_200: "f32[768]"; primals_201: "f32[768]"; primals_202: "i64[]"; primals_203: "f32[768]"; primals_204: "f32[768]"; primals_205: "i64[]"; primals_206: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:396, code: x = self.stem(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_206, primals_4, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_124, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_122, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_1: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu, primals_7, primals_8, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_127, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 192, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 192, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[192]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
    add_7: "f32[192]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0001594642002871);  squeeze_5 = None
    mul_11: "f32[192]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[192]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
    add_8: "f32[192]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:401, code: x = self.pos_drop(x + self.pos_embed1)
    add_10: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_9, primals_1);  add_9 = primals_1 = None
    clone: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(add_10);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_11: "i64[]" = torch.ops.aten.add.Tensor(primals_130, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 192, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 192, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_12: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_2: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(clone, getitem_5)
    mul_14: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[192]" = torch.ops.aten.mul.Tensor(primals_128, 0.9)
    add_13: "f32[192]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0001594642002871);  squeeze_8 = None
    mul_18: "f32[192]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[192]" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
    add_14: "f32[192]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_15: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_2: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_15, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_21: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.5)
    mul_22: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
    erf: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
    add_16: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_23: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_21, add_16);  mul_21 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_1: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_3: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_1, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_24: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
    mul_25: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_1: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_17: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_26: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_24, add_17);  mul_24 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_4: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_26, primals_15, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_2: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_18: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(clone, clone_2);  clone_2 = None
    add_19: "i64[]" = torch.ops.aten.add.Tensor(primals_133, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 192, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 192, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_20: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_3: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_18, getitem_7)
    mul_27: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_28: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_29: "f32[192]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
    add_21: "f32[192]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    squeeze_11: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_30: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0001594642002871);  squeeze_11 = None
    mul_31: "f32[192]" = torch.ops.aten.mul.Tensor(mul_30, 0.1);  mul_30 = None
    mul_32: "f32[192]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
    add_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
    unsqueeze_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1)
    unsqueeze_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_33: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_13);  mul_27 = unsqueeze_13 = None
    unsqueeze_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1);  primals_17 = None
    unsqueeze_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_23: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_15);  mul_33 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_5: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_23, primals_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_34: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
    mul_35: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_2: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_24: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_36: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_34, add_24);  mul_34 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_3: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_6: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_3, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_37: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_38: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_3: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_25: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, add_25);  mul_37 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_7: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_39, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_4: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_26: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_18, clone_4);  clone_4 = None
    add_27: "i64[]" = torch.ops.aten.add.Tensor(primals_136, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 192, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 192, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_28: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_4: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_26, getitem_9)
    mul_40: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_41: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_42: "f32[192]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
    add_29: "f32[192]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    squeeze_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_43: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0001594642002871);  squeeze_14 = None
    mul_44: "f32[192]" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
    mul_45: "f32[192]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
    add_30: "f32[192]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_46: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_17);  mul_40 = unsqueeze_17 = None
    unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_31: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_19);  mul_46 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_8: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_31, primals_23, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_47: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.5)
    mul_48: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476)
    erf_4: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_32: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_49: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_47, add_32);  mul_47 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_5: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_9: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_5, primals_24, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_50: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
    mul_51: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_5: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_33: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, add_33);  mul_50 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_10: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_52, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_6: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_34: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_26, clone_6);  clone_6 = None
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_139, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 192, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 192, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_36: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_5: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_34, getitem_11)
    mul_53: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_54: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_55: "f32[192]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
    add_37: "f32[192]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    squeeze_17: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_56: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001594642002871);  squeeze_17 = None
    mul_57: "f32[192]" = torch.ops.aten.mul.Tensor(mul_56, 0.1);  mul_56 = None
    mul_58: "f32[192]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
    add_38: "f32[192]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    unsqueeze_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_59: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_53, unsqueeze_21);  mul_53 = unsqueeze_21 = None
    unsqueeze_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_39: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_23);  mul_59 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_11: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_39, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_60: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, 0.5)
    mul_61: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476)
    erf_6: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_40: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_62: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_60, add_40);  mul_60 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_7: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_12: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_7, primals_29, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_63: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_64: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_7: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_41: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_65: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, add_41);  mul_63 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_13: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_65, primals_30, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_8: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_42: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_34, clone_8);  clone_8 = None
    add_43: "i64[]" = torch.ops.aten.add.Tensor(primals_142, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 192, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 192, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_44: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_6: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_42, getitem_13)
    mul_66: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_67: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_68: "f32[192]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
    add_45: "f32[192]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    squeeze_20: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_69: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001594642002871);  squeeze_20 = None
    mul_70: "f32[192]" = torch.ops.aten.mul.Tensor(mul_69, 0.1);  mul_69 = None
    mul_71: "f32[192]" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
    add_46: "f32[192]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
    unsqueeze_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_72: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_25);  mul_66 = unsqueeze_25 = None
    unsqueeze_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_47: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_27);  mul_72 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_14: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_47, primals_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_73: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_74: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_8: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_48: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_75: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, add_48);  mul_73 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_9: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_15: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_9, primals_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_76: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.5)
    mul_77: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.7071067811865476)
    erf_9: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_49: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_78: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_76, add_49);  mul_76 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_16: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_78, primals_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_10: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_42, clone_10);  clone_10 = None
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_145, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 192, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 192, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_52: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_7: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_50, getitem_15)
    mul_79: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_80: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_81: "f32[192]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
    add_53: "f32[192]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
    squeeze_23: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_82: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001594642002871);  squeeze_23 = None
    mul_83: "f32[192]" = torch.ops.aten.mul.Tensor(mul_82, 0.1);  mul_82 = None
    mul_84: "f32[192]" = torch.ops.aten.mul.Tensor(primals_144, 0.9)
    add_54: "f32[192]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    unsqueeze_28: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
    unsqueeze_29: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_85: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_29);  mul_79 = unsqueeze_29 = None
    unsqueeze_30: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
    unsqueeze_31: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_55: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_31);  mul_85 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_17: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_55, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_86: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, 0.5)
    mul_87: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, 0.7071067811865476)
    erf_10: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_56: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_86, add_56);  mul_86 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_11: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_18: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_11, primals_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_89: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_90: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_11: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_57: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_91: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_89, add_57);  mul_89 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_19: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_91, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_12: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_58: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_50, clone_12);  clone_12 = None
    add_59: "i64[]" = torch.ops.aten.add.Tensor(primals_148, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 192, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 192, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_60: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_8: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_58, getitem_17)
    mul_92: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_93: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_94: "f32[192]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
    add_61: "f32[192]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    squeeze_26: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_95: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
    mul_96: "f32[192]" = torch.ops.aten.mul.Tensor(mul_95, 0.1);  mul_95 = None
    mul_97: "f32[192]" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
    add_62: "f32[192]" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
    unsqueeze_32: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_33: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_98: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_33);  mul_92 = unsqueeze_33 = None
    unsqueeze_34: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_35: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_63: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_35);  mul_98 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_20: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(add_63, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_99: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_100: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_12: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_64: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_101: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_99, add_64);  mul_99 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_13: "f32[8, 384, 28, 28]" = torch.ops.aten.clone.default(mul_101);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_21: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(clone_13, primals_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_102: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.5)
    mul_103: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.7071067811865476)
    erf_13: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_65: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_104: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_102, add_65);  mul_102 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_22: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_104, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_14: "f32[8, 192, 28, 28]" = torch.ops.aten.clone.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_66: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_58, clone_14);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_23: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(add_66, primals_46, primals_47, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_151, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 384, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 384, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_68: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_9: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_19)
    mul_105: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_106: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_107: "f32[384]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
    add_69: "f32[384]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_29: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_108: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0006381620931717);  squeeze_29 = None
    mul_109: "f32[384]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[384]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_70: "f32[384]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_36: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
    unsqueeze_37: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_111: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_37);  mul_105 = unsqueeze_37 = None
    unsqueeze_38: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
    unsqueeze_39: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_71: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_39);  mul_111 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:411, code: x = self.pos_drop(x + self.pos_embed2)
    add_72: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_71, primals_2);  add_71 = primals_2 = None
    clone_15: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_154, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 384, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 384, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_74: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_10: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(clone_15, getitem_21)
    mul_112: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_113: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_114: "f32[384]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
    add_75: "f32[384]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_32: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_115: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0006381620931717);  squeeze_32 = None
    mul_116: "f32[384]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[384]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_76: "f32[384]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_40: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_41: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_118: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_41);  mul_112 = unsqueeze_41 = None
    unsqueeze_42: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_43: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_77: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_43);  mul_118 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_24: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_77, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_24, [8, 3, 6, 64, -1]);  convolution_24 = None
    permute: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view, [1, 0, 2, 4, 3]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute);  permute = None
    getitem_22: "f32[8, 6, 196, 64]" = unbind[0]
    getitem_23: "f32[8, 6, 196, 64]" = unbind[1]
    getitem_24: "f32[8, 6, 196, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_1: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_23, [0, 1, 3, 2]);  getitem_23 = None
    expand: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_22, [8, 6, 196, 64]);  getitem_22 = None
    clone_16: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_1: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_16, [48, 196, 64]);  clone_16 = None
    expand_1: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_1, [8, 6, 64, 196]);  permute_1 = None
    clone_17: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_2: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_17, [48, 64, 196]);  clone_17 = None
    bmm: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_1, view_2)
    view_3: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm, [8, 6, 196, 196]);  bmm = None
    mul_119: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_119, [-1], True)
    sub_11: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_119, amax);  mul_119 = amax = None
    exp: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_1: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_1: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_18: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_2: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_18, [8, 6, 196, 196]);  clone_18 = None
    view_4: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_2, [48, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_24, [8, 6, 196, 64]);  getitem_24 = None
    clone_19: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_5: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_19, [48, 196, 64]);  clone_19 = None
    bmm_1: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_4, view_5)
    view_6: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_1, [8, 6, 196, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_2: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
    clone_20: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_7: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_20, [8, 384, 14, 14]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_25: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_7, primals_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_21: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_78: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(clone_15, clone_21);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_157, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_25: "f32[1, 384, 1, 1]" = var_mean_11[0]
    getitem_26: "f32[1, 384, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_80: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-05)
    rsqrt_11: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_12: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_78, getitem_26)
    mul_120: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = None
    squeeze_33: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    squeeze_34: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_121: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_122: "f32[384]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
    add_81: "f32[384]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
    squeeze_35: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    mul_123: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0006381620931717);  squeeze_35 = None
    mul_124: "f32[384]" = torch.ops.aten.mul.Tensor(mul_123, 0.1);  mul_123 = None
    mul_125: "f32[384]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_82: "f32[384]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    unsqueeze_44: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
    unsqueeze_45: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_126: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_45);  mul_120 = unsqueeze_45 = None
    unsqueeze_46: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
    unsqueeze_47: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_83: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_126, unsqueeze_47);  mul_126 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_83, primals_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_127: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.5)
    mul_128: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_84: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_129: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, add_84);  mul_127 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_22: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_129);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_27: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_22, primals_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_23: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_78, clone_23);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_86: "i64[]" = torch.ops.aten.add.Tensor(primals_160, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_27: "f32[1, 384, 1, 1]" = var_mean_12[0]
    getitem_28: "f32[1, 384, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_87: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_27, 1e-05)
    rsqrt_12: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_13: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_85, getitem_28)
    mul_130: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = None
    squeeze_36: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    squeeze_37: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_131: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_132: "f32[384]" = torch.ops.aten.mul.Tensor(primals_158, 0.9)
    add_88: "f32[384]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
    squeeze_38: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    mul_133: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0006381620931717);  squeeze_38 = None
    mul_134: "f32[384]" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
    mul_135: "f32[384]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_89: "f32[384]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    unsqueeze_48: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1)
    unsqueeze_49: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_136: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_49);  mul_130 = unsqueeze_49 = None
    unsqueeze_50: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1);  primals_59 = None
    unsqueeze_51: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_90: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_51);  mul_136 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_28: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_90, primals_60, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_8: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_28, [8, 3, 6, 64, -1]);  convolution_28 = None
    permute_3: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 4, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_29: "f32[8, 6, 196, 64]" = unbind_1[0]
    getitem_30: "f32[8, 6, 196, 64]" = unbind_1[1]
    getitem_31: "f32[8, 6, 196, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_4: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_30, [0, 1, 3, 2]);  getitem_30 = None
    expand_4: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_29, [8, 6, 196, 64]);  getitem_29 = None
    clone_24: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_9: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_24, [48, 196, 64]);  clone_24 = None
    expand_5: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_4, [8, 6, 64, 196]);  permute_4 = None
    clone_25: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_10: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_25, [48, 64, 196]);  clone_25 = None
    bmm_2: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_2, [8, 6, 196, 196]);  bmm_2 = None
    mul_137: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_11, 0.125);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_137, [-1], True)
    sub_14: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_137, amax_1);  mul_137 = amax_1 = None
    exp_1: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_2: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_26: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_6: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_26, [8, 6, 196, 196]);  clone_26 = None
    view_12: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_6, [48, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_31, [8, 6, 196, 64]);  getitem_31 = None
    clone_27: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_13: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_27, [48, 196, 64]);  clone_27 = None
    bmm_3: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_3, [8, 6, 196, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_5: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 2]);  view_14 = None
    clone_28: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_15: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_28, [8, 384, 14, 14]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_29: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_15, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_29: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_91: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_85, clone_29);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_163, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 384, 1, 1]" = var_mean_13[0]
    getitem_33: "f32[1, 384, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_93: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_13: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_15: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_91, getitem_33)
    mul_138: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
    squeeze_39: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_40: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_139: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_140: "f32[384]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
    add_94: "f32[384]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_41: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_141: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_142: "f32[384]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[384]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_95: "f32[384]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_52: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_53: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_144: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_53);  mul_138 = unsqueeze_53 = None
    unsqueeze_54: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_55: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_96: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_55);  mul_144 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_96, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_145: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, 0.5)
    mul_146: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476)
    erf_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_97: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_147: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, add_97);  mul_145 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_30: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_147);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_31: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_30, primals_65, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_31: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_98: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_91, clone_31);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_166, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 384, 1, 1]" = var_mean_14[0]
    getitem_35: "f32[1, 384, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_100: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_14: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_16: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_98, getitem_35)
    mul_148: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = None
    squeeze_42: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_43: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_149: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_150: "f32[384]" = torch.ops.aten.mul.Tensor(primals_164, 0.9)
    add_101: "f32[384]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    squeeze_44: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_151: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_152: "f32[384]" = torch.ops.aten.mul.Tensor(mul_151, 0.1);  mul_151 = None
    mul_153: "f32[384]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
    add_102: "f32[384]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    unsqueeze_56: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
    unsqueeze_57: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_154: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_57);  mul_148 = unsqueeze_57 = None
    unsqueeze_58: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
    unsqueeze_59: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_103: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_59);  mul_154 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_32: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_103, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_16: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_32, [8, 3, 6, 64, -1]);  convolution_32 = None
    permute_6: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_16, [1, 0, 2, 4, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
    getitem_36: "f32[8, 6, 196, 64]" = unbind_2[0]
    getitem_37: "f32[8, 6, 196, 64]" = unbind_2[1]
    getitem_38: "f32[8, 6, 196, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_7: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
    expand_8: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_36, [8, 6, 196, 64]);  getitem_36 = None
    clone_32: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_17: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_32, [48, 196, 64]);  clone_32 = None
    expand_9: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_7, [8, 6, 64, 196]);  permute_7 = None
    clone_33: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_18: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_33, [48, 64, 196]);  clone_33 = None
    bmm_4: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_17, view_18)
    view_19: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 6, 196, 196]);  bmm_4 = None
    mul_155: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_19, 0.125);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_155, [-1], True)
    sub_17: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_155, amax_2);  mul_155 = amax_2 = None
    exp_2: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_3: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_3: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_34: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_10: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_34, [8, 6, 196, 196]);  clone_34 = None
    view_20: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_10, [48, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_38, [8, 6, 196, 64]);  getitem_38 = None
    clone_35: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_21: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_35, [48, 196, 64]);  clone_35 = None
    bmm_5: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_20, view_21)
    view_22: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_5, [8, 6, 196, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_8: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_22, [0, 1, 3, 2]);  view_22 = None
    clone_36: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_23: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_36, [8, 384, 14, 14]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_33: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_23, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_37: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_33);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_104: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_98, clone_37);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_169, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_104, [0, 2, 3], correction = 0, keepdim = True)
    getitem_39: "f32[1, 384, 1, 1]" = var_mean_15[0]
    getitem_40: "f32[1, 384, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_106: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05)
    rsqrt_15: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_18: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_104, getitem_40)
    mul_156: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_15);  sub_18 = None
    squeeze_45: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    squeeze_46: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_157: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_158: "f32[384]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
    add_107: "f32[384]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    squeeze_47: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    mul_159: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_160: "f32[384]" = torch.ops.aten.mul.Tensor(mul_159, 0.1);  mul_159 = None
    mul_161: "f32[384]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
    add_108: "f32[384]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
    unsqueeze_60: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
    unsqueeze_61: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_162: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_61);  mul_156 = unsqueeze_61 = None
    unsqueeze_62: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
    unsqueeze_63: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_109: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_63);  mul_162 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_34: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_109, primals_72, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_163: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
    mul_164: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476)
    erf_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_165: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, add_110);  mul_163 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_165);  mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_35: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_38, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_39: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_111: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_104, clone_39);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_112: "i64[]" = torch.ops.aten.add.Tensor(primals_172, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_111, [0, 2, 3], correction = 0, keepdim = True)
    getitem_41: "f32[1, 384, 1, 1]" = var_mean_16[0]
    getitem_42: "f32[1, 384, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_113: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05)
    rsqrt_16: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_19: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_111, getitem_42)
    mul_166: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_16);  sub_19 = None
    squeeze_48: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    squeeze_49: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_167: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_168: "f32[384]" = torch.ops.aten.mul.Tensor(primals_170, 0.9)
    add_114: "f32[384]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
    squeeze_50: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    mul_169: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_170: "f32[384]" = torch.ops.aten.mul.Tensor(mul_169, 0.1);  mul_169 = None
    mul_171: "f32[384]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
    add_115: "f32[384]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
    unsqueeze_64: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_65: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_172: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_65);  mul_166 = unsqueeze_65 = None
    unsqueeze_66: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_67: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_116: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_172, unsqueeze_67);  mul_172 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_36: "f32[8, 1152, 14, 14]" = torch.ops.aten.convolution.default(add_116, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_24: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.view.default(convolution_36, [8, 3, 6, 64, -1]);  convolution_36 = None
    permute_9: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.permute.default(view_24, [1, 0, 2, 4, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_43: "f32[8, 6, 196, 64]" = unbind_3[0]
    getitem_44: "f32[8, 6, 196, 64]" = unbind_3[1]
    getitem_45: "f32[8, 6, 196, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_10: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
    expand_12: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_43, [8, 6, 196, 64]);  getitem_43 = None
    clone_40: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_25: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_40, [48, 196, 64]);  clone_40 = None
    expand_13: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_10, [8, 6, 64, 196]);  permute_10 = None
    clone_41: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_26: "f32[48, 64, 196]" = torch.ops.aten.view.default(clone_41, [48, 64, 196]);  clone_41 = None
    bmm_6: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 6, 196, 196]);  bmm_6 = None
    mul_173: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_27, 0.125);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 6, 196, 1]" = torch.ops.aten.amax.default(mul_173, [-1], True)
    sub_20: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_173, amax_3);  mul_173 = amax_3 = None
    exp_3: "f32[8, 6, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_4: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_4: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_42: "f32[8, 6, 196, 196]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_14: "f32[8, 6, 196, 196]" = torch.ops.aten.expand.default(clone_42, [8, 6, 196, 196]);  clone_42 = None
    view_28: "f32[48, 196, 196]" = torch.ops.aten.view.default(expand_14, [48, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_45, [8, 6, 196, 64]);  getitem_45 = None
    clone_43: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_29: "f32[48, 196, 64]" = torch.ops.aten.view.default(clone_43, [48, 196, 64]);  clone_43 = None
    bmm_7: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_28, view_29)
    view_30: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_7, [8, 6, 196, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_11: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_30, [0, 1, 3, 2]);  view_30 = None
    clone_44: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_31: "f32[8, 384, 14, 14]" = torch.ops.aten.view.default(clone_44, [8, 384, 14, 14]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_37: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_31, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_45: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_117: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_111, clone_45);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_118: "i64[]" = torch.ops.aten.add.Tensor(primals_175, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_117, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 384, 1, 1]" = var_mean_17[0]
    getitem_47: "f32[1, 384, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_119: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_17: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_21: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_117, getitem_47)
    mul_174: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_17);  sub_21 = None
    squeeze_51: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_52: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_175: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_176: "f32[384]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
    add_120: "f32[384]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    squeeze_53: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_177: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_178: "f32[384]" = torch.ops.aten.mul.Tensor(mul_177, 0.1);  mul_177 = None
    mul_179: "f32[384]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
    add_121: "f32[384]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
    unsqueeze_68: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
    unsqueeze_69: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_180: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_69);  mul_174 = unsqueeze_69 = None
    unsqueeze_70: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
    unsqueeze_71: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_122: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_180, unsqueeze_71);  mul_180 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_38: "f32[8, 1536, 14, 14]" = torch.ops.aten.convolution.default(add_122, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_181: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
    mul_182: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_123: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_183: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, add_123);  mul_181 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_46: "f32[8, 1536, 14, 14]" = torch.ops.aten.clone.default(mul_183);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_39: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(clone_46, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_47: "f32[8, 384, 14, 14]" = torch.ops.aten.clone.default(convolution_39);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_124: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_117, clone_47);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_40: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(add_124, primals_82, primals_83, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_178, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 768, 1, 1]" = var_mean_18[0]
    getitem_49: "f32[1, 768, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_126: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_18: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_22: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
    mul_184: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = None
    squeeze_54: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_55: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_185: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_186: "f32[768]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
    add_127: "f32[768]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_56: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_187: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0025575447570332);  squeeze_56 = None
    mul_188: "f32[768]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[768]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
    add_128: "f32[768]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_72: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
    unsqueeze_73: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_190: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_73);  mul_184 = unsqueeze_73 = None
    unsqueeze_74: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
    unsqueeze_75: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_129: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_75);  mul_190 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:421, code: x = self.pos_drop(x + self.pos_embed3)
    add_130: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_129, primals_3);  add_129 = primals_3 = None
    clone_48: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(add_130);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_131: "i64[]" = torch.ops.aten.add.Tensor(primals_181, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 768, 1, 1]" = var_mean_19[0]
    getitem_51: "f32[1, 768, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_132: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_19: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_23: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(clone_48, getitem_51)
    mul_191: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_19);  sub_23 = None
    squeeze_57: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_58: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_192: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_193: "f32[768]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
    add_133: "f32[768]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    squeeze_59: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_194: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0025575447570332);  squeeze_59 = None
    mul_195: "f32[768]" = torch.ops.aten.mul.Tensor(mul_194, 0.1);  mul_194 = None
    mul_196: "f32[768]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
    add_134: "f32[768]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    unsqueeze_76: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_77: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_197: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_77);  mul_191 = unsqueeze_77 = None
    unsqueeze_78: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_79: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_135: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_79);  mul_197 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_41: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_135, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_32: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_41, [8, 3, 6, 128, -1]);  convolution_41 = None
    permute_12: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_32, [1, 0, 2, 4, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_12);  permute_12 = None
    getitem_52: "f32[8, 6, 49, 128]" = unbind_4[0]
    getitem_53: "f32[8, 6, 49, 128]" = unbind_4[1]
    getitem_54: "f32[8, 6, 49, 128]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_13: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_53, [0, 1, 3, 2]);  getitem_53 = None
    expand_16: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_52, [8, 6, 49, 128]);  getitem_52 = None
    clone_49: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_33: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_49, [48, 49, 128]);  clone_49 = None
    expand_17: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_13, [8, 6, 128, 49]);  permute_13 = None
    clone_50: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_34: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_50, [48, 128, 49]);  clone_50 = None
    bmm_8: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_8, [8, 6, 49, 49]);  bmm_8 = None
    mul_198: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_35, 0.08838834764831845);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_198, [-1], True)
    sub_24: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_198, amax_4);  mul_198 = amax_4 = None
    exp_4: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_5: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_5: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_51: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_18: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_51, [8, 6, 49, 49]);  clone_51 = None
    view_36: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_18, [48, 49, 49]);  expand_18 = None
    expand_19: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_54, [8, 6, 49, 128]);  getitem_54 = None
    clone_52: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_37: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_52, [48, 49, 128]);  clone_52 = None
    bmm_9: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_9, [8, 6, 49, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_14: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_38, [0, 1, 3, 2]);  view_38 = None
    clone_53: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_39: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_53, [8, 768, 7, 7]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_42: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_39, primals_89, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_54: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_136: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(clone_48, clone_54);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_184, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_136, [0, 2, 3], correction = 0, keepdim = True)
    getitem_55: "f32[1, 768, 1, 1]" = var_mean_20[0]
    getitem_56: "f32[1, 768, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_138: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05)
    rsqrt_20: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_25: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_136, getitem_56)
    mul_199: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_20);  sub_25 = None
    squeeze_60: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    squeeze_61: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_200: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_201: "f32[768]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
    add_139: "f32[768]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    squeeze_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    mul_202: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0025575447570332);  squeeze_62 = None
    mul_203: "f32[768]" = torch.ops.aten.mul.Tensor(mul_202, 0.1);  mul_202 = None
    mul_204: "f32[768]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
    add_140: "f32[768]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    unsqueeze_80: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
    unsqueeze_81: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_205: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_81);  mul_199 = unsqueeze_81 = None
    unsqueeze_82: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
    unsqueeze_83: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_141: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_83);  mul_205 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_43: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_141, primals_92, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_206: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, 0.5)
    mul_207: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_18: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_142: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_208: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_206, add_142);  mul_206 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_55: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_208);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_44: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_55, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_56: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_44);  convolution_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_143: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_136, clone_56);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_144: "i64[]" = torch.ops.aten.add.Tensor(primals_187, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_143, [0, 2, 3], correction = 0, keepdim = True)
    getitem_57: "f32[1, 768, 1, 1]" = var_mean_21[0]
    getitem_58: "f32[1, 768, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_145: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05)
    rsqrt_21: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_26: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_143, getitem_58)
    mul_209: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_21);  sub_26 = None
    squeeze_63: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    squeeze_64: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_210: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_211: "f32[768]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
    add_146: "f32[768]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    squeeze_65: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    mul_212: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0025575447570332);  squeeze_65 = None
    mul_213: "f32[768]" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
    mul_214: "f32[768]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
    add_147: "f32[768]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    unsqueeze_84: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
    unsqueeze_85: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_215: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_85);  mul_209 = unsqueeze_85 = None
    unsqueeze_86: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
    unsqueeze_87: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_148: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_87);  mul_215 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_45: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_148, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_40: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_45, [8, 3, 6, 128, -1]);  convolution_45 = None
    permute_15: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_40, [1, 0, 2, 4, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_59: "f32[8, 6, 49, 128]" = unbind_5[0]
    getitem_60: "f32[8, 6, 49, 128]" = unbind_5[1]
    getitem_61: "f32[8, 6, 49, 128]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_16: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_60, [0, 1, 3, 2]);  getitem_60 = None
    expand_20: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_59, [8, 6, 49, 128]);  getitem_59 = None
    clone_57: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_41: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_57, [48, 49, 128]);  clone_57 = None
    expand_21: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_16, [8, 6, 128, 49]);  permute_16 = None
    clone_58: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_42: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_58, [48, 128, 49]);  clone_58 = None
    bmm_10: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_10, [8, 6, 49, 49]);  bmm_10 = None
    mul_216: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_43, 0.08838834764831845);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_216, [-1], True)
    sub_27: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_216, amax_5);  mul_216 = amax_5 = None
    exp_5: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_6: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_6: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_59: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_22: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_59, [8, 6, 49, 49]);  clone_59 = None
    view_44: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_22, [48, 49, 49]);  expand_22 = None
    expand_23: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_61, [8, 6, 49, 128]);  getitem_61 = None
    clone_60: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_45: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_60, [48, 49, 128]);  clone_60 = None
    bmm_11: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_11, [8, 6, 49, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_17: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2]);  view_46 = None
    clone_61: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_47: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_61, [8, 768, 7, 7]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_46: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_47, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_62: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_149: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_143, clone_62);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_190, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_149, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 768, 1, 1]" = var_mean_22[0]
    getitem_63: "f32[1, 768, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_151: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_22: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_28: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_149, getitem_63)
    mul_217: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_22);  sub_28 = None
    squeeze_66: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_67: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_218: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_219: "f32[768]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_152: "f32[768]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_68: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_220: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0025575447570332);  squeeze_68 = None
    mul_221: "f32[768]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[768]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_153: "f32[768]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_88: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_89: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_223: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_89);  mul_217 = unsqueeze_89 = None
    unsqueeze_90: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_91: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_154: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_91);  mul_223 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_47: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_154, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_224: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, 0.5)
    mul_225: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476)
    erf_19: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_225);  mul_225 = None
    add_155: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_226: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_224, add_155);  mul_224 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_63: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_226);  mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_48: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_63, primals_101, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_64: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_156: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_149, clone_64);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_193, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_156, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 768, 1, 1]" = var_mean_23[0]
    getitem_65: "f32[1, 768, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_158: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_23: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_29: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_156, getitem_65)
    mul_227: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
    squeeze_69: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_70: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_228: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_229: "f32[768]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_159: "f32[768]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    squeeze_71: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_230: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0025575447570332);  squeeze_71 = None
    mul_231: "f32[768]" = torch.ops.aten.mul.Tensor(mul_230, 0.1);  mul_230 = None
    mul_232: "f32[768]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_160: "f32[768]" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    unsqueeze_92: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
    unsqueeze_93: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_233: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_93);  mul_227 = unsqueeze_93 = None
    unsqueeze_94: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
    unsqueeze_95: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_161: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_95);  mul_233 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_49: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_161, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_48: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_49, [8, 3, 6, 128, -1]);  convolution_49 = None
    permute_18: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_48, [1, 0, 2, 4, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
    getitem_66: "f32[8, 6, 49, 128]" = unbind_6[0]
    getitem_67: "f32[8, 6, 49, 128]" = unbind_6[1]
    getitem_68: "f32[8, 6, 49, 128]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_19: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_67, [0, 1, 3, 2]);  getitem_67 = None
    expand_24: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_66, [8, 6, 49, 128]);  getitem_66 = None
    clone_65: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_49: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_65, [48, 49, 128]);  clone_65 = None
    expand_25: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_19, [8, 6, 128, 49]);  permute_19 = None
    clone_66: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_50: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_66, [48, 128, 49]);  clone_66 = None
    bmm_12: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_49, view_50)
    view_51: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_12, [8, 6, 49, 49]);  bmm_12 = None
    mul_234: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_51, 0.08838834764831845);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_234, [-1], True)
    sub_30: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_234, amax_6);  mul_234 = amax_6 = None
    exp_6: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_7: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_7: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_67: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_26: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_67, [8, 6, 49, 49]);  clone_67 = None
    view_52: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_26, [48, 49, 49]);  expand_26 = None
    expand_27: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_68, [8, 6, 49, 128]);  getitem_68 = None
    clone_68: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_53: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_68, [48, 49, 128]);  clone_68 = None
    bmm_13: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_52, view_53)
    view_54: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_13, [8, 6, 49, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_20: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 2]);  view_54 = None
    clone_69: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_55: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_69, [8, 768, 7, 7]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_50: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_55, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_70: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_50);  convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_162: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_156, clone_70);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_196, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_162, [0, 2, 3], correction = 0, keepdim = True)
    getitem_69: "f32[1, 768, 1, 1]" = var_mean_24[0]
    getitem_70: "f32[1, 768, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_164: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
    rsqrt_24: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_31: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_162, getitem_70)
    mul_235: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_24);  sub_31 = None
    squeeze_72: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    squeeze_73: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_236: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_237: "f32[768]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_165: "f32[768]" = torch.ops.aten.add.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    squeeze_74: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    mul_238: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0025575447570332);  squeeze_74 = None
    mul_239: "f32[768]" = torch.ops.aten.mul.Tensor(mul_238, 0.1);  mul_238 = None
    mul_240: "f32[768]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_166: "f32[768]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    unsqueeze_96: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1)
    unsqueeze_97: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_241: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_97);  mul_235 = unsqueeze_97 = None
    unsqueeze_98: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1);  primals_107 = None
    unsqueeze_99: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_167: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_241, unsqueeze_99);  mul_241 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_51: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_167, primals_108, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_242: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, 0.5)
    mul_243: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_20: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
    add_168: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_244: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_242, add_168);  mul_242 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_71: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_244);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_52: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_71, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_72: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_169: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_162, clone_72);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_199, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_169, [0, 2, 3], correction = 0, keepdim = True)
    getitem_71: "f32[1, 768, 1, 1]" = var_mean_25[0]
    getitem_72: "f32[1, 768, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_171: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_25: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_32: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_169, getitem_72)
    mul_245: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_25);  sub_32 = None
    squeeze_75: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    squeeze_76: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_246: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_247: "f32[768]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
    add_172: "f32[768]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_77: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    mul_248: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0025575447570332);  squeeze_77 = None
    mul_249: "f32[768]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[768]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_173: "f32[768]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_100: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_101: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_251: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_101);  mul_245 = unsqueeze_101 = None
    unsqueeze_102: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_103: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_174: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_103);  mul_251 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    convolution_53: "f32[8, 2304, 7, 7]" = torch.ops.aten.convolution.default(add_174, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    view_56: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_53, [8, 3, 6, 128, -1]);  convolution_53 = None
    permute_21: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.permute.default(view_56, [1, 0, 2, 4, 3]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_21);  permute_21 = None
    getitem_73: "f32[8, 6, 49, 128]" = unbind_7[0]
    getitem_74: "f32[8, 6, 49, 128]" = unbind_7[1]
    getitem_75: "f32[8, 6, 49, 128]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_22: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_74, [0, 1, 3, 2]);  getitem_74 = None
    expand_28: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_73, [8, 6, 49, 128]);  getitem_73 = None
    clone_73: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_57: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_73, [48, 49, 128]);  clone_73 = None
    expand_29: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_22, [8, 6, 128, 49]);  permute_22 = None
    clone_74: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_58: "f32[48, 128, 49]" = torch.ops.aten.view.default(clone_74, [48, 128, 49]);  clone_74 = None
    bmm_14: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_57, view_58)
    view_59: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_14, [8, 6, 49, 49]);  bmm_14 = None
    mul_252: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_59, 0.08838834764831845);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 6, 49, 1]" = torch.ops.aten.amax.default(mul_252, [-1], True)
    sub_33: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_252, amax_7);  mul_252 = amax_7 = None
    exp_7: "f32[8, 6, 49, 49]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_8: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 6, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_8: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:103, code: attn = self.attn_drop(attn)
    clone_75: "f32[8, 6, 49, 49]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    expand_30: "f32[8, 6, 49, 49]" = torch.ops.aten.expand.default(clone_75, [8, 6, 49, 49]);  clone_75 = None
    view_60: "f32[48, 49, 49]" = torch.ops.aten.view.default(expand_30, [48, 49, 49]);  expand_30 = None
    expand_31: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_75, [8, 6, 49, 128]);  getitem_75 = None
    clone_76: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_61: "f32[48, 49, 128]" = torch.ops.aten.view.default(clone_76, [48, 49, 128]);  clone_76 = None
    bmm_15: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_60, view_61)
    view_62: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_15, [8, 6, 49, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_23: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_62, [0, 1, 3, 2]);  view_62 = None
    clone_77: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_63: "f32[8, 768, 7, 7]" = torch.ops.aten.view.default(clone_77, [8, 768, 7, 7]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_54: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_63, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:108, code: x = self.proj_drop(x)
    clone_78: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_175: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_169, clone_78);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_202, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_175, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 768, 1, 1]" = var_mean_26[0]
    getitem_77: "f32[1, 768, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_177: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_26: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_34: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_175, getitem_77)
    mul_253: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_26);  sub_34 = None
    squeeze_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_254: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_255: "f32[768]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_178: "f32[768]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    squeeze_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_256: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0025575447570332);  squeeze_80 = None
    mul_257: "f32[768]" = torch.ops.aten.mul.Tensor(mul_256, 0.1);  mul_256 = None
    mul_258: "f32[768]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_179: "f32[768]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    unsqueeze_104: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
    unsqueeze_105: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_259: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_105);  mul_253 = unsqueeze_105 = None
    unsqueeze_106: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
    unsqueeze_107: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_180: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_107);  mul_259 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_55: "f32[8, 3072, 7, 7]" = torch.ops.aten.convolution.default(add_180, primals_116, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_260: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.5)
    mul_261: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_21: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
    add_181: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_262: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_260, add_181);  mul_260 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:64, code: x = self.drop1(x)
    clone_79: "f32[8, 3072, 7, 7]" = torch.ops.aten.clone.default(mul_262);  mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_56: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(clone_79, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:69, code: x = self.drop3(x)
    clone_80: "f32[8, 768, 7, 7]" = torch.ops.aten.clone.default(convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_182: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_175, clone_80);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_205, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_182, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 768, 1, 1]" = var_mean_27[0]
    getitem_79: "f32[1, 768, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_184: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_27: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_35: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, getitem_79)
    mul_263: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_27);  sub_35 = None
    squeeze_81: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_82: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_264: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_265: "f32[768]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_185: "f32[768]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    squeeze_83: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_266: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0025575447570332);  squeeze_83 = None
    mul_267: "f32[768]" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
    mul_268: "f32[768]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_186: "f32[768]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    unsqueeze_108: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
    unsqueeze_109: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_269: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_109);  mul_263 = unsqueeze_109 = None
    unsqueeze_110: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
    unsqueeze_111: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_187: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_111);  mul_269 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_187, [-1, -2], True);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_64: "f32[8, 768]" = torch.ops.aten.view.default(mean, [8, 768]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:432, code: x = self.head_drop(x)
    clone_81: "f32[8, 768]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:433, code: return x if pre_logits else self.head(x)
    permute_24: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_121, clone_81, permute_24);  primals_121 = None
    permute_25: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_25);  permute_25 = None
    permute_26: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_26, clone_81);  permute_26 = clone_81 = None
    permute_27: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_9: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_65: "f32[1000]" = torch.ops.aten.view.default(sum_9, [1000]);  sum_9 = None
    permute_28: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_66: "f32[8, 768, 1, 1]" = torch.ops.aten.view.default(mm, [8, 768, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_32: "f32[8, 768, 7, 7]" = torch.ops.aten.expand.default(view_66, [8, 768, 7, 7]);  view_66 = None
    div_8: "f32[8, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand_32, 49);  expand_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    unsqueeze_112: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_113: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None
    unsqueeze_114: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 3);  unsqueeze_113 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(div_8, [0, 2, 3])
    sub_36: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, unsqueeze_114)
    mul_270: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div_8, sub_36);  sub_36 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 2, 3]);  mul_270 = None
    mul_271: "f32[768]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_115: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
    unsqueeze_116: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
    unsqueeze_117: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 3);  unsqueeze_116 = None
    mul_272: "f32[768]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_273: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_274: "f32[768]" = torch.ops.aten.mul.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    unsqueeze_118: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_119: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 2);  unsqueeze_118 = None
    unsqueeze_120: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 3);  unsqueeze_119 = None
    mul_275: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_118);  primals_118 = None
    unsqueeze_121: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
    unsqueeze_122: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
    unsqueeze_123: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
    sub_37: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, unsqueeze_114);  add_182 = unsqueeze_114 = None
    mul_276: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_120);  sub_37 = unsqueeze_120 = None
    sub_38: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(div_8, mul_276);  div_8 = mul_276 = None
    sub_39: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_38, unsqueeze_117);  sub_38 = unsqueeze_117 = None
    mul_277: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_123);  sub_39 = unsqueeze_123 = None
    mul_278: "f32[768]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_82);  sum_11 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_277, clone_79, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_79 = primals_117 = None
    getitem_80: "f32[8, 3072, 7, 7]" = convolution_backward[0]
    getitem_81: "f32[768, 3072, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_279: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_22: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_279);  mul_279 = None
    add_188: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_280: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_188, 0.5);  add_188 = None
    mul_281: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, convolution_55)
    mul_282: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_281, -0.5);  mul_281 = None
    exp_8: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_282);  mul_282 = None
    mul_283: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_284: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, mul_283);  convolution_55 = mul_283 = None
    add_189: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_280, mul_284);  mul_280 = mul_284 = None
    mul_285: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_80, add_189);  getitem_80 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_285, add_180, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_285 = add_180 = primals_116 = None
    getitem_83: "f32[8, 768, 7, 7]" = convolution_backward_1[0]
    getitem_84: "f32[3072, 768, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_124: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_125: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
    unsqueeze_126: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_83, [0, 2, 3])
    sub_40: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_175, unsqueeze_126)
    mul_286: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_83, sub_40);  sub_40 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 2, 3]);  mul_286 = None
    mul_287: "f32[768]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_127: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
    unsqueeze_128: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
    mul_288: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_289: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_290: "f32[768]" = torch.ops.aten.mul.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_130: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_131: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
    unsqueeze_132: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
    mul_291: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_114);  primals_114 = None
    unsqueeze_133: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_134: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
    sub_41: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_175, unsqueeze_126);  add_175 = unsqueeze_126 = None
    mul_292: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_132);  sub_41 = unsqueeze_132 = None
    sub_42: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_83, mul_292);  getitem_83 = mul_292 = None
    sub_43: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_129);  sub_42 = unsqueeze_129 = None
    mul_293: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_135);  sub_43 = unsqueeze_135 = None
    mul_294: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_79);  sum_13 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_190: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_277, mul_293);  mul_277 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(add_190, view_63, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_63 = primals_113 = None
    getitem_86: "f32[8, 768, 7, 7]" = convolution_backward_2[0]
    getitem_87: "f32[768, 768, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_67: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(getitem_86, [8, 6, 128, 49]);  getitem_86 = None
    permute_29: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_67, [0, 1, 3, 2]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_68: "f32[48, 49, 128]" = torch.ops.aten.view.default(permute_29, [48, 49, 128]);  permute_29 = None
    permute_30: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_16: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_30, view_68);  permute_30 = None
    permute_31: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_17: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_68, permute_31);  view_68 = permute_31 = None
    view_69: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_16, [8, 6, 49, 128]);  bmm_16 = None
    view_70: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_17, [8, 6, 49, 49]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_9: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_295: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_70, alias_9);  view_70 = None
    sum_14: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [-1], True)
    mul_296: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_9, sum_14);  alias_9 = sum_14 = None
    sub_44: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_297: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_44, 0.08838834764831845);  sub_44 = None
    view_71: "f32[48, 49, 49]" = torch.ops.aten.view.default(mul_297, [48, 49, 49]);  mul_297 = None
    permute_32: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_18: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_32, view_71);  permute_32 = None
    permute_33: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_19: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_71, permute_33);  view_71 = permute_33 = None
    view_72: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(bmm_18, [8, 6, 128, 49]);  bmm_18 = None
    view_73: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_19, [8, 6, 49, 128]);  bmm_19 = None
    permute_34: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_72, [0, 1, 3, 2]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_73, permute_34, view_69]);  view_73 = permute_34 = view_69 = None
    view_74: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.view.default(cat, [3, 8, 6, 49, 128]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_35: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_74, [1, 0, 2, 4, 3]);  view_74 = None
    clone_82: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_75: "f32[8, 2304, 7, 7]" = torch.ops.aten.view.default(clone_82, [8, 2304, 7, 7]);  clone_82 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_75, add_174, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_75 = add_174 = primals_112 = None
    getitem_89: "f32[8, 768, 7, 7]" = convolution_backward_3[0]
    getitem_90: "f32[2304, 768, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_136: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_137: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
    unsqueeze_138: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_89, [0, 2, 3])
    sub_45: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_169, unsqueeze_138)
    mul_298: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_89, sub_45);  sub_45 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3]);  mul_298 = None
    mul_299: "f32[768]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    unsqueeze_139: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
    unsqueeze_140: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    mul_300: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    mul_301: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_302: "f32[768]" = torch.ops.aten.mul.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_142: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_143: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
    unsqueeze_144: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
    mul_303: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_110);  primals_110 = None
    unsqueeze_145: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
    unsqueeze_146: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
    sub_46: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_169, unsqueeze_138);  add_169 = unsqueeze_138 = None
    mul_304: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_144);  sub_46 = unsqueeze_144 = None
    sub_47: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_89, mul_304);  getitem_89 = mul_304 = None
    sub_48: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_47, unsqueeze_141);  sub_47 = unsqueeze_141 = None
    mul_305: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_147);  sub_48 = unsqueeze_147 = None
    mul_306: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_76);  sum_16 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_191: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_190, mul_305);  add_190 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(add_191, clone_71, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_71 = primals_109 = None
    getitem_92: "f32[8, 3072, 7, 7]" = convolution_backward_4[0]
    getitem_93: "f32[768, 3072, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_307: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_23: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_307);  mul_307 = None
    add_192: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_308: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
    mul_309: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, convolution_51)
    mul_310: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_309, -0.5);  mul_309 = None
    exp_9: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_310);  mul_310 = None
    mul_311: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_312: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, mul_311);  convolution_51 = mul_311 = None
    add_193: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_308, mul_312);  mul_308 = mul_312 = None
    mul_313: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_92, add_193);  getitem_92 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_313, add_167, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_313 = add_167 = primals_108 = None
    getitem_95: "f32[8, 768, 7, 7]" = convolution_backward_5[0]
    getitem_96: "f32[3072, 768, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_148: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_149: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
    unsqueeze_150: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_95, [0, 2, 3])
    sub_49: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_162, unsqueeze_150)
    mul_314: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_95, sub_49);  sub_49 = None
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3]);  mul_314 = None
    mul_315: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    unsqueeze_151: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
    unsqueeze_152: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
    mul_316: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    mul_317: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_318: "f32[768]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    unsqueeze_154: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_155: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
    unsqueeze_156: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
    mul_319: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_106);  primals_106 = None
    unsqueeze_157: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_158: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
    sub_50: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_162, unsqueeze_150);  add_162 = unsqueeze_150 = None
    mul_320: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_156);  sub_50 = unsqueeze_156 = None
    sub_51: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_95, mul_320);  getitem_95 = mul_320 = None
    sub_52: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_51, unsqueeze_153);  sub_51 = unsqueeze_153 = None
    mul_321: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_159);  sub_52 = unsqueeze_159 = None
    mul_322: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_73);  sum_18 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_194: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_191, mul_321);  add_191 = mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_194, view_55, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_55 = primals_105 = None
    getitem_98: "f32[8, 768, 7, 7]" = convolution_backward_6[0]
    getitem_99: "f32[768, 768, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_76: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(getitem_98, [8, 6, 128, 49]);  getitem_98 = None
    permute_36: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_77: "f32[48, 49, 128]" = torch.ops.aten.view.default(permute_36, [48, 49, 128]);  permute_36 = None
    permute_37: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    bmm_20: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_37, view_77);  permute_37 = None
    permute_38: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_21: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_77, permute_38);  view_77 = permute_38 = None
    view_78: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_20, [8, 6, 49, 128]);  bmm_20 = None
    view_79: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_21, [8, 6, 49, 49]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_10: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_323: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_79, alias_10);  view_79 = None
    sum_19: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
    mul_324: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_10, sum_19);  alias_10 = sum_19 = None
    sub_53: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_325: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_53, 0.08838834764831845);  sub_53 = None
    view_80: "f32[48, 49, 49]" = torch.ops.aten.view.default(mul_325, [48, 49, 49]);  mul_325 = None
    permute_39: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
    bmm_22: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_39, view_80);  permute_39 = None
    permute_40: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    bmm_23: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_80, permute_40);  view_80 = permute_40 = None
    view_81: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(bmm_22, [8, 6, 128, 49]);  bmm_22 = None
    view_82: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_23, [8, 6, 49, 128]);  bmm_23 = None
    permute_41: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_81, [0, 1, 3, 2]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_1: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_82, permute_41, view_78]);  view_82 = permute_41 = view_78 = None
    view_83: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.view.default(cat_1, [3, 8, 6, 49, 128]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_42: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_83, [1, 0, 2, 4, 3]);  view_83 = None
    clone_83: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_84: "f32[8, 2304, 7, 7]" = torch.ops.aten.view.default(clone_83, [8, 2304, 7, 7]);  clone_83 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_84, add_161, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_84 = add_161 = primals_104 = None
    getitem_101: "f32[8, 768, 7, 7]" = convolution_backward_7[0]
    getitem_102: "f32[2304, 768, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_160: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_161: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_101, [0, 2, 3])
    sub_54: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_156, unsqueeze_162)
    mul_326: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, sub_54);  sub_54 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3]);  mul_326 = None
    mul_327: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_163: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
    unsqueeze_164: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
    mul_328: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_329: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_330: "f32[768]" = torch.ops.aten.mul.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
    unsqueeze_166: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_167: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
    unsqueeze_168: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    mul_331: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_102);  primals_102 = None
    unsqueeze_169: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_170: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    sub_55: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_156, unsqueeze_162);  add_156 = unsqueeze_162 = None
    mul_332: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_168);  sub_55 = unsqueeze_168 = None
    sub_56: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_101, mul_332);  getitem_101 = mul_332 = None
    sub_57: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_165);  sub_56 = unsqueeze_165 = None
    mul_333: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_171);  sub_57 = unsqueeze_171 = None
    mul_334: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_70);  sum_21 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_195: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_194, mul_333);  add_194 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(add_195, clone_63, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_63 = primals_101 = None
    getitem_104: "f32[8, 3072, 7, 7]" = convolution_backward_8[0]
    getitem_105: "f32[768, 3072, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_335: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476)
    erf_24: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_335);  mul_335 = None
    add_196: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_336: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_337: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, convolution_47)
    mul_338: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_337, -0.5);  mul_337 = None
    exp_10: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_338);  mul_338 = None
    mul_339: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_340: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, mul_339);  convolution_47 = mul_339 = None
    add_197: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_336, mul_340);  mul_336 = mul_340 = None
    mul_341: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_104, add_197);  getitem_104 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_341, add_154, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_341 = add_154 = primals_100 = None
    getitem_107: "f32[8, 768, 7, 7]" = convolution_backward_9[0]
    getitem_108: "f32[3072, 768, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_172: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_173: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_107, [0, 2, 3])
    sub_58: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_149, unsqueeze_174)
    mul_342: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_107, sub_58);  sub_58 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 2, 3]);  mul_342 = None
    mul_343: "f32[768]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_175: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
    unsqueeze_176: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    unsqueeze_177: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
    mul_344: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_345: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_346: "f32[768]" = torch.ops.aten.mul.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    unsqueeze_178: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_179: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
    unsqueeze_180: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
    mul_347: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_98);  primals_98 = None
    unsqueeze_181: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_182: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    sub_59: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_149, unsqueeze_174);  add_149 = unsqueeze_174 = None
    mul_348: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_180);  sub_59 = unsqueeze_180 = None
    sub_60: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_107, mul_348);  getitem_107 = mul_348 = None
    sub_61: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_177);  sub_60 = unsqueeze_177 = None
    mul_349: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_183);  sub_61 = unsqueeze_183 = None
    mul_350: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_67);  sum_23 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_198: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_195, mul_349);  add_195 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_198, view_47, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_47 = primals_97 = None
    getitem_110: "f32[8, 768, 7, 7]" = convolution_backward_10[0]
    getitem_111: "f32[768, 768, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_85: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(getitem_110, [8, 6, 128, 49]);  getitem_110 = None
    permute_43: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_85, [0, 1, 3, 2]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_86: "f32[48, 49, 128]" = torch.ops.aten.view.default(permute_43, [48, 49, 128]);  permute_43 = None
    permute_44: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_24: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_44, view_86);  permute_44 = None
    permute_45: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_25: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_86, permute_45);  view_86 = permute_45 = None
    view_87: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_24, [8, 6, 49, 128]);  bmm_24 = None
    view_88: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_25, [8, 6, 49, 49]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_11: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_351: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_88, alias_11);  view_88 = None
    sum_24: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
    mul_352: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_11, sum_24);  alias_11 = sum_24 = None
    sub_62: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_353: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_62, 0.08838834764831845);  sub_62 = None
    view_89: "f32[48, 49, 49]" = torch.ops.aten.view.default(mul_353, [48, 49, 49]);  mul_353 = None
    permute_46: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    bmm_26: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_46, view_89);  permute_46 = None
    permute_47: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    bmm_27: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_89, permute_47);  view_89 = permute_47 = None
    view_90: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(bmm_26, [8, 6, 128, 49]);  bmm_26 = None
    view_91: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_27, [8, 6, 49, 128]);  bmm_27 = None
    permute_48: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_90, [0, 1, 3, 2]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_2: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_91, permute_48, view_87]);  view_91 = permute_48 = view_87 = None
    view_92: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.view.default(cat_2, [3, 8, 6, 49, 128]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_49: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_92, [1, 0, 2, 4, 3]);  view_92 = None
    clone_84: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_93: "f32[8, 2304, 7, 7]" = torch.ops.aten.view.default(clone_84, [8, 2304, 7, 7]);  clone_84 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_93, add_148, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_93 = add_148 = primals_96 = None
    getitem_113: "f32[8, 768, 7, 7]" = convolution_backward_11[0]
    getitem_114: "f32[2304, 768, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_184: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_185: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
    sub_63: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_143, unsqueeze_186)
    mul_354: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_63);  sub_63 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
    mul_355: "f32[768]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_187: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
    unsqueeze_188: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_356: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_357: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_358: "f32[768]" = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    unsqueeze_190: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_191: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_359: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_94);  primals_94 = None
    unsqueeze_193: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
    unsqueeze_194: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    sub_64: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_143, unsqueeze_186);  add_143 = unsqueeze_186 = None
    mul_360: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_192);  sub_64 = unsqueeze_192 = None
    sub_65: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_360);  getitem_113 = mul_360 = None
    sub_66: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_189);  sub_65 = unsqueeze_189 = None
    mul_361: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_195);  sub_66 = unsqueeze_195 = None
    mul_362: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_64);  sum_26 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_199: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_198, mul_361);  add_198 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(add_199, clone_55, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_55 = primals_93 = None
    getitem_116: "f32[8, 3072, 7, 7]" = convolution_backward_12[0]
    getitem_117: "f32[768, 3072, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_363: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_25: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_363);  mul_363 = None
    add_200: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_364: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_200, 0.5);  add_200 = None
    mul_365: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, convolution_43)
    mul_366: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
    exp_11: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
    mul_367: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_368: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, mul_367);  convolution_43 = mul_367 = None
    add_201: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
    mul_369: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_116, add_201);  getitem_116 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_369, add_141, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_369 = add_141 = primals_92 = None
    getitem_119: "f32[8, 768, 7, 7]" = convolution_backward_13[0]
    getitem_120: "f32[3072, 768, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_196: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_197: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_119, [0, 2, 3])
    sub_67: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_136, unsqueeze_198)
    mul_370: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_119, sub_67);  sub_67 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 2, 3]);  mul_370 = None
    mul_371: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    unsqueeze_199: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_200: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_372: "f32[768]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    mul_373: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_374: "f32[768]" = torch.ops.aten.mul.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_202: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_203: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_375: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_90);  primals_90 = None
    unsqueeze_205: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
    unsqueeze_206: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    sub_68: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_136, unsqueeze_198);  add_136 = unsqueeze_198 = None
    mul_376: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_204);  sub_68 = unsqueeze_204 = None
    sub_69: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_119, mul_376);  getitem_119 = mul_376 = None
    sub_70: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_201);  sub_69 = unsqueeze_201 = None
    mul_377: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_207);  sub_70 = unsqueeze_207 = None
    mul_378: "f32[768]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_61);  sum_28 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_202: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_199, mul_377);  add_199 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(add_202, view_39, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_39 = primals_89 = None
    getitem_122: "f32[8, 768, 7, 7]" = convolution_backward_14[0]
    getitem_123: "f32[768, 768, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_94: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(getitem_122, [8, 6, 128, 49]);  getitem_122 = None
    permute_50: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_94, [0, 1, 3, 2]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_95: "f32[48, 49, 128]" = torch.ops.aten.view.default(permute_50, [48, 49, 128]);  permute_50 = None
    permute_51: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_28: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_51, view_95);  permute_51 = None
    permute_52: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_29: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_95, permute_52);  view_95 = permute_52 = None
    view_96: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_28, [8, 6, 49, 128]);  bmm_28 = None
    view_97: "f32[8, 6, 49, 49]" = torch.ops.aten.view.default(bmm_29, [8, 6, 49, 49]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_12: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_379: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_97, alias_12);  view_97 = None
    sum_29: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [-1], True)
    mul_380: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_12, sum_29);  alias_12 = sum_29 = None
    sub_71: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_381: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_71, 0.08838834764831845);  sub_71 = None
    view_98: "f32[48, 49, 49]" = torch.ops.aten.view.default(mul_381, [48, 49, 49]);  mul_381 = None
    permute_53: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_30: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_53, view_98);  permute_53 = None
    permute_54: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_31: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_98, permute_54);  view_98 = permute_54 = None
    view_99: "f32[8, 6, 128, 49]" = torch.ops.aten.view.default(bmm_30, [8, 6, 128, 49]);  bmm_30 = None
    view_100: "f32[8, 6, 49, 128]" = torch.ops.aten.view.default(bmm_31, [8, 6, 49, 128]);  bmm_31 = None
    permute_55: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_99, [0, 1, 3, 2]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_3: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_100, permute_55, view_96]);  view_100 = permute_55 = view_96 = None
    view_101: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.view.default(cat_3, [3, 8, 6, 49, 128]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_56: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_101, [1, 0, 2, 4, 3]);  view_101 = None
    clone_85: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_102: "f32[8, 2304, 7, 7]" = torch.ops.aten.view.default(clone_85, [8, 2304, 7, 7]);  clone_85 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_102, add_135, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_102 = add_135 = primals_88 = None
    getitem_125: "f32[8, 768, 7, 7]" = convolution_backward_15[0]
    getitem_126: "f32[2304, 768, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_208: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_209: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_125, [0, 2, 3])
    sub_72: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(clone_48, unsqueeze_210)
    mul_382: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_125, sub_72);  sub_72 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
    mul_383: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_211: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
    unsqueeze_212: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_384: "f32[768]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_385: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_386: "f32[768]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    unsqueeze_214: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_215: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_387: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_86);  primals_86 = None
    unsqueeze_217: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_218: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    sub_73: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(clone_48, unsqueeze_210);  clone_48 = unsqueeze_210 = None
    mul_388: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_216);  sub_73 = unsqueeze_216 = None
    sub_74: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_125, mul_388);  getitem_125 = mul_388 = None
    sub_75: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_213);  sub_74 = unsqueeze_213 = None
    mul_389: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_219);  sub_75 = unsqueeze_219 = None
    mul_390: "f32[768]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_58);  sum_31 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_203: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_202, mul_389);  add_202 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:421, code: x = self.pos_drop(x + self.pos_embed3)
    sum_32: "f32[1, 768, 7, 7]" = torch.ops.aten.sum.dim_IntList(add_203, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    unsqueeze_220: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_221: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 2, 3])
    sub_76: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_222)
    mul_391: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_203, sub_76);  sub_76 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_392: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    unsqueeze_223: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_224: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_393: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, 0.002551020408163265)
    mul_394: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_395: "f32[768]" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_226: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_227: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_396: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_84);  primals_84 = None
    unsqueeze_229: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_230: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    sub_77: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_222);  convolution_40 = unsqueeze_222 = None
    mul_397: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_228);  sub_77 = unsqueeze_228 = None
    sub_78: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_203, mul_397);  add_203 = mul_397 = None
    sub_79: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_225);  sub_78 = unsqueeze_225 = None
    mul_398: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_231);  sub_79 = unsqueeze_231 = None
    mul_399: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_55);  sum_34 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_398, add_124, primals_82, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_398 = add_124 = primals_82 = None
    getitem_128: "f32[8, 384, 14, 14]" = convolution_backward_16[0]
    getitem_129: "f32[768, 384, 2, 2]" = convolution_backward_16[1]
    getitem_130: "f32[768]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_128, clone_46, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_46 = primals_81 = None
    getitem_131: "f32[8, 1536, 14, 14]" = convolution_backward_17[0]
    getitem_132: "f32[384, 1536, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_400: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_26: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_400);  mul_400 = None
    add_204: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_401: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_204, 0.5);  add_204 = None
    mul_402: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, convolution_38)
    mul_403: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_402, -0.5);  mul_402 = None
    exp_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_403);  mul_403 = None
    mul_404: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_405: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, mul_404);  convolution_38 = mul_404 = None
    add_205: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
    mul_406: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_131, add_205);  getitem_131 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_406, add_122, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = add_122 = primals_80 = None
    getitem_134: "f32[8, 384, 14, 14]" = convolution_backward_18[0]
    getitem_135: "f32[1536, 384, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_232: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_233: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_134, [0, 2, 3])
    sub_80: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_117, unsqueeze_234)
    mul_407: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_134, sub_80);  sub_80 = None
    sum_36: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_408: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    unsqueeze_235: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_236: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_409: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    mul_410: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_411: "f32[384]" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    unsqueeze_238: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_239: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_412: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_78);  primals_78 = None
    unsqueeze_241: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_242: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_81: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_117, unsqueeze_234);  add_117 = unsqueeze_234 = None
    mul_413: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_240);  sub_81 = unsqueeze_240 = None
    sub_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_134, mul_413);  getitem_134 = mul_413 = None
    sub_83: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_237);  sub_82 = unsqueeze_237 = None
    mul_414: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_243);  sub_83 = unsqueeze_243 = None
    mul_415: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_52);  sum_36 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_206: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_128, mul_414);  getitem_128 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_206, view_31, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_31 = primals_77 = None
    getitem_137: "f32[8, 384, 14, 14]" = convolution_backward_19[0]
    getitem_138: "f32[384, 384, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_103: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(getitem_137, [8, 6, 64, 196]);  getitem_137 = None
    permute_57: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_104: "f32[48, 196, 64]" = torch.ops.aten.view.default(permute_57, [48, 196, 64]);  permute_57 = None
    permute_58: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_32: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_58, view_104);  permute_58 = None
    permute_59: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    bmm_33: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_104, permute_59);  view_104 = permute_59 = None
    view_105: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_32, [8, 6, 196, 64]);  bmm_32 = None
    view_106: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_33, [8, 6, 196, 196]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_13: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_416: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_106, alias_13);  view_106 = None
    sum_37: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [-1], True)
    mul_417: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_13, sum_37);  alias_13 = sum_37 = None
    sub_84: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_418: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_84, 0.125);  sub_84 = None
    view_107: "f32[48, 196, 196]" = torch.ops.aten.view.default(mul_418, [48, 196, 196]);  mul_418 = None
    permute_60: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_34: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_60, view_107);  permute_60 = None
    permute_61: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_35: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_107, permute_61);  view_107 = permute_61 = None
    view_108: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(bmm_34, [8, 6, 64, 196]);  bmm_34 = None
    view_109: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_35, [8, 6, 196, 64]);  bmm_35 = None
    permute_62: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_108, [0, 1, 3, 2]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_4: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_109, permute_62, view_105]);  view_109 = permute_62 = view_105 = None
    view_110: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.view.default(cat_4, [3, 8, 6, 196, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_63: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_110, [1, 0, 2, 4, 3]);  view_110 = None
    clone_86: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_111: "f32[8, 1152, 14, 14]" = torch.ops.aten.view.default(clone_86, [8, 1152, 14, 14]);  clone_86 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_111, add_116, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_111 = add_116 = primals_76 = None
    getitem_140: "f32[8, 384, 14, 14]" = convolution_backward_20[0]
    getitem_141: "f32[1152, 384, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_244: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_245: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_38: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_140, [0, 2, 3])
    sub_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_111, unsqueeze_246)
    mul_419: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_140, sub_85);  sub_85 = None
    sum_39: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3]);  mul_419 = None
    mul_420: "f32[384]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_247: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
    unsqueeze_248: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_421: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_422: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_423: "f32[384]" = torch.ops.aten.mul.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    unsqueeze_250: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_251: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_424: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_74);  primals_74 = None
    unsqueeze_253: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
    unsqueeze_254: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_86: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_111, unsqueeze_246);  add_111 = unsqueeze_246 = None
    mul_425: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_252);  sub_86 = unsqueeze_252 = None
    sub_87: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_140, mul_425);  getitem_140 = mul_425 = None
    sub_88: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_249);  sub_87 = unsqueeze_249 = None
    mul_426: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_255);  sub_88 = unsqueeze_255 = None
    mul_427: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_49);  sum_39 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_207: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_206, mul_426);  add_206 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(add_207, clone_38, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_38 = primals_73 = None
    getitem_143: "f32[8, 1536, 14, 14]" = convolution_backward_21[0]
    getitem_144: "f32[384, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_428: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476)
    erf_27: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_428);  mul_428 = None
    add_208: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_429: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_430: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, convolution_34)
    mul_431: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_430, -0.5);  mul_430 = None
    exp_13: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_431);  mul_431 = None
    mul_432: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_433: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, mul_432);  convolution_34 = mul_432 = None
    add_209: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_429, mul_433);  mul_429 = mul_433 = None
    mul_434: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_143, add_209);  getitem_143 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_434, add_109, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = add_109 = primals_72 = None
    getitem_146: "f32[8, 384, 14, 14]" = convolution_backward_22[0]
    getitem_147: "f32[1536, 384, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_256: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_257: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_40: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_146, [0, 2, 3])
    sub_89: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_104, unsqueeze_258)
    mul_435: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_146, sub_89);  sub_89 = None
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3]);  mul_435 = None
    mul_436: "f32[384]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_259: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_260: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_437: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_438: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_439: "f32[384]" = torch.ops.aten.mul.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    unsqueeze_262: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
    unsqueeze_263: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_440: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_70);  primals_70 = None
    unsqueeze_265: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_266: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_90: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_104, unsqueeze_258);  add_104 = unsqueeze_258 = None
    mul_441: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_264);  sub_90 = unsqueeze_264 = None
    sub_91: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_146, mul_441);  getitem_146 = mul_441 = None
    sub_92: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_261);  sub_91 = unsqueeze_261 = None
    mul_442: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_267);  sub_92 = unsqueeze_267 = None
    mul_443: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_46);  sum_41 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_210: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_207, mul_442);  add_207 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(add_210, view_23, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_23 = primals_69 = None
    getitem_149: "f32[8, 384, 14, 14]" = convolution_backward_23[0]
    getitem_150: "f32[384, 384, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_112: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(getitem_149, [8, 6, 64, 196]);  getitem_149 = None
    permute_64: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 2]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_113: "f32[48, 196, 64]" = torch.ops.aten.view.default(permute_64, [48, 196, 64]);  permute_64 = None
    permute_65: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    bmm_36: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_65, view_113);  permute_65 = None
    permute_66: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    bmm_37: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_113, permute_66);  view_113 = permute_66 = None
    view_114: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_36, [8, 6, 196, 64]);  bmm_36 = None
    view_115: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_37, [8, 6, 196, 196]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_14: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_444: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_115, alias_14);  view_115 = None
    sum_42: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [-1], True)
    mul_445: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_14, sum_42);  alias_14 = sum_42 = None
    sub_93: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_446: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_93, 0.125);  sub_93 = None
    view_116: "f32[48, 196, 196]" = torch.ops.aten.view.default(mul_446, [48, 196, 196]);  mul_446 = None
    permute_67: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    bmm_38: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_67, view_116);  permute_67 = None
    permute_68: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    bmm_39: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_116, permute_68);  view_116 = permute_68 = None
    view_117: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(bmm_38, [8, 6, 64, 196]);  bmm_38 = None
    view_118: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_39, [8, 6, 196, 64]);  bmm_39 = None
    permute_69: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_5: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_118, permute_69, view_114]);  view_118 = permute_69 = view_114 = None
    view_119: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.view.default(cat_5, [3, 8, 6, 196, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_70: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_119, [1, 0, 2, 4, 3]);  view_119 = None
    clone_87: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    view_120: "f32[8, 1152, 14, 14]" = torch.ops.aten.view.default(clone_87, [8, 1152, 14, 14]);  clone_87 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_120, add_103, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_120 = add_103 = primals_68 = None
    getitem_152: "f32[8, 384, 14, 14]" = convolution_backward_24[0]
    getitem_153: "f32[1152, 384, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_268: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_269: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_152, [0, 2, 3])
    sub_94: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_98, unsqueeze_270)
    mul_447: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_152, sub_94);  sub_94 = None
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
    mul_448: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    unsqueeze_271: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_272: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_449: "f32[384]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    mul_450: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_451: "f32[384]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_274: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_275: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_452: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_66);  primals_66 = None
    unsqueeze_277: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_278: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_95: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_98, unsqueeze_270);  add_98 = unsqueeze_270 = None
    mul_453: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_276);  sub_95 = unsqueeze_276 = None
    sub_96: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_152, mul_453);  getitem_152 = mul_453 = None
    sub_97: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_273);  sub_96 = unsqueeze_273 = None
    mul_454: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_279);  sub_97 = unsqueeze_279 = None
    mul_455: "f32[384]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_43);  sum_44 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_211: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_210, mul_454);  add_210 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(add_211, clone_30, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_30 = primals_65 = None
    getitem_155: "f32[8, 1536, 14, 14]" = convolution_backward_25[0]
    getitem_156: "f32[384, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_456: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476)
    erf_28: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_456);  mul_456 = None
    add_212: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_457: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_212, 0.5);  add_212 = None
    mul_458: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, convolution_30)
    mul_459: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_458, -0.5);  mul_458 = None
    exp_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_459);  mul_459 = None
    mul_460: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_461: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, mul_460);  convolution_30 = mul_460 = None
    add_213: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_457, mul_461);  mul_457 = mul_461 = None
    mul_462: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_155, add_213);  getitem_155 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_462, add_96, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_462 = add_96 = primals_64 = None
    getitem_158: "f32[8, 384, 14, 14]" = convolution_backward_26[0]
    getitem_159: "f32[1536, 384, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_280: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_281: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_158, [0, 2, 3])
    sub_98: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_91, unsqueeze_282)
    mul_463: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_158, sub_98);  sub_98 = None
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_464: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    unsqueeze_283: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_284: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_465: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    mul_466: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_467: "f32[384]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_286: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_287: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_468: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_62);  primals_62 = None
    unsqueeze_289: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_290: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_99: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_91, unsqueeze_282);  add_91 = unsqueeze_282 = None
    mul_469: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_288);  sub_99 = unsqueeze_288 = None
    sub_100: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_158, mul_469);  getitem_158 = mul_469 = None
    sub_101: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_285);  sub_100 = unsqueeze_285 = None
    mul_470: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_291);  sub_101 = unsqueeze_291 = None
    mul_471: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_40);  sum_46 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_214: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_211, mul_470);  add_211 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(add_214, view_15, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_15 = primals_61 = None
    getitem_161: "f32[8, 384, 14, 14]" = convolution_backward_27[0]
    getitem_162: "f32[384, 384, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_121: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(getitem_161, [8, 6, 64, 196]);  getitem_161 = None
    permute_71: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_121, [0, 1, 3, 2]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_122: "f32[48, 196, 64]" = torch.ops.aten.view.default(permute_71, [48, 196, 64]);  permute_71 = None
    permute_72: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_40: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_72, view_122);  permute_72 = None
    permute_73: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_41: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_122, permute_73);  view_122 = permute_73 = None
    view_123: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_40, [8, 6, 196, 64]);  bmm_40 = None
    view_124: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_41, [8, 6, 196, 196]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_15: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_472: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_124, alias_15);  view_124 = None
    sum_47: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
    mul_473: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_15, sum_47);  alias_15 = sum_47 = None
    sub_102: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_474: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_102, 0.125);  sub_102 = None
    view_125: "f32[48, 196, 196]" = torch.ops.aten.view.default(mul_474, [48, 196, 196]);  mul_474 = None
    permute_74: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_42: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_74, view_125);  permute_74 = None
    permute_75: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_43: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_125, permute_75);  view_125 = permute_75 = None
    view_126: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(bmm_42, [8, 6, 64, 196]);  bmm_42 = None
    view_127: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_43, [8, 6, 196, 64]);  bmm_43 = None
    permute_76: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_126, [0, 1, 3, 2]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_6: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_127, permute_76, view_123]);  view_127 = permute_76 = view_123 = None
    view_128: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.view.default(cat_6, [3, 8, 6, 196, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_77: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_128, [1, 0, 2, 4, 3]);  view_128 = None
    clone_88: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_129: "f32[8, 1152, 14, 14]" = torch.ops.aten.view.default(clone_88, [8, 1152, 14, 14]);  clone_88 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_129, add_90, primals_60, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_129 = add_90 = primals_60 = None
    getitem_164: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_165: "f32[1152, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_292: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_293: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_164, [0, 2, 3])
    sub_103: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_85, unsqueeze_294)
    mul_475: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_164, sub_103);  sub_103 = None
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 2, 3]);  mul_475 = None
    mul_476: "f32[384]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_295: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_296: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_477: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_478: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_479: "f32[384]" = torch.ops.aten.mul.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    unsqueeze_298: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
    unsqueeze_299: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_480: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_58);  primals_58 = None
    unsqueeze_301: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_302: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_104: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_85, unsqueeze_294);  add_85 = unsqueeze_294 = None
    mul_481: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_300);  sub_104 = unsqueeze_300 = None
    sub_105: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_164, mul_481);  getitem_164 = mul_481 = None
    sub_106: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_297);  sub_105 = unsqueeze_297 = None
    mul_482: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_303);  sub_106 = unsqueeze_303 = None
    mul_483: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_37);  sum_49 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_215: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_214, mul_482);  add_214 = mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(add_215, clone_22, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_22 = primals_57 = None
    getitem_167: "f32[8, 1536, 14, 14]" = convolution_backward_29[0]
    getitem_168: "f32[384, 1536, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_484: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_29: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_484);  mul_484 = None
    add_216: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_485: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, 0.5);  add_216 = None
    mul_486: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, convolution_26)
    mul_487: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
    exp_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
    mul_488: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_489: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, mul_488);  convolution_26 = mul_488 = None
    add_217: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
    mul_490: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, add_217);  getitem_167 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_490, add_83, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_490 = add_83 = primals_56 = None
    getitem_170: "f32[8, 384, 14, 14]" = convolution_backward_30[0]
    getitem_171: "f32[1536, 384, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_304: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_305: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_50: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_170, [0, 2, 3])
    sub_107: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_78, unsqueeze_306)
    mul_491: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_170, sub_107);  sub_107 = None
    sum_51: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[384]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_307: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_308: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_493: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_310: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_311: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_496: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_54);  primals_54 = None
    unsqueeze_313: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_314: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_108: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_78, unsqueeze_306);  add_78 = unsqueeze_306 = None
    mul_497: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_312);  sub_108 = unsqueeze_312 = None
    sub_109: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_170, mul_497);  getitem_170 = mul_497 = None
    sub_110: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_309);  sub_109 = unsqueeze_309 = None
    mul_498: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_315);  sub_110 = unsqueeze_315 = None
    mul_499: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_34);  sum_51 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_218: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_215, mul_498);  add_215 = mul_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(add_218, view_7, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_7 = primals_53 = None
    getitem_173: "f32[8, 384, 14, 14]" = convolution_backward_31[0]
    getitem_174: "f32[384, 384, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_130: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(getitem_173, [8, 6, 64, 196]);  getitem_173 = None
    permute_78: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_130, [0, 1, 3, 2]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_131: "f32[48, 196, 64]" = torch.ops.aten.view.default(permute_78, [48, 196, 64]);  permute_78 = None
    permute_79: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
    bmm_44: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_79, view_131);  permute_79 = None
    permute_80: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    bmm_45: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_131, permute_80);  view_131 = permute_80 = None
    view_132: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_44, [8, 6, 196, 64]);  bmm_44 = None
    view_133: "f32[8, 6, 196, 196]" = torch.ops.aten.view.default(bmm_45, [8, 6, 196, 196]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_16: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_500: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_133, alias_16);  view_133 = None
    sum_52: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [-1], True)
    mul_501: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_16, sum_52);  alias_16 = sum_52 = None
    sub_111: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_502: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_111, 0.125);  sub_111 = None
    view_134: "f32[48, 196, 196]" = torch.ops.aten.view.default(mul_502, [48, 196, 196]);  mul_502 = None
    permute_81: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
    bmm_46: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_81, view_134);  permute_81 = None
    permute_82: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    bmm_47: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_134, permute_82);  view_134 = permute_82 = None
    view_135: "f32[8, 6, 64, 196]" = torch.ops.aten.view.default(bmm_46, [8, 6, 64, 196]);  bmm_46 = None
    view_136: "f32[8, 6, 196, 64]" = torch.ops.aten.view.default(bmm_47, [8, 6, 196, 64]);  bmm_47 = None
    permute_83: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_135, [0, 1, 3, 2]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_7: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_136, permute_83, view_132]);  view_136 = permute_83 = view_132 = None
    view_137: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.view.default(cat_7, [3, 8, 6, 196, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_84: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_137, [1, 0, 2, 4, 3]);  view_137 = None
    clone_89: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_138: "f32[8, 1152, 14, 14]" = torch.ops.aten.view.default(clone_89, [8, 1152, 14, 14]);  clone_89 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_138, add_77, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_138 = add_77 = primals_52 = None
    getitem_176: "f32[8, 384, 14, 14]" = convolution_backward_32[0]
    getitem_177: "f32[1152, 384, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_316: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_317: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_53: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_176, [0, 2, 3])
    sub_112: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(clone_15, unsqueeze_318)
    mul_503: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, sub_112);  sub_112 = None
    sum_54: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_319: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_320: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_505: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_506: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_322: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_323: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_508: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_50);  primals_50 = None
    unsqueeze_325: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_326: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_113: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(clone_15, unsqueeze_318);  clone_15 = unsqueeze_318 = None
    mul_509: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_324);  sub_113 = unsqueeze_324 = None
    sub_114: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_176, mul_509);  getitem_176 = mul_509 = None
    sub_115: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_321);  sub_114 = unsqueeze_321 = None
    mul_510: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_327);  sub_115 = unsqueeze_327 = None
    mul_511: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_31);  sum_54 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_219: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_218, mul_510);  add_218 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:411, code: x = self.pos_drop(x + self.pos_embed2)
    sum_55: "f32[1, 384, 14, 14]" = torch.ops.aten.sum.dim_IntList(add_219, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    unsqueeze_328: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_329: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 2, 3])
    sub_116: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_330)
    mul_512: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_219, sub_116);  sub_116 = None
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3]);  mul_512 = None
    mul_513: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_331: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_332: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_514: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_515: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_516: "f32[384]" = torch.ops.aten.mul.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    unsqueeze_334: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_335: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_517: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_48);  primals_48 = None
    unsqueeze_337: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_338: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_117: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_330);  convolution_23 = unsqueeze_330 = None
    mul_518: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_336);  sub_117 = unsqueeze_336 = None
    sub_118: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_219, mul_518);  add_219 = mul_518 = None
    sub_119: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_333);  sub_118 = unsqueeze_333 = None
    mul_519: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_339);  sub_119 = unsqueeze_339 = None
    mul_520: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_28);  sum_57 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_519, add_66, primals_46, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_519 = add_66 = primals_46 = None
    getitem_179: "f32[8, 192, 28, 28]" = convolution_backward_33[0]
    getitem_180: "f32[384, 192, 2, 2]" = convolution_backward_33[1]
    getitem_181: "f32[384]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(getitem_179, mul_104, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_104 = primals_45 = None
    getitem_182: "f32[8, 384, 28, 28]" = convolution_backward_34[0]
    getitem_183: "f32[192, 384, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_521: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.7071067811865476)
    erf_30: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_521);  mul_521 = None
    add_220: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_522: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_523: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, convolution_21)
    mul_524: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_523, -0.5);  mul_523 = None
    exp_16: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_524);  mul_524 = None
    mul_525: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_526: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, mul_525);  convolution_21 = mul_525 = None
    add_221: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_522, mul_526);  mul_522 = mul_526 = None
    mul_527: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_182, add_221);  getitem_182 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_527, clone_13, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_527 = clone_13 = primals_44 = None
    getitem_185: "f32[8, 384, 28, 28]" = convolution_backward_35[0]
    getitem_186: "f32[384, 48, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_528: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_31: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_528);  mul_528 = None
    add_222: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_529: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_222, 0.5);  add_222 = None
    mul_530: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, convolution_20)
    mul_531: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_530, -0.5);  mul_530 = None
    exp_17: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_531);  mul_531 = None
    mul_532: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_533: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, mul_532);  convolution_20 = mul_532 = None
    add_223: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_529, mul_533);  mul_529 = mul_533 = None
    mul_534: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_185, add_223);  getitem_185 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_534, add_63, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_534 = add_63 = primals_43 = None
    getitem_188: "f32[8, 192, 28, 28]" = convolution_backward_36[0]
    getitem_189: "f32[384, 192, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_340: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_341: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_58: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_188, [0, 2, 3])
    sub_120: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_58, unsqueeze_342)
    mul_535: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_188, sub_120);  sub_120 = None
    sum_59: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[192]" = torch.ops.aten.mul.Tensor(sum_58, 0.00015943877551020407)
    unsqueeze_343: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_344: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_537: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, 0.00015943877551020407)
    mul_538: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_539: "f32[192]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_346: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_347: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_540: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_41);  primals_41 = None
    unsqueeze_349: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_350: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_121: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_58, unsqueeze_342);  add_58 = unsqueeze_342 = None
    mul_541: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_348);  sub_121 = unsqueeze_348 = None
    sub_122: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_188, mul_541);  getitem_188 = mul_541 = None
    sub_123: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_345);  sub_122 = unsqueeze_345 = None
    mul_542: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_351);  sub_123 = unsqueeze_351 = None
    mul_543: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_25);  sum_59 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_224: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_179, mul_542);  getitem_179 = mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(add_224, mul_91, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_91 = primals_40 = None
    getitem_191: "f32[8, 384, 28, 28]" = convolution_backward_37[0]
    getitem_192: "f32[192, 384, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_544: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_32: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_544);  mul_544 = None
    add_225: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_545: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_225, 0.5);  add_225 = None
    mul_546: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, convolution_18)
    mul_547: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
    exp_18: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
    mul_548: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_549: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, mul_548);  convolution_18 = mul_548 = None
    add_226: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
    mul_550: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_191, add_226);  getitem_191 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_550, clone_11, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_550 = clone_11 = primals_39 = None
    getitem_194: "f32[8, 384, 28, 28]" = convolution_backward_38[0]
    getitem_195: "f32[384, 48, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_551: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, 0.7071067811865476)
    erf_33: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_551);  mul_551 = None
    add_227: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_552: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_227, 0.5);  add_227 = None
    mul_553: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, convolution_17)
    mul_554: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_553, -0.5);  mul_553 = None
    exp_19: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_554);  mul_554 = None
    mul_555: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_556: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, mul_555);  convolution_17 = mul_555 = None
    add_228: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_552, mul_556);  mul_552 = mul_556 = None
    mul_557: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_194, add_228);  getitem_194 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_557, add_55, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = add_55 = primals_38 = None
    getitem_197: "f32[8, 192, 28, 28]" = convolution_backward_39[0]
    getitem_198: "f32[384, 192, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_352: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_353: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_60: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_197, [0, 2, 3])
    sub_124: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_50, unsqueeze_354)
    mul_558: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_197, sub_124);  sub_124 = None
    sum_61: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3]);  mul_558 = None
    mul_559: "f32[192]" = torch.ops.aten.mul.Tensor(sum_60, 0.00015943877551020407)
    unsqueeze_355: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
    unsqueeze_356: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_560: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, 0.00015943877551020407)
    mul_561: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_562: "f32[192]" = torch.ops.aten.mul.Tensor(mul_560, mul_561);  mul_560 = mul_561 = None
    unsqueeze_358: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_359: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_563: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_36);  primals_36 = None
    unsqueeze_361: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_362: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_125: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_50, unsqueeze_354);  add_50 = unsqueeze_354 = None
    mul_564: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_360);  sub_125 = unsqueeze_360 = None
    sub_126: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_197, mul_564);  getitem_197 = mul_564 = None
    sub_127: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_357);  sub_126 = unsqueeze_357 = None
    mul_565: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_363);  sub_127 = unsqueeze_363 = None
    mul_566: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_22);  sum_61 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_229: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_224, mul_565);  add_224 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_229, mul_78, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_78 = primals_35 = None
    getitem_200: "f32[8, 384, 28, 28]" = convolution_backward_40[0]
    getitem_201: "f32[192, 384, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_567: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.7071067811865476)
    erf_34: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_567);  mul_567 = None
    add_230: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_568: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_230, 0.5);  add_230 = None
    mul_569: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, convolution_15)
    mul_570: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_569, -0.5);  mul_569 = None
    exp_20: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_570);  mul_570 = None
    mul_571: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_572: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, mul_571);  convolution_15 = mul_571 = None
    add_231: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_568, mul_572);  mul_568 = mul_572 = None
    mul_573: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_200, add_231);  getitem_200 = add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_573, clone_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_573 = clone_9 = primals_34 = None
    getitem_203: "f32[8, 384, 28, 28]" = convolution_backward_41[0]
    getitem_204: "f32[384, 48, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_574: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_35: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_574);  mul_574 = None
    add_232: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_575: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_232, 0.5);  add_232 = None
    mul_576: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, convolution_14)
    mul_577: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_576, -0.5);  mul_576 = None
    exp_21: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_577);  mul_577 = None
    mul_578: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_579: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, mul_578);  convolution_14 = mul_578 = None
    add_233: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_575, mul_579);  mul_575 = mul_579 = None
    mul_580: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_203, add_233);  getitem_203 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_580, add_47, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_580 = add_47 = primals_33 = None
    getitem_206: "f32[8, 192, 28, 28]" = convolution_backward_42[0]
    getitem_207: "f32[384, 192, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_364: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_365: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_206, [0, 2, 3])
    sub_128: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_42, unsqueeze_366)
    mul_581: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_206, sub_128);  sub_128 = None
    sum_63: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
    mul_582: "f32[192]" = torch.ops.aten.mul.Tensor(sum_62, 0.00015943877551020407)
    unsqueeze_367: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_368: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_583: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
    mul_584: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_585: "f32[192]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    unsqueeze_370: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_371: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_586: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_31);  primals_31 = None
    unsqueeze_373: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_374: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_129: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_42, unsqueeze_366);  add_42 = unsqueeze_366 = None
    mul_587: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_372);  sub_129 = unsqueeze_372 = None
    sub_130: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_206, mul_587);  getitem_206 = mul_587 = None
    sub_131: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_369);  sub_130 = unsqueeze_369 = None
    mul_588: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_375);  sub_131 = unsqueeze_375 = None
    mul_589: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_19);  sum_63 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_234: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_229, mul_588);  add_229 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(add_234, mul_65, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_65 = primals_30 = None
    getitem_209: "f32[8, 384, 28, 28]" = convolution_backward_43[0]
    getitem_210: "f32[192, 384, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_590: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_36: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_590);  mul_590 = None
    add_235: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_591: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_235, 0.5);  add_235 = None
    mul_592: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, convolution_12)
    mul_593: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_592, -0.5);  mul_592 = None
    exp_22: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_593);  mul_593 = None
    mul_594: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_595: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, mul_594);  convolution_12 = mul_594 = None
    add_236: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_591, mul_595);  mul_591 = mul_595 = None
    mul_596: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_209, add_236);  getitem_209 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_596, clone_7, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_596 = clone_7 = primals_29 = None
    getitem_212: "f32[8, 384, 28, 28]" = convolution_backward_44[0]
    getitem_213: "f32[384, 48, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_597: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476)
    erf_37: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_597);  mul_597 = None
    add_237: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_598: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_237, 0.5);  add_237 = None
    mul_599: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, convolution_11)
    mul_600: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_599, -0.5);  mul_599 = None
    exp_23: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_600);  mul_600 = None
    mul_601: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_602: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, mul_601);  convolution_11 = mul_601 = None
    add_238: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_598, mul_602);  mul_598 = mul_602 = None
    mul_603: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_212, add_238);  getitem_212 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_603, add_39, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = add_39 = primals_28 = None
    getitem_215: "f32[8, 192, 28, 28]" = convolution_backward_45[0]
    getitem_216: "f32[384, 192, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_376: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_377: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_64: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_215, [0, 2, 3])
    sub_132: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_34, unsqueeze_378)
    mul_604: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_215, sub_132);  sub_132 = None
    sum_65: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 2, 3]);  mul_604 = None
    mul_605: "f32[192]" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
    unsqueeze_379: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_380: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_606: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
    mul_607: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_608: "f32[192]" = torch.ops.aten.mul.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_382: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_383: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_609: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_26);  primals_26 = None
    unsqueeze_385: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_386: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_133: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_34, unsqueeze_378);  add_34 = unsqueeze_378 = None
    mul_610: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_384);  sub_133 = unsqueeze_384 = None
    sub_134: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_215, mul_610);  getitem_215 = mul_610 = None
    sub_135: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_381);  sub_134 = unsqueeze_381 = None
    mul_611: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_387);  sub_135 = unsqueeze_387 = None
    mul_612: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_16);  sum_65 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_239: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_234, mul_611);  add_234 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_239, mul_52, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_52 = primals_25 = None
    getitem_218: "f32[8, 384, 28, 28]" = convolution_backward_46[0]
    getitem_219: "f32[192, 384, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_613: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_38: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_613);  mul_613 = None
    add_240: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_614: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_240, 0.5);  add_240 = None
    mul_615: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, convolution_9)
    mul_616: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_615, -0.5);  mul_615 = None
    exp_24: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_616);  mul_616 = None
    mul_617: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_618: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, mul_617);  convolution_9 = mul_617 = None
    add_241: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_614, mul_618);  mul_614 = mul_618 = None
    mul_619: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_218, add_241);  getitem_218 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_619, clone_5, primals_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_619 = clone_5 = primals_24 = None
    getitem_221: "f32[8, 384, 28, 28]" = convolution_backward_47[0]
    getitem_222: "f32[384, 48, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_620: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476)
    erf_39: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_620);  mul_620 = None
    add_242: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_621: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_242, 0.5);  add_242 = None
    mul_622: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, convolution_8)
    mul_623: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_622, -0.5);  mul_622 = None
    exp_25: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_623);  mul_623 = None
    mul_624: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_625: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, mul_624);  convolution_8 = mul_624 = None
    add_243: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_621, mul_625);  mul_621 = mul_625 = None
    mul_626: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_221, add_243);  getitem_221 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_626, add_31, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = add_31 = primals_23 = None
    getitem_224: "f32[8, 192, 28, 28]" = convolution_backward_48[0]
    getitem_225: "f32[384, 192, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_388: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_389: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_224, [0, 2, 3])
    sub_136: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_26, unsqueeze_390)
    mul_627: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_224, sub_136);  sub_136 = None
    sum_67: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
    mul_628: "f32[192]" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
    unsqueeze_391: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_392: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_629: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
    mul_630: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_631: "f32[192]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_394: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_395: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_632: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_21);  primals_21 = None
    unsqueeze_397: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_398: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_137: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_26, unsqueeze_390);  add_26 = unsqueeze_390 = None
    mul_633: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_396);  sub_137 = unsqueeze_396 = None
    sub_138: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_224, mul_633);  getitem_224 = mul_633 = None
    sub_139: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_393);  sub_138 = unsqueeze_393 = None
    mul_634: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_399);  sub_139 = unsqueeze_399 = None
    mul_635: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_13);  sum_67 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_244: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_239, mul_634);  add_239 = mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(add_244, mul_39, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_39 = primals_20 = None
    getitem_227: "f32[8, 384, 28, 28]" = convolution_backward_49[0]
    getitem_228: "f32[192, 384, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_636: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_40: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_636);  mul_636 = None
    add_245: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_637: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
    mul_638: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, convolution_6)
    mul_639: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_638, -0.5);  mul_638 = None
    exp_26: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_639);  mul_639 = None
    mul_640: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_641: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, mul_640);  convolution_6 = mul_640 = None
    add_246: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_637, mul_641);  mul_637 = mul_641 = None
    mul_642: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_227, add_246);  getitem_227 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_642, clone_3, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_642 = clone_3 = primals_19 = None
    getitem_230: "f32[8, 384, 28, 28]" = convolution_backward_50[0]
    getitem_231: "f32[384, 48, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_643: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_41: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_643);  mul_643 = None
    add_247: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_644: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_247, 0.5);  add_247 = None
    mul_645: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, convolution_5)
    mul_646: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_645, -0.5);  mul_645 = None
    exp_27: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_646);  mul_646 = None
    mul_647: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_648: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, mul_647);  convolution_5 = mul_647 = None
    add_248: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_644, mul_648);  mul_644 = mul_648 = None
    mul_649: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_230, add_248);  getitem_230 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_649, add_23, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_649 = add_23 = primals_18 = None
    getitem_233: "f32[8, 192, 28, 28]" = convolution_backward_51[0]
    getitem_234: "f32[384, 192, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_400: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_401: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_233, [0, 2, 3])
    sub_140: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_18, unsqueeze_402)
    mul_650: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_233, sub_140);  sub_140 = None
    sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3]);  mul_650 = None
    mul_651: "f32[192]" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
    unsqueeze_403: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_404: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_652: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, 0.00015943877551020407)
    mul_653: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_654: "f32[192]" = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    unsqueeze_406: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_407: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_655: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_16);  primals_16 = None
    unsqueeze_409: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_410: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_141: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_18, unsqueeze_402);  add_18 = unsqueeze_402 = None
    mul_656: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_408);  sub_141 = unsqueeze_408 = None
    sub_142: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_233, mul_656);  getitem_233 = mul_656 = None
    sub_143: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_405);  sub_142 = unsqueeze_405 = None
    mul_657: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_411);  sub_143 = unsqueeze_411 = None
    mul_658: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_10);  sum_69 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_249: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_244, mul_657);  add_244 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_249, mul_26, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_26 = primals_15 = None
    getitem_236: "f32[8, 384, 28, 28]" = convolution_backward_52[0]
    getitem_237: "f32[192, 384, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_659: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_42: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_659);  mul_659 = None
    add_250: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_660: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_250, 0.5);  add_250 = None
    mul_661: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, convolution_3)
    mul_662: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_661, -0.5);  mul_661 = None
    exp_28: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_662);  mul_662 = None
    mul_663: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_664: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, mul_663);  convolution_3 = mul_663 = None
    add_251: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_660, mul_664);  mul_660 = mul_664 = None
    mul_665: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_236, add_251);  getitem_236 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_665, clone_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_665 = clone_1 = primals_14 = None
    getitem_239: "f32[8, 384, 28, 28]" = convolution_backward_53[0]
    getitem_240: "f32[384, 48, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_666: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
    erf_43: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_666);  mul_666 = None
    add_252: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_667: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_252, 0.5);  add_252 = None
    mul_668: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
    mul_669: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_668, -0.5);  mul_668 = None
    exp_29: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_669);  mul_669 = None
    mul_670: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_671: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, mul_670);  convolution_2 = mul_670 = None
    add_253: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_667, mul_671);  mul_667 = mul_671 = None
    mul_672: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, add_253);  getitem_239 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_672, add_15, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_672 = add_15 = primals_13 = None
    getitem_242: "f32[8, 192, 28, 28]" = convolution_backward_54[0]
    getitem_243: "f32[384, 192, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_412: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_413: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_70: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_242, [0, 2, 3])
    sub_144: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_414)
    mul_673: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_242, sub_144);  sub_144 = None
    sum_71: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_673, [0, 2, 3]);  mul_673 = None
    mul_674: "f32[192]" = torch.ops.aten.mul.Tensor(sum_70, 0.00015943877551020407)
    unsqueeze_415: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_416: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_675: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, 0.00015943877551020407)
    mul_676: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_677: "f32[192]" = torch.ops.aten.mul.Tensor(mul_675, mul_676);  mul_675 = mul_676 = None
    unsqueeze_418: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_419: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_678: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_11);  primals_11 = None
    unsqueeze_421: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
    unsqueeze_422: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_145: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_414);  clone = unsqueeze_414 = None
    mul_679: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_420);  sub_145 = unsqueeze_420 = None
    sub_146: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_242, mul_679);  getitem_242 = mul_679 = None
    sub_147: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_417);  sub_146 = unsqueeze_417 = None
    mul_680: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_423);  sub_147 = unsqueeze_423 = None
    mul_681: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_7);  sum_71 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_254: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_249, mul_680);  add_249 = mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:401, code: x = self.pos_drop(x + self.pos_embed1)
    sum_72: "f32[1, 192, 28, 28]" = torch.ops.aten.sum.dim_IntList(add_254, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    unsqueeze_424: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_425: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 2, 3])
    sub_148: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_426)
    mul_682: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_254, sub_148);  sub_148 = None
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 2, 3]);  mul_682 = None
    mul_683: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
    unsqueeze_427: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_428: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_684: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
    mul_685: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_686: "f32[192]" = torch.ops.aten.mul.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
    unsqueeze_430: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_431: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_687: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_9);  primals_9 = None
    unsqueeze_433: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
    unsqueeze_434: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_149: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_426);  convolution_1 = unsqueeze_426 = None
    mul_688: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_432);  sub_149 = unsqueeze_432 = None
    sub_150: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_254, mul_688);  add_254 = mul_688 = None
    sub_151: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_429);  sub_150 = unsqueeze_429 = None
    mul_689: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_435);  sub_151 = unsqueeze_435 = None
    mul_690: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_4);  sum_74 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_689, relu, primals_7, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_689 = primals_7 = None
    getitem_245: "f32[8, 32, 112, 112]" = convolution_backward_55[0]
    getitem_246: "f32[192, 32, 4, 4]" = convolution_backward_55[1]
    getitem_247: "f32[192]" = convolution_backward_55[2];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:396, code: x = self.stem(x)
    alias_18: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_19: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    le: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_19, 0);  alias_19 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le, scalar_tensor, getitem_245);  le = scalar_tensor = getitem_245 = None
    unsqueeze_436: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_437: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_75: "f32[32]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_152: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_438)
    mul_691: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where, sub_152);  sub_152 = None
    sum_76: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
    mul_692: "f32[32]" = torch.ops.aten.mul.Tensor(sum_75, 9.964923469387754e-06)
    unsqueeze_439: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_440: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_693: "f32[32]" = torch.ops.aten.mul.Tensor(sum_76, 9.964923469387754e-06)
    mul_694: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_695: "f32[32]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_442: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_443: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_696: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_5);  primals_5 = None
    unsqueeze_445: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_446: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_153: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_438);  convolution = unsqueeze_438 = None
    mul_697: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_444);  sub_153 = unsqueeze_444 = None
    sub_154: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where, mul_697);  where = mul_697 = None
    sub_155: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_441);  sub_154 = unsqueeze_441 = None
    mul_698: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_447);  sub_155 = unsqueeze_447 = None
    mul_699: "f32[32]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_1);  sum_76 = squeeze_1 = None
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_698, primals_206, primals_4, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_698 = primals_206 = primals_4 = None
    getitem_249: "f32[32, 3, 7, 7]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[32]" = torch.ops.aten.copy_.default(primals_122, add_2);  primals_122 = add_2 = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_123, add_3);  primals_123 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_124, add);  primals_124 = add = None
    copy__3: "f32[192]" = torch.ops.aten.copy_.default(primals_125, add_7);  primals_125 = add_7 = None
    copy__4: "f32[192]" = torch.ops.aten.copy_.default(primals_126, add_8);  primals_126 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_127, add_5);  primals_127 = add_5 = None
    copy__6: "f32[192]" = torch.ops.aten.copy_.default(primals_128, add_13);  primals_128 = add_13 = None
    copy__7: "f32[192]" = torch.ops.aten.copy_.default(primals_129, add_14);  primals_129 = add_14 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_130, add_11);  primals_130 = add_11 = None
    copy__9: "f32[192]" = torch.ops.aten.copy_.default(primals_131, add_21);  primals_131 = add_21 = None
    copy__10: "f32[192]" = torch.ops.aten.copy_.default(primals_132, add_22);  primals_132 = add_22 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_133, add_19);  primals_133 = add_19 = None
    copy__12: "f32[192]" = torch.ops.aten.copy_.default(primals_134, add_29);  primals_134 = add_29 = None
    copy__13: "f32[192]" = torch.ops.aten.copy_.default(primals_135, add_30);  primals_135 = add_30 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_136, add_27);  primals_136 = add_27 = None
    copy__15: "f32[192]" = torch.ops.aten.copy_.default(primals_137, add_37);  primals_137 = add_37 = None
    copy__16: "f32[192]" = torch.ops.aten.copy_.default(primals_138, add_38);  primals_138 = add_38 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_139, add_35);  primals_139 = add_35 = None
    copy__18: "f32[192]" = torch.ops.aten.copy_.default(primals_140, add_45);  primals_140 = add_45 = None
    copy__19: "f32[192]" = torch.ops.aten.copy_.default(primals_141, add_46);  primals_141 = add_46 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_142, add_43);  primals_142 = add_43 = None
    copy__21: "f32[192]" = torch.ops.aten.copy_.default(primals_143, add_53);  primals_143 = add_53 = None
    copy__22: "f32[192]" = torch.ops.aten.copy_.default(primals_144, add_54);  primals_144 = add_54 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_145, add_51);  primals_145 = add_51 = None
    copy__24: "f32[192]" = torch.ops.aten.copy_.default(primals_146, add_61);  primals_146 = add_61 = None
    copy__25: "f32[192]" = torch.ops.aten.copy_.default(primals_147, add_62);  primals_147 = add_62 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_148, add_59);  primals_148 = add_59 = None
    copy__27: "f32[384]" = torch.ops.aten.copy_.default(primals_149, add_69);  primals_149 = add_69 = None
    copy__28: "f32[384]" = torch.ops.aten.copy_.default(primals_150, add_70);  primals_150 = add_70 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_151, add_67);  primals_151 = add_67 = None
    copy__30: "f32[384]" = torch.ops.aten.copy_.default(primals_152, add_75);  primals_152 = add_75 = None
    copy__31: "f32[384]" = torch.ops.aten.copy_.default(primals_153, add_76);  primals_153 = add_76 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_154, add_73);  primals_154 = add_73 = None
    copy__33: "f32[384]" = torch.ops.aten.copy_.default(primals_155, add_81);  primals_155 = add_81 = None
    copy__34: "f32[384]" = torch.ops.aten.copy_.default(primals_156, add_82);  primals_156 = add_82 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_157, add_79);  primals_157 = add_79 = None
    copy__36: "f32[384]" = torch.ops.aten.copy_.default(primals_158, add_88);  primals_158 = add_88 = None
    copy__37: "f32[384]" = torch.ops.aten.copy_.default(primals_159, add_89);  primals_159 = add_89 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_160, add_86);  primals_160 = add_86 = None
    copy__39: "f32[384]" = torch.ops.aten.copy_.default(primals_161, add_94);  primals_161 = add_94 = None
    copy__40: "f32[384]" = torch.ops.aten.copy_.default(primals_162, add_95);  primals_162 = add_95 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_163, add_92);  primals_163 = add_92 = None
    copy__42: "f32[384]" = torch.ops.aten.copy_.default(primals_164, add_101);  primals_164 = add_101 = None
    copy__43: "f32[384]" = torch.ops.aten.copy_.default(primals_165, add_102);  primals_165 = add_102 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_166, add_99);  primals_166 = add_99 = None
    copy__45: "f32[384]" = torch.ops.aten.copy_.default(primals_167, add_107);  primals_167 = add_107 = None
    copy__46: "f32[384]" = torch.ops.aten.copy_.default(primals_168, add_108);  primals_168 = add_108 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_169, add_105);  primals_169 = add_105 = None
    copy__48: "f32[384]" = torch.ops.aten.copy_.default(primals_170, add_114);  primals_170 = add_114 = None
    copy__49: "f32[384]" = torch.ops.aten.copy_.default(primals_171, add_115);  primals_171 = add_115 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_172, add_112);  primals_172 = add_112 = None
    copy__51: "f32[384]" = torch.ops.aten.copy_.default(primals_173, add_120);  primals_173 = add_120 = None
    copy__52: "f32[384]" = torch.ops.aten.copy_.default(primals_174, add_121);  primals_174 = add_121 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_175, add_118);  primals_175 = add_118 = None
    copy__54: "f32[768]" = torch.ops.aten.copy_.default(primals_176, add_127);  primals_176 = add_127 = None
    copy__55: "f32[768]" = torch.ops.aten.copy_.default(primals_177, add_128);  primals_177 = add_128 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_178, add_125);  primals_178 = add_125 = None
    copy__57: "f32[768]" = torch.ops.aten.copy_.default(primals_179, add_133);  primals_179 = add_133 = None
    copy__58: "f32[768]" = torch.ops.aten.copy_.default(primals_180, add_134);  primals_180 = add_134 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_181, add_131);  primals_181 = add_131 = None
    copy__60: "f32[768]" = torch.ops.aten.copy_.default(primals_182, add_139);  primals_182 = add_139 = None
    copy__61: "f32[768]" = torch.ops.aten.copy_.default(primals_183, add_140);  primals_183 = add_140 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_184, add_137);  primals_184 = add_137 = None
    copy__63: "f32[768]" = torch.ops.aten.copy_.default(primals_185, add_146);  primals_185 = add_146 = None
    copy__64: "f32[768]" = torch.ops.aten.copy_.default(primals_186, add_147);  primals_186 = add_147 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_187, add_144);  primals_187 = add_144 = None
    copy__66: "f32[768]" = torch.ops.aten.copy_.default(primals_188, add_152);  primals_188 = add_152 = None
    copy__67: "f32[768]" = torch.ops.aten.copy_.default(primals_189, add_153);  primals_189 = add_153 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_190, add_150);  primals_190 = add_150 = None
    copy__69: "f32[768]" = torch.ops.aten.copy_.default(primals_191, add_159);  primals_191 = add_159 = None
    copy__70: "f32[768]" = torch.ops.aten.copy_.default(primals_192, add_160);  primals_192 = add_160 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_193, add_157);  primals_193 = add_157 = None
    copy__72: "f32[768]" = torch.ops.aten.copy_.default(primals_194, add_165);  primals_194 = add_165 = None
    copy__73: "f32[768]" = torch.ops.aten.copy_.default(primals_195, add_166);  primals_195 = add_166 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_196, add_163);  primals_196 = add_163 = None
    copy__75: "f32[768]" = torch.ops.aten.copy_.default(primals_197, add_172);  primals_197 = add_172 = None
    copy__76: "f32[768]" = torch.ops.aten.copy_.default(primals_198, add_173);  primals_198 = add_173 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_199, add_170);  primals_199 = add_170 = None
    copy__78: "f32[768]" = torch.ops.aten.copy_.default(primals_200, add_178);  primals_200 = add_178 = None
    copy__79: "f32[768]" = torch.ops.aten.copy_.default(primals_201, add_179);  primals_201 = add_179 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_202, add_176);  primals_202 = add_176 = None
    copy__81: "f32[768]" = torch.ops.aten.copy_.default(primals_203, add_185);  primals_203 = add_185 = None
    copy__82: "f32[768]" = torch.ops.aten.copy_.default(primals_204, add_186);  primals_204 = add_186 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_205, add_183);  primals_205 = add_183 = None
    return pytree.tree_unflatten([addmm, sum_72, sum_55, sum_32, getitem_249, mul_699, sum_75, getitem_246, getitem_247, mul_690, sum_73, mul_681, sum_70, getitem_243, getitem_240, getitem_237, mul_658, sum_68, getitem_234, getitem_231, getitem_228, mul_635, sum_66, getitem_225, getitem_222, getitem_219, mul_612, sum_64, getitem_216, getitem_213, getitem_210, mul_589, sum_62, getitem_207, getitem_204, getitem_201, mul_566, sum_60, getitem_198, getitem_195, getitem_192, mul_543, sum_58, getitem_189, getitem_186, getitem_183, getitem_180, getitem_181, mul_520, sum_56, mul_511, sum_53, getitem_177, getitem_174, mul_499, sum_50, getitem_171, getitem_168, mul_483, sum_48, getitem_165, getitem_162, mul_471, sum_45, getitem_159, getitem_156, mul_455, sum_43, getitem_153, getitem_150, mul_443, sum_40, getitem_147, getitem_144, mul_427, sum_38, getitem_141, getitem_138, mul_415, sum_35, getitem_135, getitem_132, getitem_129, getitem_130, mul_399, sum_33, mul_390, sum_30, getitem_126, getitem_123, mul_378, sum_27, getitem_120, getitem_117, mul_362, sum_25, getitem_114, getitem_111, mul_350, sum_22, getitem_108, getitem_105, mul_334, sum_20, getitem_102, getitem_99, mul_322, sum_17, getitem_96, getitem_93, mul_306, sum_15, getitem_90, getitem_87, mul_294, sum_12, getitem_84, getitem_81, mul_278, sum_10, permute_28, view_65, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    