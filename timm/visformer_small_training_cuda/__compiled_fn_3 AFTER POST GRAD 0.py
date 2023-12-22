from __future__ import annotations



def forward(self, primals_1: "f32[1, 192, 28, 28]", primals_2: "f32[1, 384, 14, 14]", primals_3: "f32[1, 768, 7, 7]", primals_4: "f32[32, 3, 7, 7]", primals_5: "f32[32]", primals_6: "f32[32]", primals_7: "f32[192, 32, 4, 4]", primals_8: "f32[192]", primals_9: "f32[192]", primals_10: "f32[192]", primals_11: "f32[192]", primals_12: "f32[192]", primals_13: "f32[384, 192, 1, 1]", primals_14: "f32[384, 48, 3, 3]", primals_15: "f32[192, 384, 1, 1]", primals_16: "f32[192]", primals_17: "f32[192]", primals_18: "f32[384, 192, 1, 1]", primals_19: "f32[384, 48, 3, 3]", primals_20: "f32[192, 384, 1, 1]", primals_21: "f32[192]", primals_22: "f32[192]", primals_23: "f32[384, 192, 1, 1]", primals_24: "f32[384, 48, 3, 3]", primals_25: "f32[192, 384, 1, 1]", primals_26: "f32[192]", primals_27: "f32[192]", primals_28: "f32[384, 192, 1, 1]", primals_29: "f32[384, 48, 3, 3]", primals_30: "f32[192, 384, 1, 1]", primals_31: "f32[192]", primals_32: "f32[192]", primals_33: "f32[384, 192, 1, 1]", primals_34: "f32[384, 48, 3, 3]", primals_35: "f32[192, 384, 1, 1]", primals_36: "f32[192]", primals_37: "f32[192]", primals_38: "f32[384, 192, 1, 1]", primals_39: "f32[384, 48, 3, 3]", primals_40: "f32[192, 384, 1, 1]", primals_41: "f32[192]", primals_42: "f32[192]", primals_43: "f32[384, 192, 1, 1]", primals_44: "f32[384, 48, 3, 3]", primals_45: "f32[192, 384, 1, 1]", primals_46: "f32[384, 192, 2, 2]", primals_47: "f32[384]", primals_48: "f32[384]", primals_49: "f32[384]", primals_50: "f32[384]", primals_51: "f32[384]", primals_52: "f32[1152, 384, 1, 1]", primals_53: "f32[384, 384, 1, 1]", primals_54: "f32[384]", primals_55: "f32[384]", primals_56: "f32[1536, 384, 1, 1]", primals_57: "f32[384, 1536, 1, 1]", primals_58: "f32[384]", primals_59: "f32[384]", primals_60: "f32[1152, 384, 1, 1]", primals_61: "f32[384, 384, 1, 1]", primals_62: "f32[384]", primals_63: "f32[384]", primals_64: "f32[1536, 384, 1, 1]", primals_65: "f32[384, 1536, 1, 1]", primals_66: "f32[384]", primals_67: "f32[384]", primals_68: "f32[1152, 384, 1, 1]", primals_69: "f32[384, 384, 1, 1]", primals_70: "f32[384]", primals_71: "f32[384]", primals_72: "f32[1536, 384, 1, 1]", primals_73: "f32[384, 1536, 1, 1]", primals_74: "f32[384]", primals_75: "f32[384]", primals_76: "f32[1152, 384, 1, 1]", primals_77: "f32[384, 384, 1, 1]", primals_78: "f32[384]", primals_79: "f32[384]", primals_80: "f32[1536, 384, 1, 1]", primals_81: "f32[384, 1536, 1, 1]", primals_82: "f32[768, 384, 2, 2]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[2304, 768, 1, 1]", primals_89: "f32[768, 768, 1, 1]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[3072, 768, 1, 1]", primals_93: "f32[768, 3072, 1, 1]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[2304, 768, 1, 1]", primals_97: "f32[768, 768, 1, 1]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[3072, 768, 1, 1]", primals_101: "f32[768, 3072, 1, 1]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[2304, 768, 1, 1]", primals_105: "f32[768, 768, 1, 1]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[3072, 768, 1, 1]", primals_109: "f32[768, 3072, 1, 1]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[2304, 768, 1, 1]", primals_113: "f32[768, 768, 1, 1]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[3072, 768, 1, 1]", primals_117: "f32[768, 3072, 1, 1]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[1000, 768]", primals_121: "f32[1000]", primals_122: "f32[32]", primals_123: "f32[32]", primals_124: "i64[]", primals_125: "f32[192]", primals_126: "f32[192]", primals_127: "i64[]", primals_128: "f32[192]", primals_129: "f32[192]", primals_130: "i64[]", primals_131: "f32[192]", primals_132: "f32[192]", primals_133: "i64[]", primals_134: "f32[192]", primals_135: "f32[192]", primals_136: "i64[]", primals_137: "f32[192]", primals_138: "f32[192]", primals_139: "i64[]", primals_140: "f32[192]", primals_141: "f32[192]", primals_142: "i64[]", primals_143: "f32[192]", primals_144: "f32[192]", primals_145: "i64[]", primals_146: "f32[192]", primals_147: "f32[192]", primals_148: "i64[]", primals_149: "f32[384]", primals_150: "f32[384]", primals_151: "i64[]", primals_152: "f32[384]", primals_153: "f32[384]", primals_154: "i64[]", primals_155: "f32[384]", primals_156: "f32[384]", primals_157: "i64[]", primals_158: "f32[384]", primals_159: "f32[384]", primals_160: "i64[]", primals_161: "f32[384]", primals_162: "f32[384]", primals_163: "i64[]", primals_164: "f32[384]", primals_165: "f32[384]", primals_166: "i64[]", primals_167: "f32[384]", primals_168: "f32[384]", primals_169: "i64[]", primals_170: "f32[384]", primals_171: "f32[384]", primals_172: "i64[]", primals_173: "f32[384]", primals_174: "f32[384]", primals_175: "i64[]", primals_176: "f32[768]", primals_177: "f32[768]", primals_178: "i64[]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "i64[]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "i64[]", primals_185: "f32[768]", primals_186: "f32[768]", primals_187: "i64[]", primals_188: "f32[768]", primals_189: "f32[768]", primals_190: "i64[]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "i64[]", primals_194: "f32[768]", primals_195: "f32[768]", primals_196: "i64[]", primals_197: "f32[768]", primals_198: "f32[768]", primals_199: "i64[]", primals_200: "f32[768]", primals_201: "f32[768]", primals_202: "i64[]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "i64[]", primals_206: "f32[8, 3, 224, 224]"):
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_11: "i64[]" = torch.ops.aten.add.Tensor(primals_130, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 192, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 192, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_12: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_2: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_10, getitem_5)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_3: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_23, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_24: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.5)
    mul_25: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_1: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_17: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_26: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_24, add_17);  mul_24 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_4: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_26, primals_15, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_18: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_10, convolution_4)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_6: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_36, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_37: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_38: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_3: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_25: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, add_25);  mul_37 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_7: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_39, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_26: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_18, convolution_7);  add_18 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_9: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_49, primals_24, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_50: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
    mul_51: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_5: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_33: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_50, add_33);  mul_50 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_10: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_52, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_34: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_26, convolution_10);  add_26 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_12: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_62, primals_29, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_63: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_64: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_7: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_41: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_65: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_63, add_41);  mul_63 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_13: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_65, primals_30, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_42: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_34, convolution_13);  add_34 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_15: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_75, primals_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_76: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.5)
    mul_77: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.7071067811865476)
    erf_9: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_49: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_78: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_76, add_49);  mul_76 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_16: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_78, primals_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_42, convolution_16);  add_42 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_18: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_88, primals_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_89: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_90: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_11: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_57: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_91: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_89, add_57);  mul_89 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_19: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_91, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_58: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_50, convolution_19);  add_50 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_21: "f32[8, 384, 28, 28]" = torch.ops.aten.convolution.default(mul_101, primals_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_102: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.5)
    mul_103: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.7071067811865476)
    erf_13: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_65: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_104: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_102, add_65);  mul_102 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_22: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(mul_104, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_66: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_58, convolution_22);  add_58 = convolution_22 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_154, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 384, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 384, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_74: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_10: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_72, getitem_21)
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
    view: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_24, [8, 3, 6, 64, -1]);  convolution_24 = None
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
    view_1: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_16, [48, 196, 64]);  clone_16 = None
    expand_1: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_1, [8, 6, 64, 196]);  permute_1 = None
    clone_17: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_2: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_17, [48, 64, 196]);  clone_17 = None
    bmm: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_1, view_2)
    view_3: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm, [8, 6, 196, 196]);  bmm = None
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
    view_4: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_2, [48, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_24, [8, 6, 196, 64]);  getitem_24 = None
    clone_19: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_5: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_19, [48, 196, 64]);  clone_19 = None
    bmm_1: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_4, view_5)
    view_6: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 6, 196, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_2: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
    clone_20: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_7: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_20, [8, 384, 14, 14]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_25: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_7, primals_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_78: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_72, convolution_25)
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_27: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_129, primals_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_78, convolution_27);  add_78 = None
    
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
    view_8: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_28, [8, 3, 6, 64, -1]);  convolution_28 = None
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
    view_9: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_24, [48, 196, 64]);  clone_24 = None
    expand_5: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_4, [8, 6, 64, 196]);  permute_4 = None
    clone_25: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_10: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_25, [48, 64, 196]);  clone_25 = None
    bmm_2: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_2, [8, 6, 196, 196]);  bmm_2 = None
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
    view_12: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_6, [48, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_31, [8, 6, 196, 64]);  getitem_31 = None
    clone_27: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_13: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_27, [48, 196, 64]);  clone_27 = None
    bmm_3: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 6, 196, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_5: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 2]);  view_14 = None
    clone_28: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_15: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_28, [8, 384, 14, 14]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_29: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_15, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_91: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_85, convolution_29);  add_85 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_31: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_147, primals_65, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_98: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_91, convolution_31);  add_91 = None
    
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
    view_16: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_32, [8, 3, 6, 64, -1]);  convolution_32 = None
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
    view_17: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_32, [48, 196, 64]);  clone_32 = None
    expand_9: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_7, [8, 6, 64, 196]);  permute_7 = None
    clone_33: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_18: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_33, [48, 64, 196]);  clone_33 = None
    bmm_4: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_17, view_18)
    view_19: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_4, [8, 6, 196, 196]);  bmm_4 = None
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
    view_20: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_10, [48, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_38, [8, 6, 196, 64]);  getitem_38 = None
    clone_35: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_21: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_35, [48, 196, 64]);  clone_35 = None
    bmm_5: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_20, view_21)
    view_22: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 6, 196, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_8: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_22, [0, 1, 3, 2]);  view_22 = None
    clone_36: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_23: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_36, [8, 384, 14, 14]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_33: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_23, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_104: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_98, convolution_33);  add_98 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_35: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_165, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_111: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_104, convolution_35);  add_104 = None
    
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
    view_24: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.reshape.default(convolution_36, [8, 3, 6, 64, -1]);  convolution_36 = None
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
    view_25: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_40, [48, 196, 64]);  clone_40 = None
    expand_13: "f32[8, 6, 64, 196]" = torch.ops.aten.expand.default(permute_10, [8, 6, 64, 196]);  permute_10 = None
    clone_41: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_26: "f32[48, 64, 196]" = torch.ops.aten.reshape.default(clone_41, [48, 64, 196]);  clone_41 = None
    bmm_6: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, [8, 6, 196, 196]);  bmm_6 = None
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
    view_28: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(expand_14, [48, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 6, 196, 64]" = torch.ops.aten.expand.default(getitem_45, [8, 6, 196, 64]);  getitem_45 = None
    clone_43: "f32[8, 6, 196, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_29: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(clone_43, [48, 196, 64]);  clone_43 = None
    bmm_7: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_28, view_29)
    view_30: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 6, 196, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_11: "f32[8, 6, 64, 196]" = torch.ops.aten.permute.default(view_30, [0, 1, 3, 2]);  view_30 = None
    clone_44: "f32[8, 6, 64, 196]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_31: "f32[8, 384, 14, 14]" = torch.ops.aten.reshape.default(clone_44, [8, 384, 14, 14]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_37: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(view_31, primals_77, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_117: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_111, convolution_37);  add_111 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_39: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(mul_183, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_124: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_117, convolution_39);  add_117 = convolution_39 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_131: "i64[]" = torch.ops.aten.add.Tensor(primals_181, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_130, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 768, 1, 1]" = var_mean_19[0]
    getitem_51: "f32[1, 768, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_132: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_19: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_23: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_130, getitem_51)
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
    view_32: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_41, [8, 3, 6, 128, -1]);  convolution_41 = None
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
    view_33: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_49, [48, 49, 128]);  clone_49 = None
    expand_17: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_13, [8, 6, 128, 49]);  permute_13 = None
    clone_50: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_34: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_50, [48, 128, 49]);  clone_50 = None
    bmm_8: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_8, [8, 6, 49, 49]);  bmm_8 = None
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
    view_36: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_18, [48, 49, 49]);  expand_18 = None
    expand_19: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_54, [8, 6, 49, 128]);  getitem_54 = None
    clone_52: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_37: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_52, [48, 49, 128]);  clone_52 = None
    bmm_9: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_9, [8, 6, 49, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_14: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_38, [0, 1, 3, 2]);  view_38 = None
    clone_53: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_39: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_53, [8, 768, 7, 7]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_42: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_39, primals_89, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_136: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_130, convolution_42)
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_44: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_208, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_143: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_136, convolution_44);  add_136 = None
    
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
    view_40: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_45, [8, 3, 6, 128, -1]);  convolution_45 = None
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
    view_41: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_57, [48, 49, 128]);  clone_57 = None
    expand_21: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_16, [8, 6, 128, 49]);  permute_16 = None
    clone_58: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_42: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_58, [48, 128, 49]);  clone_58 = None
    bmm_10: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_10, [8, 6, 49, 49]);  bmm_10 = None
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
    view_44: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_22, [48, 49, 49]);  expand_22 = None
    expand_23: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_61, [8, 6, 49, 128]);  getitem_61 = None
    clone_60: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_45: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_60, [48, 49, 128]);  clone_60 = None
    bmm_11: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_11, [8, 6, 49, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_17: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2]);  view_46 = None
    clone_61: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_47: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_61, [8, 768, 7, 7]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_46: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_47, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_149: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_143, convolution_46);  add_143 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_48: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_226, primals_101, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_156: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_149, convolution_48);  add_149 = None
    
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
    view_48: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_49, [8, 3, 6, 128, -1]);  convolution_49 = None
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
    view_49: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_65, [48, 49, 128]);  clone_65 = None
    expand_25: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_19, [8, 6, 128, 49]);  permute_19 = None
    clone_66: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_50: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_66, [48, 128, 49]);  clone_66 = None
    bmm_12: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_49, view_50)
    view_51: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_12, [8, 6, 49, 49]);  bmm_12 = None
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
    view_52: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_26, [48, 49, 49]);  expand_26 = None
    expand_27: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_68, [8, 6, 49, 128]);  getitem_68 = None
    clone_68: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_53: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_68, [48, 49, 128]);  clone_68 = None
    bmm_13: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_52, view_53)
    view_54: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_13, [8, 6, 49, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_20: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 2]);  view_54 = None
    clone_69: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_55: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_69, [8, 768, 7, 7]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_50: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_55, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_162: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_156, convolution_50);  add_156 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_52: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_244, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_169: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_162, convolution_52);  add_162 = None
    
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
    view_56: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_53, [8, 3, 6, 128, -1]);  convolution_53 = None
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
    view_57: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_73, [48, 49, 128]);  clone_73 = None
    expand_29: "f32[8, 6, 128, 49]" = torch.ops.aten.expand.default(permute_22, [8, 6, 128, 49]);  permute_22 = None
    clone_74: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_58: "f32[48, 128, 49]" = torch.ops.aten.reshape.default(clone_74, [48, 128, 49]);  clone_74 = None
    bmm_14: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_57, view_58)
    view_59: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_14, [8, 6, 49, 49]);  bmm_14 = None
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
    view_60: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(expand_30, [48, 49, 49]);  expand_30 = None
    expand_31: "f32[8, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_75, [8, 6, 49, 128]);  getitem_75 = None
    clone_76: "f32[8, 6, 49, 128]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_61: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(clone_76, [48, 49, 128]);  clone_76 = None
    bmm_15: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_60, view_61)
    view_62: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_15, [8, 6, 49, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    permute_23: "f32[8, 6, 128, 49]" = torch.ops.aten.permute.default(view_62, [0, 1, 3, 2]);  view_62 = None
    clone_77: "f32[8, 6, 128, 49]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_63: "f32[8, 768, 7, 7]" = torch.ops.aten.reshape.default(clone_77, [8, 768, 7, 7]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_54: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(view_63, primals_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_175: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_169, convolution_54);  add_169 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_56: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_262, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_182: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_175, convolution_56);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_205, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_182, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 768, 1, 1]" = var_mean_27[0]
    getitem_79: "f32[1, 768, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_184: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_27: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_35: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, getitem_79);  add_182 = None
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
    view_64: "f32[8, 768]" = torch.ops.aten.reshape.default(mean, [8, 768]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:433, code: return x if pre_logits else self.head(x)
    permute_24: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_121, view_64, permute_24);  primals_121 = None
    permute_25: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    unsqueeze_112: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_113: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 2);  unsqueeze_112 = None
    unsqueeze_114: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 3);  unsqueeze_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_124: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_125: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 2);  unsqueeze_124 = None
    unsqueeze_126: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 3);  unsqueeze_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_30: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    permute_31: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_9: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_32: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    permute_33: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_136: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_137: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 2);  unsqueeze_136 = None
    unsqueeze_138: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 3);  unsqueeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_148: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_149: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
    unsqueeze_150: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_37: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    permute_38: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_10: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_39: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
    permute_40: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_160: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_161: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
    unsqueeze_162: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_172: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_173: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
    unsqueeze_174: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_44: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    permute_45: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_11: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_46: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    permute_47: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_184: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_185: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
    unsqueeze_186: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_196: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_197: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
    unsqueeze_198: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_51: "f32[48, 49, 49]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    permute_52: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_12: "f32[8, 6, 49, 49]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_53: "f32[48, 128, 49]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    permute_54: "f32[48, 49, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_208: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_209: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
    unsqueeze_210: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    unsqueeze_220: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_221: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
    unsqueeze_222: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_232: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_233: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_58: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    permute_59: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_13: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_60: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    permute_61: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_244: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_245: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_256: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_257: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_65: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
    permute_66: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_14: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_67: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    permute_68: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_268: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_269: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_280: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_281: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_72: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    permute_73: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_15: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_74: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_75: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_292: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_293: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_304: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_305: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    permute_79: "f32[48, 196, 196]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
    permute_80: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    alias_16: "f32[8, 6, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_81: "f32[48, 64, 196]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
    permute_82: "f32[48, 196, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    unsqueeze_316: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_317: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    unsqueeze_328: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_329: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    unsqueeze_340: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_341: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    unsqueeze_352: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_353: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    unsqueeze_364: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_365: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    unsqueeze_376: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_377: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    unsqueeze_388: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_389: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    unsqueeze_400: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_401: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    unsqueeze_412: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_413: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    unsqueeze_424: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_425: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:396, code: x = self.stem(x)
    unsqueeze_436: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_437: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    
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
    return [addmm, primals_4, primals_5, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_206, convolution, squeeze_1, relu, convolution_1, squeeze_4, add_10, squeeze_7, add_15, convolution_2, mul_23, convolution_3, mul_26, convolution_4, squeeze_10, add_23, convolution_5, mul_36, convolution_6, mul_39, convolution_7, squeeze_13, add_31, convolution_8, mul_49, convolution_9, mul_52, convolution_10, squeeze_16, add_39, convolution_11, mul_62, convolution_12, mul_65, convolution_13, squeeze_19, add_47, convolution_14, mul_75, convolution_15, mul_78, convolution_16, squeeze_22, add_55, convolution_17, mul_88, convolution_18, mul_91, convolution_19, squeeze_25, add_63, convolution_20, mul_101, convolution_21, mul_104, add_66, convolution_23, squeeze_28, add_72, squeeze_31, add_77, view_7, convolution_25, squeeze_34, add_83, convolution_26, mul_129, convolution_27, squeeze_37, add_90, view_15, convolution_29, squeeze_40, add_96, convolution_30, mul_147, convolution_31, squeeze_43, add_103, view_23, convolution_33, squeeze_46, add_109, convolution_34, mul_165, convolution_35, squeeze_49, add_116, view_31, convolution_37, squeeze_52, add_122, convolution_38, mul_183, add_124, convolution_40, squeeze_55, add_130, squeeze_58, add_135, view_39, convolution_42, squeeze_61, add_141, convolution_43, mul_208, convolution_44, squeeze_64, add_148, view_47, convolution_46, squeeze_67, add_154, convolution_47, mul_226, convolution_48, squeeze_70, add_161, view_55, convolution_50, squeeze_73, add_167, convolution_51, mul_244, convolution_52, squeeze_76, add_174, view_63, convolution_54, squeeze_79, add_180, convolution_55, mul_262, convolution_56, squeeze_82, view_64, permute_25, unsqueeze_114, unsqueeze_126, permute_30, permute_31, alias_9, permute_32, permute_33, unsqueeze_138, unsqueeze_150, permute_37, permute_38, alias_10, permute_39, permute_40, unsqueeze_162, unsqueeze_174, permute_44, permute_45, alias_11, permute_46, permute_47, unsqueeze_186, unsqueeze_198, permute_51, permute_52, alias_12, permute_53, permute_54, unsqueeze_210, unsqueeze_222, unsqueeze_234, permute_58, permute_59, alias_13, permute_60, permute_61, unsqueeze_246, unsqueeze_258, permute_65, permute_66, alias_14, permute_67, permute_68, unsqueeze_270, unsqueeze_282, permute_72, permute_73, alias_15, permute_74, permute_75, unsqueeze_294, unsqueeze_306, permute_79, permute_80, alias_16, permute_81, permute_82, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438]
    