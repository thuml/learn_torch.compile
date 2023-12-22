from __future__ import annotations



def forward(self, primals_1: "f32[24]", primals_2: "f32[24]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64]", primals_9: "f32[64]", primals_10: "f32[64]", primals_11: "f32[256]", primals_12: "f32[256]", primals_13: "f32[256]", primals_14: "f32[256]", primals_15: "f32[64]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[256]", primals_20: "f32[256]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[63, 32]", primals_38: "f32[63, 32]", primals_39: "f32[128]", primals_40: "f32[128]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[256]", primals_44: "f32[256]", primals_45: "f32[256]", primals_46: "f32[256]", primals_47: "f32[1024]", primals_48: "f32[1024]", primals_49: "f32[1024]", primals_50: "f32[1024]", primals_51: "f32[256]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[256]", primals_55: "f32[1024]", primals_56: "f32[1024]", primals_57: "f32[256]", primals_58: "f32[256]", primals_59: "f32[31, 64]", primals_60: "f32[31, 64]", primals_61: "f32[256]", primals_62: "f32[256]", primals_63: "f32[1024]", primals_64: "f32[1024]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[31, 128]", primals_68: "f32[31, 128]", primals_69: "f32[512]", primals_70: "f32[512]", primals_71: "f32[1536]", primals_72: "f32[1536]", primals_73: "f32[1536]", primals_74: "f32[1536]", primals_75: "f32[512]", primals_76: "f32[512]", primals_77: "f32[15, 128]", primals_78: "f32[15, 128]", primals_79: "f32[512]", primals_80: "f32[512]", primals_81: "f32[1536]", primals_82: "f32[1536]", primals_83: "f32[1280]", primals_84: "f32[1280]", primals_85: "f32[24, 3, 3, 3]", primals_86: "f32[32, 24, 3, 3]", primals_87: "f32[64, 32, 3, 3]", primals_88: "f32[64, 64, 1, 1]", primals_89: "f32[64, 64, 3, 3]", primals_90: "f32[8, 64, 1, 1]", primals_91: "f32[8]", primals_92: "f32[64, 8, 1, 1]", primals_93: "f32[64]", primals_94: "f32[256, 64, 1, 1]", primals_95: "f32[256, 64, 1, 1]", primals_96: "f32[64, 256, 1, 1]", primals_97: "f32[64, 64, 3, 3]", primals_98: "f32[8, 64, 1, 1]", primals_99: "f32[8]", primals_100: "f32[64, 8, 1, 1]", primals_101: "f32[64]", primals_102: "f32[256, 64, 1, 1]", primals_103: "f32[128, 256, 1, 1]", primals_104: "f32[128, 128, 3, 3]", primals_105: "f32[8, 128, 1, 1]", primals_106: "f32[8]", primals_107: "f32[128, 8, 1, 1]", primals_108: "f32[128]", primals_109: "f32[512, 128, 1, 1]", primals_110: "f32[512, 256, 1, 1]", primals_111: "f32[128, 512, 1, 1]", primals_112: "f32[128, 128, 3, 3]", primals_113: "f32[8, 128, 1, 1]", primals_114: "f32[8]", primals_115: "f32[128, 8, 1, 1]", primals_116: "f32[128]", primals_117: "f32[512, 128, 1, 1]", primals_118: "f32[128, 512, 1, 1]", primals_119: "f32[384, 128, 1, 1]", primals_120: "f32[512, 128, 1, 1]", primals_121: "f32[256, 512, 1, 1]", primals_122: "f32[256, 256, 3, 3]", primals_123: "f32[16, 256, 1, 1]", primals_124: "f32[16]", primals_125: "f32[256, 16, 1, 1]", primals_126: "f32[256]", primals_127: "f32[1024, 256, 1, 1]", primals_128: "f32[1024, 512, 1, 1]", primals_129: "f32[256, 1024, 1, 1]", primals_130: "f32[256, 256, 3, 3]", primals_131: "f32[16, 256, 1, 1]", primals_132: "f32[16]", primals_133: "f32[256, 16, 1, 1]", primals_134: "f32[256]", primals_135: "f32[1024, 256, 1, 1]", primals_136: "f32[256, 1024, 1, 1]", primals_137: "f32[768, 256, 1, 1]", primals_138: "f32[1024, 256, 1, 1]", primals_139: "f32[512, 1024, 1, 1]", primals_140: "f32[1536, 512, 1, 1]", primals_141: "f32[1536, 512, 1, 1]", primals_142: "f32[1536, 1024, 1, 1]", primals_143: "f32[512, 1536, 1, 1]", primals_144: "f32[1536, 512, 1, 1]", primals_145: "f32[1536, 512, 1, 1]", primals_146: "f32[1280, 1536, 1, 1]", primals_147: "f32[1000, 1280]", primals_148: "f32[1000]", primals_149: "i64[]", primals_150: "f32[24]", primals_151: "f32[24]", primals_152: "i64[]", primals_153: "f32[32]", primals_154: "f32[32]", primals_155: "i64[]", primals_156: "f32[64]", primals_157: "f32[64]", primals_158: "i64[]", primals_159: "f32[64]", primals_160: "f32[64]", primals_161: "i64[]", primals_162: "f32[64]", primals_163: "f32[64]", primals_164: "i64[]", primals_165: "f32[256]", primals_166: "f32[256]", primals_167: "i64[]", primals_168: "f32[256]", primals_169: "f32[256]", primals_170: "i64[]", primals_171: "f32[64]", primals_172: "f32[64]", primals_173: "i64[]", primals_174: "f32[64]", primals_175: "f32[64]", primals_176: "i64[]", primals_177: "f32[256]", primals_178: "f32[256]", primals_179: "i64[]", primals_180: "f32[128]", primals_181: "f32[128]", primals_182: "i64[]", primals_183: "f32[128]", primals_184: "f32[128]", primals_185: "i64[]", primals_186: "f32[512]", primals_187: "f32[512]", primals_188: "i64[]", primals_189: "f32[512]", primals_190: "f32[512]", primals_191: "i64[]", primals_192: "f32[128]", primals_193: "f32[128]", primals_194: "i64[]", primals_195: "f32[128]", primals_196: "f32[128]", primals_197: "i64[]", primals_198: "f32[512]", primals_199: "f32[512]", primals_200: "i64[]", primals_201: "f32[128]", primals_202: "f32[128]", primals_203: "i64[]", primals_204: "f32[128]", primals_205: "f32[128]", primals_206: "i64[]", primals_207: "f32[512]", primals_208: "f32[512]", primals_209: "i64[]", primals_210: "f32[256]", primals_211: "f32[256]", primals_212: "i64[]", primals_213: "f32[256]", primals_214: "f32[256]", primals_215: "i64[]", primals_216: "f32[1024]", primals_217: "f32[1024]", primals_218: "i64[]", primals_219: "f32[1024]", primals_220: "f32[1024]", primals_221: "i64[]", primals_222: "f32[256]", primals_223: "f32[256]", primals_224: "i64[]", primals_225: "f32[256]", primals_226: "f32[256]", primals_227: "i64[]", primals_228: "f32[1024]", primals_229: "f32[1024]", primals_230: "i64[]", primals_231: "f32[256]", primals_232: "f32[256]", primals_233: "i64[]", primals_234: "f32[256]", primals_235: "f32[256]", primals_236: "i64[]", primals_237: "f32[1024]", primals_238: "f32[1024]", primals_239: "i64[]", primals_240: "f32[512]", primals_241: "f32[512]", primals_242: "i64[]", primals_243: "f32[512]", primals_244: "f32[512]", primals_245: "i64[]", primals_246: "f32[1536]", primals_247: "f32[1536]", primals_248: "i64[]", primals_249: "f32[1536]", primals_250: "f32[1536]", primals_251: "i64[]", primals_252: "f32[512]", primals_253: "f32[512]", primals_254: "i64[]", primals_255: "f32[512]", primals_256: "f32[512]", primals_257: "i64[]", primals_258: "f32[1536]", primals_259: "f32[1536]", primals_260: "i64[]", primals_261: "f32[1280]", primals_262: "f32[1280]", primals_263: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 24, 128, 128]" = torch.ops.aten.convolution.default(primals_263, primals_85, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_149, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 24, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 24, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[24]" = torch.ops.aten.mul.Tensor(primals_150, 0.9)
    add_2: "f32[24]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[24]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[24]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[24]" = torch.ops.aten.mul.Tensor(primals_151, 0.9)
    add_3: "f32[24]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(mul_7, primals_86, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_152, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    mul_15: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_15, primals_87, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_155, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_156, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.000030518509476);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_157, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_2: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_14)
    mul_23: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_14, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_23, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_158, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 64, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
    add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
    mul_28: "f32[64]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_19)
    mul_31: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_3);  sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_31, primals_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_161, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_32: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_34: "f32[64]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.000030518509476);  squeeze_14 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(mul_35, 0.1);  mul_35 = None
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_38: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_32, unsqueeze_17);  mul_32 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_19);  mul_38 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_24)
    mul_39: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_24, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(mul_39, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_5: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_90, primals_91, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_6: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(relu, primals_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_40: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_39, sigmoid_5);  mul_39 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_40, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_164, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 256, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_11)
    mul_41: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_42: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
    add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.000030518509476);  squeeze_17 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(mul_44, 0.1);  mul_44 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
    unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_47: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_41, unsqueeze_21);  mul_41 = unsqueeze_21 = None
    unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_23);  mul_47 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_23, primals_95, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_167, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
    mul_48: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_49: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(mul_51, 0.1);  mul_51 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_54: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_25);  mul_48 = unsqueeze_25 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_54, unsqueeze_27);  mul_54 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_35: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_29, add_34);  add_29 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_6: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_35)
    mul_55: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_55, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_170, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_15)
    mul_56: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
    add_39: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_62: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_29);  mul_56 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_31);  mul_62 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_40)
    mul_63: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, sigmoid_7);  sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(mul_63, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_17)
    mul_64: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[64]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
    add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_70: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_33);  mul_64 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_35);  mul_70 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_45)
    mul_71: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_45, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(mul_71, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_11: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_98, primals_99, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_1: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_12: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_100, primals_101, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 64, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_72: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_71, sigmoid_9);  mul_71 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(mul_72, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_176, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 256, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 256, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_19)
    mul_73: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
    add_48: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    squeeze_29: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_76: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_77: "f32[256]" = torch.ops.aten.mul.Tensor(mul_76, 0.1);  mul_76 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
    add_49: "f32[256]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
    unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_79: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_37);  mul_73 = unsqueeze_37 = None
    unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_39);  mul_79 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_51: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_50, mul_55);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_10: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_51)
    mul_80: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_51, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(mul_80, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_21)
    mul_81: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_83: "f32[128]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_84: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.000030518509476);  squeeze_32 = None
    mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
    mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
    add_55: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_87: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_41);  mul_81 = unsqueeze_41 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_43);  mul_87 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_11: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_56)
    mul_88: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_11);  sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_88, primals_104, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_182, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_23)
    mul_89: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
    add_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001220852154804);  squeeze_35 = None
    mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(mul_92, 0.1);  mul_92 = None
    mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
    add_60: "f32[128]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_95: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_45);  mul_89 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_47);  mul_95 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_61)
    mul_96: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_61, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(mul_96, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_16: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_105, primals_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_2: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_17: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_2, primals_107, primals_108, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_13: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_97: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_96, sigmoid_13);  mul_96 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_97, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_25)
    mul_98: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
    add_64: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_38: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001220852154804);  squeeze_38 = None
    mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_65: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_48: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_104: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_49);  mul_98 = unsqueeze_49 = None
    unsqueeze_50: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_51);  mul_104 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_80, primals_110, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_188, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_27)
    mul_105: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_106: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_107: "f32[512]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_69: "f32[512]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_108: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001220852154804);  squeeze_41 = None
    mul_109: "f32[512]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[512]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
    add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_111: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_53);  mul_105 = unsqueeze_53 = None
    unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_55);  mul_111 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_72: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_66, add_71);  add_66 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_14: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_72)
    mul_112: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_72, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_112, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_29)
    mul_113: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_75: "f32[128]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    squeeze_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
    mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
    add_76: "f32[128]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    unsqueeze_56: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_119: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_57);  mul_113 = unsqueeze_57 = None
    unsqueeze_58: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_59);  mul_119 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_77)
    mul_120: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_77, sigmoid_15);  sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_120, primals_112, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_194, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_31)
    mul_121: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_122: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_123: "f32[128]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_80: "f32[128]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_124: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_125: "f32[128]" = torch.ops.aten.mul.Tensor(mul_124, 0.1);  mul_124 = None
    mul_126: "f32[128]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
    add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_127: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_61);  mul_121 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_63);  mul_127 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_82)
    mul_128: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(mul_128, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_22: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_113, primals_114, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_23: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_115, primals_116, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_17: "f32[8, 128, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_129: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_128, sigmoid_17);  mul_128 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_129, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
    mul_130: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_131: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_132: "f32[512]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_85: "f32[512]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
    squeeze_50: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_133: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_134: "f32[512]" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
    mul_135: "f32[512]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
    add_86: "f32[512]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    unsqueeze_64: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_136: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_65);  mul_130 = unsqueeze_65 = None
    unsqueeze_66: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_67);  mul_136 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_88: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_87, mul_112);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_18: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_88)
    mul_137: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_88, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(mul_137, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_200, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_35)
    mul_138: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_139: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_140: "f32[128]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_91: "f32[128]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_142: "f32[128]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[128]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
    add_92: "f32[128]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_144: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_69);  mul_138 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_71);  mul_144 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
    mul_145: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sigmoid_19);  sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_26: "f32[8, 384, 32, 32]" = torch.ops.aten.convolution.default(mul_145, primals_119, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(convolution_26, [128, 128, 128], 1);  convolution_26 = None
    getitem_36: "f32[8, 128, 32, 32]" = split_with_sizes[0]
    getitem_37: "f32[8, 128, 32, 32]" = split_with_sizes[1]
    getitem_38: "f32[8, 128, 32, 32]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_16: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_36, memory_format = torch.contiguous_format);  getitem_36 = None
    view: "f32[32, 32, 1024]" = torch.ops.aten.reshape.default(clone_16, [32, 32, 1024]);  clone_16 = None
    permute: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_17: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_37, memory_format = torch.contiguous_format);  getitem_37 = None
    view_1: "f32[32, 32, 1024]" = torch.ops.aten.reshape.default(clone_17, [32, 32, 1024]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_18: "f32[8, 128, 32, 32]" = torch.ops.aten.clone.default(getitem_38, memory_format = torch.contiguous_format);  getitem_38 = None
    view_2: "f32[32, 32, 1024]" = torch.ops.aten.reshape.default(clone_18, [32, 32, 1024]);  clone_18 = None
    permute_1: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand: "f32[32, 1024, 32]" = torch.ops.aten.expand.default(permute, [32, 1024, 32])
    expand_1: "f32[32, 32, 1024]" = torch.ops.aten.expand.default(view_1, [32, 32, 1024]);  view_1 = None
    bmm: "f32[32, 1024, 1024]" = torch.ops.aten.bmm.default(expand, expand_1)
    mul_146: "f32[32, 1024, 1024]" = torch.ops.aten.mul.Tensor(bmm, 0.1767766952966369);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_6: "f32[32, 32, 32, 32]" = torch.ops.aten.reshape.default(permute, [32, 32, 32, 32]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_2: "f32[32, 63]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    clone_19: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(view_6, memory_format = torch.contiguous_format)
    view_7: "f32[32768, 32]" = torch.ops.aten.reshape.default(clone_19, [32768, 32]);  clone_19 = None
    mm: "f32[32768, 63]" = torch.ops.aten.mm.default(view_7, permute_2)
    view_8: "f32[32, 32, 32, 63]" = torch.ops.aten.reshape.default(mm, [32, 32, 32, 63]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_9: "f32[1024, 32, 63]" = torch.ops.aten.reshape.default(view_8, [-1, 32, 63]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd: "f32[1024, 32, 64]" = torch.ops.aten.constant_pad_nd.default(view_9, [0, 1], 0.0);  view_9 = None
    view_10: "f32[1024, 2048]" = torch.ops.aten.reshape.default(constant_pad_nd, [1024, 2048]);  constant_pad_nd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_1: "f32[1024, 2079]" = torch.ops.aten.constant_pad_nd.default(view_10, [0, 31], 0.0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_11: "f32[1024, 33, 63]" = torch.ops.aten.reshape.default(constant_pad_nd_1, [-1, 33, 63]);  constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_2: "f32[1024, 32, 63]" = torch.ops.aten.slice.Tensor(view_11, 1, 0, 32);  view_11 = None
    slice_3: "f32[1024, 32, 32]" = torch.ops.aten.slice.Tensor(slice_2, 2, 31, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_12: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.reshape.default(slice_3, [32, 32, 1, 32, 32]);  slice_3 = None
    expand_2: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.expand.default(view_12, [-1, -1, 32, -1, -1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_3: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(expand_2, [0, 1, 3, 2, 4]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_4: "f32[32, 32, 32, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_5: "f32[32, 63]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    clone_20: "f32[32, 32, 32, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_13: "f32[32768, 32]" = torch.ops.aten.reshape.default(clone_20, [32768, 32]);  clone_20 = None
    mm_1: "f32[32768, 63]" = torch.ops.aten.mm.default(view_13, permute_5)
    view_14: "f32[32, 32, 32, 63]" = torch.ops.aten.reshape.default(mm_1, [32, 32, 32, 63]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_15: "f32[1024, 32, 63]" = torch.ops.aten.reshape.default(view_14, [-1, 32, 63]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_2: "f32[1024, 32, 64]" = torch.ops.aten.constant_pad_nd.default(view_15, [0, 1], 0.0);  view_15 = None
    view_16: "f32[1024, 2048]" = torch.ops.aten.reshape.default(constant_pad_nd_2, [1024, 2048]);  constant_pad_nd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_3: "f32[1024, 2079]" = torch.ops.aten.constant_pad_nd.default(view_16, [0, 31], 0.0);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_17: "f32[1024, 33, 63]" = torch.ops.aten.reshape.default(constant_pad_nd_3, [-1, 33, 63]);  constant_pad_nd_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_5: "f32[1024, 32, 63]" = torch.ops.aten.slice.Tensor(view_17, 1, 0, 32);  view_17 = None
    slice_6: "f32[1024, 32, 32]" = torch.ops.aten.slice.Tensor(slice_5, 2, 31, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_18: "f32[32, 32, 1, 32, 32]" = torch.ops.aten.reshape.default(slice_6, [32, 32, 1, 32, 32]);  slice_6 = None
    expand_3: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.expand.default(view_18, [-1, -1, 32, -1, -1]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_6: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.permute.default(expand_3, [0, 3, 1, 4, 2]);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_94: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.add.Tensor(permute_6, permute_3);  permute_6 = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_21: "f32[32, 32, 32, 32, 32]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format);  add_94 = None
    view_19: "f32[32, 1024, 1024]" = torch.ops.aten.reshape.default(clone_21, [32, 1024, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_95: "f32[32, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_146, view_19);  mul_146 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax: "f32[32, 1024, 1]" = torch.ops.aten.amax.default(add_95, [-1], True)
    sub_18: "f32[32, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_95, amax);  add_95 = amax = None
    exp: "f32[32, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_1: "f32[32, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[32, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_8: "f32[32, 1024, 1024]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_4: "f32[32, 1024, 1024]" = torch.ops.aten.expand.default(div, [32, 1024, 1024]);  div = None
    expand_5: "f32[32, 1024, 32]" = torch.ops.aten.expand.default(permute_1, [32, 1024, 32]);  permute_1 = None
    bmm_1: "f32[32, 1024, 32]" = torch.ops.aten.bmm.default(expand_4, expand_5)
    permute_7: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1])
    clone_22: "f32[32, 32, 1024]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_23: "f32[8, 128, 32, 32]" = torch.ops.aten.reshape.default(clone_22, [8, 128, 32, 32]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_96: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(view_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_39: "f32[1, 128, 1, 1]" = var_mean_18[0]
    getitem_40: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_97: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05)
    rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(view_23, getitem_40);  view_23 = None
    mul_147: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = None
    squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_98: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_99: "f32[128]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_153: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_73);  mul_147 = unsqueeze_73 = None
    unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_100: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_75);  mul_153 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_100)
    mul_154: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_100, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_154, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_206, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_41: "f32[1, 512, 1, 1]" = var_mean_19[0]
    getitem_42: "f32[1, 512, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_102: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05)
    rsqrt_19: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_20: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_42)
    mul_155: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = None
    squeeze_57: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    squeeze_58: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_156: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_157: "f32[512]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
    add_103: "f32[512]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_59: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    mul_158: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_159: "f32[512]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[512]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_104: "f32[512]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    unsqueeze_76: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_77: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_161: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_77);  mul_155 = unsqueeze_77 = None
    unsqueeze_78: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_79: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_105: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_79);  mul_161 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_106: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_105, mul_137);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_21: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_106)
    mul_162: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_106, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(mul_162, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_43: "f32[1, 256, 1, 1]" = var_mean_20[0]
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_108: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05)
    rsqrt_20: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_21: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_44)
    mul_163: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = None
    squeeze_60: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    squeeze_61: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_164: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_165: "f32[256]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
    add_109: "f32[256]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    squeeze_62: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    mul_166: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001220852154804);  squeeze_62 = None
    mul_167: "f32[256]" = torch.ops.aten.mul.Tensor(mul_166, 0.1);  mul_166 = None
    mul_168: "f32[256]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_110: "f32[256]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
    unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_169: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_81);  mul_163 = unsqueeze_81 = None
    unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_111: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_83);  mul_169 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_22: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_111)
    mul_170: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_111, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_170, primals_122, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_112: "i64[]" = torch.ops.aten.add.Tensor(primals_212, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_21[0]
    getitem_46: "f32[1, 256, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_113: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_45, 1e-05)
    rsqrt_21: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_22: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_46)
    mul_171: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = None
    squeeze_63: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    squeeze_64: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
    add_114: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_65: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_175: "f32[256]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_115: "f32[256]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_177: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_85);  mul_171 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_116: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_87);  mul_177 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_23: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_116)
    mul_178: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_23);  sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(mul_178, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_30: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_123, primals_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_4: "f32[8, 16, 1, 1]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_31: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_4, primals_125, primals_126, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_24: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_179: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_178, sigmoid_24);  mul_178 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_179, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_215, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_47: "f32[1, 1024, 1, 1]" = var_mean_22[0]
    getitem_48: "f32[1, 1024, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-05)
    rsqrt_22: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_48)
    mul_180: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = None
    squeeze_66: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    squeeze_67: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_181: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_182: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_119: "f32[1024]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    squeeze_68: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    mul_183: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_184: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_183, 0.1);  mul_183 = None
    mul_185: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_120: "f32[1024]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    unsqueeze_88: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_89: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_186: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_89);  mul_180 = unsqueeze_89 = None
    unsqueeze_90: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_91: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_91);  mul_186 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_162, primals_128, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_218, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_49: "f32[1, 1024, 1, 1]" = var_mean_23[0]
    getitem_50: "f32[1, 1024, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05)
    rsqrt_23: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_24: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_50)
    mul_187: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = None
    squeeze_69: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    squeeze_70: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_188: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_189: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_124: "f32[1024]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    squeeze_71: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    mul_190: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_191: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_190, 0.1);  mul_190 = None
    mul_192: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_125: "f32[1024]" = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
    unsqueeze_92: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_93: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_193: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_93);  mul_187 = unsqueeze_93 = None
    unsqueeze_94: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_95: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_95);  mul_193 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_127: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_121, add_126);  add_121 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_25: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_127)
    mul_194: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_127, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_194, primals_129, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_221, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_51: "f32[1, 256, 1, 1]" = var_mean_24[0]
    getitem_52: "f32[1, 256, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_129: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_51, 1e-05)
    rsqrt_24: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_52)
    mul_195: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
    squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_196: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_130: "f32[256]" = torch.ops.aten.add.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(mul_198, 0.1);  mul_198 = None
    mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_131: "f32[256]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_201: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_97);  mul_195 = unsqueeze_97 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_132: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_201, unsqueeze_99);  mul_201 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_26: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_132)
    mul_202: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_132, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_202, primals_130, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_224, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_53: "f32[1, 256, 1, 1]" = var_mean_25[0]
    getitem_54: "f32[1, 256, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05)
    rsqrt_25: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_26: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_54)
    mul_203: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_25);  sub_26 = None
    squeeze_75: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    squeeze_76: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_135: "f32[256]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_77: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0004885197850513);  squeeze_77 = None
    mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[256]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_136: "f32[256]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_209: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_101);  mul_203 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_137: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_103);  mul_209 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_27: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_137)
    mul_210: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_27);  sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(mul_210, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_36: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_5: "f32[8, 16, 1, 1]" = torch.ops.aten.relu.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_37: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_133, primals_134, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_28: "f32[8, 256, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_211: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_210, sigmoid_28);  mul_210 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_211, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_227, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_55: "f32[1, 1024, 1, 1]" = var_mean_26[0]
    getitem_56: "f32[1, 1024, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_139: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05)
    rsqrt_26: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_27: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_56)
    mul_212: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_26);  sub_27 = None
    squeeze_78: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    squeeze_79: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_213: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_214: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_140: "f32[1024]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    squeeze_80: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    mul_215: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0004885197850513);  squeeze_80 = None
    mul_216: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_215, 0.1);  mul_215 = None
    mul_217: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_141: "f32[1024]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    unsqueeze_104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_218: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_105);  mul_212 = unsqueeze_105 = None
    unsqueeze_106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_142: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_107);  mul_218 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_143: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_142, mul_194);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_29: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_143)
    mul_219: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_143, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(mul_219, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_144: "i64[]" = torch.ops.aten.add.Tensor(primals_230, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_57: "f32[1, 256, 1, 1]" = var_mean_27[0]
    getitem_58: "f32[1, 256, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_145: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05)
    rsqrt_27: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_28: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_58)
    mul_220: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_27);  sub_28 = None
    squeeze_81: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    squeeze_82: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_146: "f32[256]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    squeeze_83: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    mul_223: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0004885197850513);  squeeze_83 = None
    mul_224: "f32[256]" = torch.ops.aten.mul.Tensor(mul_223, 0.1);  mul_223 = None
    mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_147: "f32[256]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_226: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_109);  mul_220 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_148: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_111);  mul_226 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_30: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_148)
    mul_227: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_148, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_40: "f32[8, 768, 16, 16]" = torch.ops.aten.convolution.default(mul_227, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(convolution_40, [256, 256, 256], 1);  convolution_40 = None
    getitem_59: "f32[8, 256, 16, 16]" = split_with_sizes_1[0]
    getitem_60: "f32[8, 256, 16, 16]" = split_with_sizes_1[1]
    getitem_61: "f32[8, 256, 16, 16]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_32: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_59, memory_format = torch.contiguous_format);  getitem_59 = None
    view_24: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_32, [32, 64, 256]);  clone_32 = None
    permute_8: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_33: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_60, memory_format = torch.contiguous_format);  getitem_60 = None
    view_25: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_33, [32, 64, 256]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_34: "f32[8, 256, 16, 16]" = torch.ops.aten.clone.default(getitem_61, memory_format = torch.contiguous_format);  getitem_61 = None
    view_26: "f32[32, 64, 256]" = torch.ops.aten.reshape.default(clone_34, [32, 64, 256]);  clone_34 = None
    permute_9: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_6: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_8, [32, 256, 64])
    expand_7: "f32[32, 64, 256]" = torch.ops.aten.expand.default(view_25, [32, 64, 256]);  view_25 = None
    bmm_2: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_6, expand_7)
    mul_228: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_2, 0.125);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_30: "f32[32, 16, 16, 64]" = torch.ops.aten.reshape.default(permute_8, [32, 16, 16, 64]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_10: "f32[64, 31]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    clone_35: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(view_30, memory_format = torch.contiguous_format)
    view_31: "f32[8192, 64]" = torch.ops.aten.reshape.default(clone_35, [8192, 64]);  clone_35 = None
    mm_2: "f32[8192, 31]" = torch.ops.aten.mm.default(view_31, permute_10)
    view_32: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_2, [32, 16, 16, 31]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_33: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_32, [-1, 16, 31]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_4: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_33, [0, 1], 0.0);  view_33 = None
    view_34: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_4, [512, 512]);  constant_pad_nd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_5: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_34, [0, 15], 0.0);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_35: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_5, [-1, 17, 31]);  constant_pad_nd_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_8: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_35, 1, 0, 16);  view_35 = None
    slice_9: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_8, 2, 15, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_36: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_9, [32, 16, 1, 16, 16]);  slice_9 = None
    expand_8: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_36, [-1, -1, 16, -1, -1]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_11: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_8, [0, 1, 3, 2, 4]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_12: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_13: "f32[64, 31]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    clone_36: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
    view_37: "f32[8192, 64]" = torch.ops.aten.reshape.default(clone_36, [8192, 64]);  clone_36 = None
    mm_3: "f32[8192, 31]" = torch.ops.aten.mm.default(view_37, permute_13)
    view_38: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_3, [32, 16, 16, 31]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_39: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_38, [-1, 16, 31]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_6: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_39, [0, 1], 0.0);  view_39 = None
    view_40: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_6, [512, 512]);  constant_pad_nd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_7: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_40, [0, 15], 0.0);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_41: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_7, [-1, 17, 31]);  constant_pad_nd_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_11: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_41, 1, 0, 16);  view_41 = None
    slice_12: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_11, 2, 15, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_42: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_12, [32, 16, 1, 16, 16]);  slice_12 = None
    expand_9: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_42, [-1, -1, 16, -1, -1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_9, [0, 3, 1, 4, 2]);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_149: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_14, permute_11);  permute_14 = permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_37: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format);  add_149 = None
    view_43: "f32[32, 256, 256]" = torch.ops.aten.reshape.default(clone_37, [32, 256, 256]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_150: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_228, view_43);  mul_228 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
    sub_29: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_150, amax_1);  add_150 = amax_1 = None
    exp_1: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_2: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_13: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_10: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_1, [32, 256, 256]);  div_1 = None
    expand_11: "f32[32, 256, 64]" = torch.ops.aten.expand.default(permute_9, [32, 256, 64]);  permute_9 = None
    bmm_3: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(expand_10, expand_11)
    permute_15: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_3, [0, 2, 1])
    clone_38: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_47: "f32[8, 256, 16, 16]" = torch.ops.aten.reshape.default(clone_38, [8, 256, 16, 16]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_233, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(view_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 256, 1, 1]" = var_mean_28[0]
    getitem_63: "f32[1, 256, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_152: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_28: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_30: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_47, getitem_63);  view_47 = None
    mul_229: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_28);  sub_30 = None
    squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_230: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_231: "f32[256]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_153: "f32[256]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_232: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0004885197850513);  squeeze_86 = None
    mul_233: "f32[256]" = torch.ops.aten.mul.Tensor(mul_232, 0.1);  mul_232 = None
    mul_234: "f32[256]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_154: "f32[256]" = torch.ops.aten.add.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_235: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_113);  mul_229 = unsqueeze_113 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_155: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_115);  mul_235 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_31: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_155)
    mul_236: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_155, sigmoid_31);  sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(mul_236, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_236, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 1024, 1, 1]" = var_mean_29[0]
    getitem_65: "f32[1, 1024, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_157: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_29: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_31: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_65)
    mul_237: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_29);  sub_31 = None
    squeeze_87: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_88: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_238: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_239: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_158: "f32[1024]" = torch.ops.aten.add.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
    squeeze_89: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_240: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0004885197850513);  squeeze_89 = None
    mul_241: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_240, 0.1);  mul_240 = None
    mul_242: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_159: "f32[1024]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    unsqueeze_116: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_117: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_243: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_117);  mul_237 = unsqueeze_117 = None
    unsqueeze_118: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_119: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_160: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_243, unsqueeze_119);  mul_243 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_161: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_160, mul_219);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_32: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_161)
    mul_244: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_32);  sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(mul_244, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_239, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 512, 1, 1]" = var_mean_30[0]
    getitem_67: "f32[1, 512, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_163: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_30: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_32: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_67)
    mul_245: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_30);  sub_32 = None
    squeeze_90: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_91: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_247: "f32[512]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_164: "f32[512]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_92: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0004885197850513);  squeeze_92 = None
    mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[512]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_165: "f32[512]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_120: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_121: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_251: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_121);  mul_245 = unsqueeze_121 = None
    unsqueeze_122: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_123: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_166: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_123);  mul_251 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_166)
    mul_252: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_33);  sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_43: "f32[8, 1536, 16, 16]" = torch.ops.aten.convolution.default(mul_252, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(convolution_43, [512, 512, 512], 1);  convolution_43 = None
    getitem_68: "f32[8, 512, 16, 16]" = split_with_sizes_2[0]
    getitem_69: "f32[8, 512, 16, 16]" = split_with_sizes_2[1]
    getitem_70: "f32[8, 512, 16, 16]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_42: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_68, memory_format = torch.contiguous_format);  getitem_68 = None
    view_48: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_42, [32, 128, 256]);  clone_42 = None
    permute_16: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_43: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_69, memory_format = torch.contiguous_format);  getitem_69 = None
    view_49: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_43, [32, 128, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_44: "f32[8, 512, 16, 16]" = torch.ops.aten.clone.default(getitem_70, memory_format = torch.contiguous_format);  getitem_70 = None
    view_50: "f32[32, 128, 256]" = torch.ops.aten.reshape.default(clone_44, [32, 128, 256]);  clone_44 = None
    permute_17: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_12: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_16, [32, 256, 128])
    expand_13: "f32[32, 128, 256]" = torch.ops.aten.expand.default(view_49, [32, 128, 256]);  view_49 = None
    bmm_4: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(expand_12, expand_13)
    mul_253: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_4, 0.08838834764831845);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_54: "f32[32, 16, 16, 128]" = torch.ops.aten.reshape.default(permute_16, [32, 16, 16, 128]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_18: "f32[128, 31]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    clone_45: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(view_54, memory_format = torch.contiguous_format)
    view_55: "f32[8192, 128]" = torch.ops.aten.reshape.default(clone_45, [8192, 128]);  clone_45 = None
    mm_4: "f32[8192, 31]" = torch.ops.aten.mm.default(view_55, permute_18)
    view_56: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_4, [32, 16, 16, 31]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_57: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_56, [-1, 16, 31]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_8: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_57, [0, 1], 0.0);  view_57 = None
    view_58: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_8, [512, 512]);  constant_pad_nd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_9: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_58, [0, 15], 0.0);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_59: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_9, [-1, 17, 31]);  constant_pad_nd_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_14: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_59, 1, 0, 16);  view_59 = None
    slice_15: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_14, 2, 15, 9223372036854775807);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_60: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_15, [32, 16, 1, 16, 16]);  slice_15 = None
    expand_14: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_60, [-1, -1, 16, -1, -1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_19: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_14, [0, 1, 3, 2, 4]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_20: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_21: "f32[128, 31]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    clone_46: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_61: "f32[8192, 128]" = torch.ops.aten.reshape.default(clone_46, [8192, 128]);  clone_46 = None
    mm_5: "f32[8192, 31]" = torch.ops.aten.mm.default(view_61, permute_21)
    view_62: "f32[32, 16, 16, 31]" = torch.ops.aten.reshape.default(mm_5, [32, 16, 16, 31]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_63: "f32[512, 16, 31]" = torch.ops.aten.reshape.default(view_62, [-1, 16, 31]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_10: "f32[512, 16, 32]" = torch.ops.aten.constant_pad_nd.default(view_63, [0, 1], 0.0);  view_63 = None
    view_64: "f32[512, 512]" = torch.ops.aten.reshape.default(constant_pad_nd_10, [512, 512]);  constant_pad_nd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_11: "f32[512, 527]" = torch.ops.aten.constant_pad_nd.default(view_64, [0, 15], 0.0);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_65: "f32[512, 17, 31]" = torch.ops.aten.reshape.default(constant_pad_nd_11, [-1, 17, 31]);  constant_pad_nd_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_17: "f32[512, 16, 31]" = torch.ops.aten.slice.Tensor(view_65, 1, 0, 16);  view_65 = None
    slice_18: "f32[512, 16, 16]" = torch.ops.aten.slice.Tensor(slice_17, 2, 15, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_66: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.reshape.default(slice_18, [32, 16, 1, 16, 16]);  slice_18 = None
    expand_15: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.expand.default(view_66, [-1, -1, 16, -1, -1]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_22: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(expand_15, [0, 3, 1, 4, 2]);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_167: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.add.Tensor(permute_22, permute_19);  permute_22 = permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_47: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format);  add_167 = None
    view_67: "f32[32, 256, 256]" = torch.ops.aten.reshape.default(clone_47, [32, 256, 256]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_168: "f32[32, 256, 256]" = torch.ops.aten.add.Tensor(mul_253, view_67);  mul_253 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[32, 256, 1]" = torch.ops.aten.amax.default(add_168, [-1], True)
    sub_33: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(add_168, amax_2);  add_168 = amax_2 = None
    exp_2: "f32[32, 256, 256]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_3: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[32, 256, 256]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_14: "f32[32, 256, 256]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_16: "f32[32, 256, 256]" = torch.ops.aten.expand.default(div_2, [32, 256, 256]);  div_2 = None
    expand_17: "f32[32, 256, 128]" = torch.ops.aten.expand.default(permute_17, [32, 256, 128]);  permute_17 = None
    bmm_5: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(expand_16, expand_17)
    permute_23: "f32[32, 128, 256]" = torch.ops.aten.permute.default(bmm_5, [0, 2, 1]);  bmm_5 = None
    clone_48: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_71: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(clone_48, [8, 512, 16, 16]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(view_71, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_242, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(avg_pool2d, [0, 2, 3], correction = 0, keepdim = True)
    getitem_71: "f32[1, 512, 1, 1]" = var_mean_31[0]
    getitem_72: "f32[1, 512, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_170: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_31: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_34: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_72)
    mul_254: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_31);  sub_34 = None
    squeeze_93: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    squeeze_94: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_255: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_256: "f32[512]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_171: "f32[512]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    squeeze_95: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    mul_257: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0019569471624266);  squeeze_95 = None
    mul_258: "f32[512]" = torch.ops.aten.mul.Tensor(mul_257, 0.1);  mul_257 = None
    mul_259: "f32[512]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_172: "f32[512]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    unsqueeze_124: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_125: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_260: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_125);  mul_254 = unsqueeze_125 = None
    unsqueeze_126: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_127: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_173: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_127);  mul_260 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_34: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_173)
    mul_261: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_173, sigmoid_34);  sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_261, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_245, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_73: "f32[1, 1536, 1, 1]" = var_mean_32[0]
    getitem_74: "f32[1, 1536, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_175: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_73, 1e-05)
    rsqrt_32: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_35: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_74)
    mul_262: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_32);  sub_35 = None
    squeeze_96: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    squeeze_97: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_263: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_264: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_176: "f32[1536]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    squeeze_98: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    mul_265: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0019569471624266);  squeeze_98 = None
    mul_266: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_265, 0.1);  mul_265 = None
    mul_267: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_177: "f32[1536]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    unsqueeze_128: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_129: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_268: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_129);  mul_262 = unsqueeze_129 = None
    unsqueeze_130: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_131: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_178: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_131);  mul_268 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_244, primals_142, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_179: "i64[]" = torch.ops.aten.add.Tensor(primals_248, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_75: "f32[1, 1536, 1, 1]" = var_mean_33[0]
    getitem_76: "f32[1, 1536, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_180: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05)
    rsqrt_33: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_36: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_76)
    mul_269: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_33);  sub_36 = None
    squeeze_99: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    squeeze_100: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_270: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_271: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
    add_181: "f32[1536]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    squeeze_101: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    mul_272: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0019569471624266);  squeeze_101 = None
    mul_273: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_272, 0.1);  mul_272 = None
    mul_274: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_182: "f32[1536]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
    unsqueeze_132: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_133: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_275: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_133);  mul_269 = unsqueeze_133 = None
    unsqueeze_134: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_135: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_183: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_135);  mul_275 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_184: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(add_178, add_183);  add_178 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_35: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_184)
    mul_276: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_184, sigmoid_35);  sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(mul_276, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_251, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_77: "f32[1, 512, 1, 1]" = var_mean_34[0]
    getitem_78: "f32[1, 512, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_186: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05)
    rsqrt_34: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_37: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_78)
    mul_277: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_34);  sub_37 = None
    squeeze_102: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    squeeze_103: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_278: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_279: "f32[512]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_187: "f32[512]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    squeeze_104: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    mul_280: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0019569471624266);  squeeze_104 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(mul_280, 0.1);  mul_280 = None
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_188: "f32[512]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_283: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_137);  mul_277 = unsqueeze_137 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_189: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_139);  mul_283 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_189)
    mul_284: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_189, sigmoid_36);  sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_47: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_284, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(convolution_47, [512, 512, 512], 1);  convolution_47 = None
    getitem_79: "f32[8, 512, 8, 8]" = split_with_sizes_3[0]
    getitem_80: "f32[8, 512, 8, 8]" = split_with_sizes_3[1]
    getitem_81: "f32[8, 512, 8, 8]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    clone_52: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_79, memory_format = torch.contiguous_format);  getitem_79 = None
    view_72: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_52, [32, 128, 64]);  clone_52 = None
    permute_24: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    clone_53: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_80, memory_format = torch.contiguous_format);  getitem_80 = None
    view_73: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_53, [32, 128, 64]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    clone_54: "f32[8, 512, 8, 8]" = torch.ops.aten.clone.default(getitem_81, memory_format = torch.contiguous_format);  getitem_81 = None
    view_74: "f32[32, 128, 64]" = torch.ops.aten.reshape.default(clone_54, [32, 128, 64]);  clone_54 = None
    permute_25: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    expand_18: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_24, [32, 64, 128])
    expand_19: "f32[32, 128, 64]" = torch.ops.aten.expand.default(view_73, [32, 128, 64]);  view_73 = None
    bmm_6: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(expand_18, expand_19)
    mul_285: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(bmm_6, 0.08838834764831845);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    view_78: "f32[32, 8, 8, 128]" = torch.ops.aten.reshape.default(permute_24, [32, 8, 8, 128]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_26: "f32[128, 15]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    clone_55: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(view_78, memory_format = torch.contiguous_format)
    view_79: "f32[2048, 128]" = torch.ops.aten.reshape.default(clone_55, [2048, 128]);  clone_55 = None
    mm_6: "f32[2048, 15]" = torch.ops.aten.mm.default(view_79, permute_26)
    view_80: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(mm_6, [32, 8, 8, 15]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_81: "f32[256, 8, 15]" = torch.ops.aten.reshape.default(view_80, [-1, 8, 15]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_12: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_81, [0, 1], 0.0);  view_81 = None
    view_82: "f32[256, 128]" = torch.ops.aten.reshape.default(constant_pad_nd_12, [256, 128]);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_13: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_82, [0, 7], 0.0);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_83: "f32[256, 9, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_13, [-1, 9, 15]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_20: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_83, 1, 0, 8);  view_83 = None
    slice_21: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_20, 2, 7, 9223372036854775807);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_84: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.reshape.default(slice_21, [32, 8, 1, 8, 8]);  slice_21 = None
    expand_20: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_84, [-1, -1, 8, -1, -1]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_27: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_20, [0, 1, 3, 2, 4]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_28: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_29: "f32[128, 15]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    clone_56: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    view_85: "f32[2048, 128]" = torch.ops.aten.reshape.default(clone_56, [2048, 128]);  clone_56 = None
    mm_7: "f32[2048, 15]" = torch.ops.aten.mm.default(view_85, permute_29)
    view_86: "f32[32, 8, 8, 15]" = torch.ops.aten.reshape.default(mm_7, [32, 8, 8, 15]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_87: "f32[256, 8, 15]" = torch.ops.aten.reshape.default(view_86, [-1, 8, 15]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    constant_pad_nd_14: "f32[256, 8, 16]" = torch.ops.aten.constant_pad_nd.default(view_87, [0, 1], 0.0);  view_87 = None
    view_88: "f32[256, 128]" = torch.ops.aten.reshape.default(constant_pad_nd_14, [256, 128]);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_15: "f32[256, 135]" = torch.ops.aten.constant_pad_nd.default(view_88, [0, 7], 0.0);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_89: "f32[256, 9, 15]" = torch.ops.aten.reshape.default(constant_pad_nd_15, [-1, 9, 15]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_23: "f32[256, 8, 15]" = torch.ops.aten.slice.Tensor(view_89, 1, 0, 8);  view_89 = None
    slice_24: "f32[256, 8, 8]" = torch.ops.aten.slice.Tensor(slice_23, 2, 7, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    view_90: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.reshape.default(slice_24, [32, 8, 1, 8, 8]);  slice_24 = None
    expand_21: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.expand.default(view_90, [-1, -1, 8, -1, -1]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_30: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(expand_21, [0, 3, 1, 4, 2]);  expand_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:79, code: rel_logits = rel_logits_h + rel_logits_w
    add_190: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.add.Tensor(permute_30, permute_27);  permute_30 = permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    clone_57: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.clone.default(add_190, memory_format = torch.contiguous_format);  add_190 = None
    view_91: "f32[32, 64, 64]" = torch.ops.aten.reshape.default(clone_57, [32, 64, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_191: "f32[32, 64, 64]" = torch.ops.aten.add.Tensor(mul_285, view_91);  mul_285 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[32, 64, 1]" = torch.ops.aten.amax.default(add_191, [-1], True)
    sub_38: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(add_191, amax_3);  add_191 = amax_3 = None
    exp_3: "f32[32, 64, 64]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_4: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[32, 64, 64]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_15: "f32[32, 64, 64]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    expand_22: "f32[32, 64, 64]" = torch.ops.aten.expand.default(div_3, [32, 64, 64]);  div_3 = None
    expand_23: "f32[32, 64, 128]" = torch.ops.aten.expand.default(permute_25, [32, 64, 128]);  permute_25 = None
    bmm_7: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(expand_22, expand_23)
    permute_31: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_7, [0, 2, 1])
    clone_58: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_95: "f32[8, 512, 8, 8]" = torch.ops.aten.reshape.default(clone_58, [8, 512, 8, 8]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_254, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(view_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1, 1]" = var_mean_35[0]
    getitem_83: "f32[1, 512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_193: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_35: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_39: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_95, getitem_83);  view_95 = None
    mul_286: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_35);  sub_39 = None
    squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_194: "f32[512]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0019569471624266);  squeeze_107 = None
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
    mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_195: "f32[512]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_292: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_141);  mul_286 = unsqueeze_141 = None
    unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_196: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_143);  mul_292 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_196)
    mul_293: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_196, sigmoid_37);  sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 1536, 8, 8]" = torch.ops.aten.convolution.default(mul_293, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_197: "i64[]" = torch.ops.aten.add.Tensor(primals_257, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1536, 1, 1]" = var_mean_36[0]
    getitem_85: "f32[1, 1536, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_198: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_36: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    sub_40: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_85)
    mul_294: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_36);  sub_40 = None
    squeeze_108: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_109: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_295: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_296: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_199: "f32[1536]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_110: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_297: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0019569471624266);  squeeze_110 = None
    mul_298: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_200: "f32[1536]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_144: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_145: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_300: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_145);  mul_294 = unsqueeze_145 = None
    unsqueeze_146: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_147: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_201: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_147);  mul_300 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:888, code: x = x + self.shortcut(shortcut)
    add_202: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Tensor(add_201, mul_276);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_38: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_202)
    mul_301: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_202, sigmoid_38);  sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 1280, 8, 8]" = torch.ops.aten.convolution.default(mul_301, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1280, 1, 1]" = var_mean_37[0]
    getitem_87: "f32[1, 1280, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_204: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_37: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_41: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_87)
    mul_302: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_37);  sub_41 = None
    squeeze_111: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_112: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_303: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_304: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_205: "f32[1280]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    squeeze_113: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_305: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0019569471624266);  squeeze_113 = None
    mul_306: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_305, 0.1);  mul_305 = None
    mul_307: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_206: "f32[1280]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    unsqueeze_148: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_149: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_308: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_149);  mul_302 = unsqueeze_149 = None
    unsqueeze_150: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_151: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_207: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_151);  mul_308 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_39: "f32[8, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_207)
    mul_309: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_207, sigmoid_39);  sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_6: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_309, [-1, -2], True);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_96: "f32[8, 1280]" = torch.ops.aten.reshape.default(mean_6, [8, 1280]);  mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute_32: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_148, view_96, permute_32);  primals_148 = None
    permute_33: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 1280, 8, 8]" = torch.ops.aten.sigmoid.default(add_207)
    full_default: "f32[8, 1280, 8, 8]" = torch.ops.aten.full.default([8, 1280, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_42: "f32[8, 1280, 8, 8]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_40);  full_default = None
    mul_310: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(add_207, sub_42);  add_207 = sub_42 = None
    add_208: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Scalar(mul_310, 1);  mul_310 = None
    mul_311: "f32[8, 1280, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_208);  sigmoid_40 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_152: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_153: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
    unsqueeze_154: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_41: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_202)
    full_default_1: "f32[8, 1536, 8, 8]" = torch.ops.aten.full.default([8, 1536, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_47: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_41)
    mul_322: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_202, sub_47);  add_202 = sub_47 = None
    add_209: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Scalar(mul_322, 1);  mul_322 = None
    mul_323: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_209);  sigmoid_41 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_164: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_165: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
    unsqueeze_166: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_196)
    full_default_2: "f32[8, 512, 8, 8]" = torch.ops.aten.full.default([8, 512, 8, 8], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_52: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_42)
    mul_334: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_196, sub_52);  add_196 = sub_52 = None
    add_210: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_334, 1);  mul_334 = None
    mul_335: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_210);  sigmoid_42 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_177: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_41: "f32[32, 64, 64]" = torch.ops.aten.permute.default(expand_22, [0, 2, 1]);  expand_22 = None
    permute_42: "f32[32, 128, 64]" = torch.ops.aten.permute.default(expand_23, [0, 2, 1]);  expand_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_16: "f32[32, 64, 64]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_46: "f32[15, 128]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_52: "f32[15, 128]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_54: "f32[32, 128, 64]" = torch.ops.aten.permute.default(expand_18, [0, 2, 1]);  expand_18 = None
    permute_55: "f32[32, 64, 128]" = torch.ops.aten.permute.default(expand_19, [0, 2, 1]);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_189)
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_43)
    mul_349: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_189, sub_58);  add_189 = sub_58 = None
    add_213: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_349, 1);  mul_349 = None
    mul_350: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_213);  sigmoid_43 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_189: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_44: "f32[8, 1536, 8, 8]" = torch.ops.aten.sigmoid.default(add_184)
    sub_63: "f32[8, 1536, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_44);  full_default_1 = None
    mul_361: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(add_184, sub_63);  add_184 = sub_63 = None
    add_215: "f32[8, 1536, 8, 8]" = torch.ops.aten.add.Scalar(mul_361, 1);  mul_361 = None
    mul_362: "f32[8, 1536, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_215);  sigmoid_44 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_201: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_213: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 512, 8, 8]" = torch.ops.aten.sigmoid.default(add_173)
    sub_72: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_45);  full_default_2 = None
    mul_382: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_173, sub_72);  add_173 = sub_72 = None
    add_216: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Scalar(mul_382, 1);  mul_382 = None
    mul_383: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sigmoid_45, add_216);  sigmoid_45 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_225: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_62: "f32[32, 256, 256]" = torch.ops.aten.permute.default(expand_16, [0, 2, 1]);  expand_16 = None
    permute_63: "f32[32, 128, 256]" = torch.ops.aten.permute.default(expand_17, [0, 2, 1]);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_17: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_67: "f32[31, 128]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_73: "f32[31, 128]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_75: "f32[32, 128, 256]" = torch.ops.aten.permute.default(expand_12, [0, 2, 1]);  expand_12 = None
    permute_76: "f32[32, 256, 128]" = torch.ops.aten.permute.default(expand_13, [0, 2, 1]);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_46: "f32[8, 512, 16, 16]" = torch.ops.aten.sigmoid.default(add_166)
    full_default_18: "f32[8, 512, 16, 16]" = torch.ops.aten.full.default([8, 512, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_78: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_18, sigmoid_46);  full_default_18 = None
    mul_397: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(add_166, sub_78);  add_166 = sub_78 = None
    add_219: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Scalar(mul_397, 1);  mul_397 = None
    mul_398: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_219);  sigmoid_46 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_237: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_47: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_161)
    full_default_19: "f32[8, 1024, 16, 16]" = torch.ops.aten.full.default([8, 1024, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_83: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_19, sigmoid_47)
    mul_409: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_161, sub_83);  add_161 = sub_83 = None
    add_221: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_409, 1);  mul_409 = None
    mul_410: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_47, add_221);  sigmoid_47 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_249: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_155)
    full_default_20: "f32[8, 256, 16, 16]" = torch.ops.aten.full.default([8, 256, 16, 16], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_88: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_48)
    mul_421: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_155, sub_88);  add_155 = sub_88 = None
    add_222: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_421, 1);  mul_421 = None
    mul_422: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_48, add_222);  sigmoid_48 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_261: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_83: "f32[32, 256, 256]" = torch.ops.aten.permute.default(expand_10, [0, 2, 1]);  expand_10 = None
    permute_84: "f32[32, 64, 256]" = torch.ops.aten.permute.default(expand_11, [0, 2, 1]);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_18: "f32[32, 256, 256]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_88: "f32[31, 64]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_94: "f32[31, 64]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_96: "f32[32, 64, 256]" = torch.ops.aten.permute.default(expand_6, [0, 2, 1]);  expand_6 = None
    permute_97: "f32[32, 256, 64]" = torch.ops.aten.permute.default(expand_7, [0, 2, 1]);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_148)
    sub_94: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_49)
    mul_436: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_148, sub_94);  add_148 = sub_94 = None
    add_225: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_436, 1);  mul_436 = None
    mul_437: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_49, add_225);  sigmoid_49 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_273: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_50: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_143)
    sub_99: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_19, sigmoid_50)
    mul_448: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_143, sub_99);  add_143 = sub_99 = None
    add_227: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_448, 1);  mul_448 = None
    mul_449: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_50, add_227);  sigmoid_50 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_285: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_297: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sigmoid.default(add_132)
    sub_110: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_52);  full_default_20 = None
    mul_476: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_132, sub_110);  add_132 = sub_110 = None
    add_230: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Scalar(mul_476, 1);  mul_476 = None
    mul_477: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_52, add_230);  sigmoid_52 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_309: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_53: "f32[8, 1024, 16, 16]" = torch.ops.aten.sigmoid.default(add_127)
    sub_115: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(full_default_19, sigmoid_53);  full_default_19 = None
    mul_488: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(add_127, sub_115);  add_127 = sub_115 = None
    add_232: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Scalar(mul_488, 1);  mul_488 = None
    mul_489: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sigmoid_53, add_232);  sigmoid_53 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_321: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_333: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_345: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_55: "f32[8, 256, 32, 32]" = torch.ops.aten.sigmoid.default(add_111)
    full_default_35: "f32[8, 256, 32, 32]" = torch.ops.aten.full.default([8, 256, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_130: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_35, sigmoid_55);  full_default_35 = None
    mul_525: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(add_111, sub_130);  add_111 = sub_130 = None
    add_235: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Scalar(mul_525, 1);  mul_525 = None
    mul_526: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_55, add_235);  sigmoid_55 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_357: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    sigmoid_56: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_106)
    full_default_36: "f32[8, 512, 32, 32]" = torch.ops.aten.full.default([8, 512, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_135: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_56)
    mul_537: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_106, sub_135);  add_106 = sub_135 = None
    add_237: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_537, 1);  mul_537 = None
    mul_538: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_56, add_237);  sigmoid_56 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_369: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_100)
    full_default_37: "f32[8, 128, 32, 32]" = torch.ops.aten.full.default([8, 128, 32, 32], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_140: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_57)
    mul_549: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_100, sub_140);  add_100 = sub_140 = None
    add_238: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_549, 1);  mul_549 = None
    mul_550: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_57, add_238);  sigmoid_57 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_381: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_110: "f32[32, 1024, 1024]" = torch.ops.aten.permute.default(expand_4, [0, 2, 1]);  expand_4 = None
    permute_111: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(expand_5, [0, 2, 1]);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[32, 1024, 1024]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_115: "f32[63, 32]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_121: "f32[63, 32]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    permute_123: "f32[32, 32, 1024]" = torch.ops.aten.permute.default(expand, [0, 2, 1]);  expand = None
    permute_124: "f32[32, 1024, 32]" = torch.ops.aten.permute.default(expand_1, [0, 2, 1]);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_58: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_93)
    sub_146: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_58)
    mul_564: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_93, sub_146);  add_93 = sub_146 = None
    add_241: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_564, 1);  mul_564 = None
    mul_565: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_58, add_241);  sigmoid_58 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_393: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_59: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_88)
    sub_151: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_59)
    mul_576: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_88, sub_151);  add_88 = sub_151 = None
    add_243: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_576, 1);  mul_576 = None
    mul_577: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_59, add_243);  sigmoid_59 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_405: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_417: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 128, 32, 32]" = torch.ops.aten.sigmoid.default(add_77)
    sub_162: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_61);  full_default_37 = None
    mul_604: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_77, sub_162);  add_77 = sub_162 = None
    add_246: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Scalar(mul_604, 1);  mul_604 = None
    mul_605: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_61, add_246);  sigmoid_61 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_429: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_62: "f32[8, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_72)
    sub_167: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_62);  full_default_36 = None
    mul_616: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_72, sub_167);  add_72 = sub_167 = None
    add_248: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Scalar(mul_616, 1);  mul_616 = None
    mul_617: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sigmoid_62, add_248);  sigmoid_62 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_441: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_453: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_465: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_64: "f32[8, 128, 64, 64]" = torch.ops.aten.sigmoid.default(add_56)
    full_default_52: "f32[8, 128, 64, 64]" = torch.ops.aten.full.default([8, 128, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_182: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_52, sigmoid_64);  full_default_52 = None
    mul_653: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(add_56, sub_182);  add_56 = sub_182 = None
    add_251: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Scalar(mul_653, 1);  mul_653 = None
    mul_654: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_251);  sigmoid_64 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_477: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_65: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_51)
    full_default_53: "f32[8, 256, 64, 64]" = torch.ops.aten.full.default([8, 256, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_187: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_53, sigmoid_65)
    mul_665: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_51, sub_187);  add_51 = sub_187 = None
    add_253: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_665, 1);  mul_665 = None
    mul_666: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_253);  sigmoid_65 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_489: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_55: "f32[8, 64, 64, 64]" = torch.ops.aten.full.default([8, 64, 64, 64], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_501: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_67: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_40)
    sub_198: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_67)
    mul_693: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_40, sub_198);  add_40 = sub_198 = None
    add_256: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_693, 1);  mul_693 = None
    mul_694: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_256);  sigmoid_67 = add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_512: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_513: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    sigmoid_68: "f32[8, 256, 64, 64]" = torch.ops.aten.sigmoid.default(add_35)
    sub_203: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_53, sigmoid_68);  full_default_53 = None
    mul_705: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(add_35, sub_203);  add_35 = sub_203 = None
    add_258: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Scalar(mul_705, 1);  mul_705 = None
    mul_706: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_258);  sigmoid_68 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_524: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_525: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_537: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_548: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_549: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_70: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_19)
    sub_218: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_70)
    mul_742: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_19, sub_218);  add_19 = sub_218 = None
    add_261: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_742, 1);  mul_742 = None
    mul_743: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_261);  sigmoid_70 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_560: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_561: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_71: "f32[8, 64, 64, 64]" = torch.ops.aten.sigmoid.default(add_14)
    sub_223: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(full_default_55, sigmoid_71);  full_default_55 = None
    mul_754: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_14, sub_223);  add_14 = sub_223 = None
    add_263: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Scalar(mul_754, 1);  mul_754 = None
    mul_755: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_263);  sigmoid_71 = add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_572: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_573: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_72: "f32[8, 32, 128, 128]" = torch.ops.aten.sigmoid.default(add_9)
    full_default_62: "f32[8, 32, 128, 128]" = torch.ops.aten.full.default([8, 32, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_228: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_62, sigmoid_72);  full_default_62 = None
    mul_766: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, sub_228);  add_9 = sub_228 = None
    add_264: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Scalar(mul_766, 1);  mul_766 = None
    mul_767: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_264);  sigmoid_72 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_585: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_73: "f32[8, 24, 128, 128]" = torch.ops.aten.sigmoid.default(add_4)
    full_default_63: "f32[8, 24, 128, 128]" = torch.ops.aten.full.default([8, 24, 128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_233: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(full_default_63, sigmoid_73);  full_default_63 = None
    mul_778: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, sub_233);  add_4 = sub_233 = None
    add_265: "f32[8, 24, 128, 128]" = torch.ops.aten.add.Scalar(mul_778, 1);  mul_778 = None
    mul_779: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_265);  sigmoid_73 = add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_596: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_597: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_149, add);  primals_149 = add = None
    copy__1: "f32[24]" = torch.ops.aten.copy_.default(primals_150, add_2);  primals_150 = add_2 = None
    copy__2: "f32[24]" = torch.ops.aten.copy_.default(primals_151, add_3);  primals_151 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_152, add_5);  primals_152 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_153, add_7);  primals_153 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_154, add_8);  primals_154 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_155, add_10);  primals_155 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_156, add_12);  primals_156 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_157, add_13);  primals_157 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_158, add_15);  primals_158 = add_15 = None
    copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_159, add_17);  primals_159 = add_17 = None
    copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_160, add_18);  primals_160 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_161, add_20);  primals_161 = add_20 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_162, add_22);  primals_162 = add_22 = None
    copy__14: "f32[64]" = torch.ops.aten.copy_.default(primals_163, add_23);  primals_163 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_164, add_25);  primals_164 = add_25 = None
    copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_165, add_27);  primals_165 = add_27 = None
    copy__17: "f32[256]" = torch.ops.aten.copy_.default(primals_166, add_28);  primals_166 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_167, add_30);  primals_167 = add_30 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_168, add_32);  primals_168 = add_32 = None
    copy__20: "f32[256]" = torch.ops.aten.copy_.default(primals_169, add_33);  primals_169 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_170, add_36);  primals_170 = add_36 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_171, add_38);  primals_171 = add_38 = None
    copy__23: "f32[64]" = torch.ops.aten.copy_.default(primals_172, add_39);  primals_172 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_41);  primals_173 = add_41 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_174, add_43);  primals_174 = add_43 = None
    copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_175, add_44);  primals_175 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_176, add_46);  primals_176 = add_46 = None
    copy__28: "f32[256]" = torch.ops.aten.copy_.default(primals_177, add_48);  primals_177 = add_48 = None
    copy__29: "f32[256]" = torch.ops.aten.copy_.default(primals_178, add_49);  primals_178 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_52);  primals_179 = add_52 = None
    copy__31: "f32[128]" = torch.ops.aten.copy_.default(primals_180, add_54);  primals_180 = add_54 = None
    copy__32: "f32[128]" = torch.ops.aten.copy_.default(primals_181, add_55);  primals_181 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_182, add_57);  primals_182 = add_57 = None
    copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_183, add_59);  primals_183 = add_59 = None
    copy__35: "f32[128]" = torch.ops.aten.copy_.default(primals_184, add_60);  primals_184 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_62);  primals_185 = add_62 = None
    copy__37: "f32[512]" = torch.ops.aten.copy_.default(primals_186, add_64);  primals_186 = add_64 = None
    copy__38: "f32[512]" = torch.ops.aten.copy_.default(primals_187, add_65);  primals_187 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_188, add_67);  primals_188 = add_67 = None
    copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_189, add_69);  primals_189 = add_69 = None
    copy__41: "f32[512]" = torch.ops.aten.copy_.default(primals_190, add_70);  primals_190 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_73);  primals_191 = add_73 = None
    copy__43: "f32[128]" = torch.ops.aten.copy_.default(primals_192, add_75);  primals_192 = add_75 = None
    copy__44: "f32[128]" = torch.ops.aten.copy_.default(primals_193, add_76);  primals_193 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_194, add_78);  primals_194 = add_78 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_195, add_80);  primals_195 = add_80 = None
    copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_196, add_81);  primals_196 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_83);  primals_197 = add_83 = None
    copy__49: "f32[512]" = torch.ops.aten.copy_.default(primals_198, add_85);  primals_198 = add_85 = None
    copy__50: "f32[512]" = torch.ops.aten.copy_.default(primals_199, add_86);  primals_199 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_200, add_89);  primals_200 = add_89 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_201, add_91);  primals_201 = add_91 = None
    copy__53: "f32[128]" = torch.ops.aten.copy_.default(primals_202, add_92);  primals_202 = add_92 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_96);  primals_203 = add_96 = None
    copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_204, add_98);  primals_204 = add_98 = None
    copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_205, add_99);  primals_205 = add_99 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_206, add_101);  primals_206 = add_101 = None
    copy__58: "f32[512]" = torch.ops.aten.copy_.default(primals_207, add_103);  primals_207 = add_103 = None
    copy__59: "f32[512]" = torch.ops.aten.copy_.default(primals_208, add_104);  primals_208 = add_104 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_107);  primals_209 = add_107 = None
    copy__61: "f32[256]" = torch.ops.aten.copy_.default(primals_210, add_109);  primals_210 = add_109 = None
    copy__62: "f32[256]" = torch.ops.aten.copy_.default(primals_211, add_110);  primals_211 = add_110 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_212, add_112);  primals_212 = add_112 = None
    copy__64: "f32[256]" = torch.ops.aten.copy_.default(primals_213, add_114);  primals_213 = add_114 = None
    copy__65: "f32[256]" = torch.ops.aten.copy_.default(primals_214, add_115);  primals_214 = add_115 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_215, add_117);  primals_215 = add_117 = None
    copy__67: "f32[1024]" = torch.ops.aten.copy_.default(primals_216, add_119);  primals_216 = add_119 = None
    copy__68: "f32[1024]" = torch.ops.aten.copy_.default(primals_217, add_120);  primals_217 = add_120 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_218, add_122);  primals_218 = add_122 = None
    copy__70: "f32[1024]" = torch.ops.aten.copy_.default(primals_219, add_124);  primals_219 = add_124 = None
    copy__71: "f32[1024]" = torch.ops.aten.copy_.default(primals_220, add_125);  primals_220 = add_125 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_221, add_128);  primals_221 = add_128 = None
    copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_222, add_130);  primals_222 = add_130 = None
    copy__74: "f32[256]" = torch.ops.aten.copy_.default(primals_223, add_131);  primals_223 = add_131 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_224, add_133);  primals_224 = add_133 = None
    copy__76: "f32[256]" = torch.ops.aten.copy_.default(primals_225, add_135);  primals_225 = add_135 = None
    copy__77: "f32[256]" = torch.ops.aten.copy_.default(primals_226, add_136);  primals_226 = add_136 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_227, add_138);  primals_227 = add_138 = None
    copy__79: "f32[1024]" = torch.ops.aten.copy_.default(primals_228, add_140);  primals_228 = add_140 = None
    copy__80: "f32[1024]" = torch.ops.aten.copy_.default(primals_229, add_141);  primals_229 = add_141 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_230, add_144);  primals_230 = add_144 = None
    copy__82: "f32[256]" = torch.ops.aten.copy_.default(primals_231, add_146);  primals_231 = add_146 = None
    copy__83: "f32[256]" = torch.ops.aten.copy_.default(primals_232, add_147);  primals_232 = add_147 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_233, add_151);  primals_233 = add_151 = None
    copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_234, add_153);  primals_234 = add_153 = None
    copy__86: "f32[256]" = torch.ops.aten.copy_.default(primals_235, add_154);  primals_235 = add_154 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_236, add_156);  primals_236 = add_156 = None
    copy__88: "f32[1024]" = torch.ops.aten.copy_.default(primals_237, add_158);  primals_237 = add_158 = None
    copy__89: "f32[1024]" = torch.ops.aten.copy_.default(primals_238, add_159);  primals_238 = add_159 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_239, add_162);  primals_239 = add_162 = None
    copy__91: "f32[512]" = torch.ops.aten.copy_.default(primals_240, add_164);  primals_240 = add_164 = None
    copy__92: "f32[512]" = torch.ops.aten.copy_.default(primals_241, add_165);  primals_241 = add_165 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_242, add_169);  primals_242 = add_169 = None
    copy__94: "f32[512]" = torch.ops.aten.copy_.default(primals_243, add_171);  primals_243 = add_171 = None
    copy__95: "f32[512]" = torch.ops.aten.copy_.default(primals_244, add_172);  primals_244 = add_172 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_245, add_174);  primals_245 = add_174 = None
    copy__97: "f32[1536]" = torch.ops.aten.copy_.default(primals_246, add_176);  primals_246 = add_176 = None
    copy__98: "f32[1536]" = torch.ops.aten.copy_.default(primals_247, add_177);  primals_247 = add_177 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_248, add_179);  primals_248 = add_179 = None
    copy__100: "f32[1536]" = torch.ops.aten.copy_.default(primals_249, add_181);  primals_249 = add_181 = None
    copy__101: "f32[1536]" = torch.ops.aten.copy_.default(primals_250, add_182);  primals_250 = add_182 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_251, add_185);  primals_251 = add_185 = None
    copy__103: "f32[512]" = torch.ops.aten.copy_.default(primals_252, add_187);  primals_252 = add_187 = None
    copy__104: "f32[512]" = torch.ops.aten.copy_.default(primals_253, add_188);  primals_253 = add_188 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_254, add_192);  primals_254 = add_192 = None
    copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_255, add_194);  primals_255 = add_194 = None
    copy__107: "f32[512]" = torch.ops.aten.copy_.default(primals_256, add_195);  primals_256 = add_195 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_257, add_197);  primals_257 = add_197 = None
    copy__109: "f32[1536]" = torch.ops.aten.copy_.default(primals_258, add_199);  primals_258 = add_199 = None
    copy__110: "f32[1536]" = torch.ops.aten.copy_.default(primals_259, add_200);  primals_259 = add_200 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_260, add_203);  primals_260 = add_203 = None
    copy__112: "f32[1280]" = torch.ops.aten.copy_.default(primals_261, add_205);  primals_261 = add_205 = None
    copy__113: "f32[1280]" = torch.ops.aten.copy_.default(primals_262, add_206);  primals_262 = add_206 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_61, primals_63, primals_65, primals_69, primals_71, primals_73, primals_75, primals_79, primals_81, primals_83, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_263, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, mean, relu, convolution_6, mul_40, convolution_7, squeeze_16, convolution_8, squeeze_19, mul_55, convolution_9, squeeze_22, mul_63, convolution_10, squeeze_25, add_45, mean_1, relu_1, convolution_12, mul_72, convolution_13, squeeze_28, mul_80, convolution_14, squeeze_31, mul_88, convolution_15, squeeze_34, add_61, mean_2, relu_2, convolution_17, mul_97, convolution_18, squeeze_37, convolution_19, squeeze_40, mul_112, convolution_20, squeeze_43, mul_120, convolution_21, squeeze_46, add_82, mean_3, relu_3, convolution_23, mul_129, convolution_24, squeeze_49, mul_137, convolution_25, squeeze_52, mul_145, view_7, view_13, bmm_1, squeeze_55, mul_154, convolution_27, squeeze_58, mul_162, convolution_28, squeeze_61, mul_170, convolution_29, squeeze_64, add_116, mean_4, relu_4, convolution_31, mul_179, convolution_32, squeeze_67, convolution_33, squeeze_70, mul_194, convolution_34, squeeze_73, mul_202, convolution_35, squeeze_76, add_137, mean_5, relu_5, convolution_37, mul_211, convolution_38, squeeze_79, mul_219, convolution_39, squeeze_82, mul_227, view_31, view_37, bmm_3, squeeze_85, mul_236, convolution_41, squeeze_88, mul_244, convolution_42, squeeze_91, mul_252, view_55, view_61, view_71, avg_pool2d, squeeze_94, mul_261, convolution_44, squeeze_97, convolution_45, squeeze_100, mul_276, convolution_46, squeeze_103, mul_284, view_79, view_85, bmm_7, squeeze_106, mul_293, convolution_48, squeeze_109, mul_301, convolution_49, squeeze_112, view_96, permute_33, mul_311, unsqueeze_154, mul_323, unsqueeze_166, mul_335, unsqueeze_178, permute_41, permute_42, alias_16, permute_46, permute_52, permute_54, permute_55, mul_350, unsqueeze_190, mul_362, unsqueeze_202, unsqueeze_214, mul_383, unsqueeze_226, permute_62, permute_63, alias_17, permute_67, permute_73, permute_75, permute_76, mul_398, unsqueeze_238, mul_410, unsqueeze_250, mul_422, unsqueeze_262, permute_83, permute_84, alias_18, permute_88, permute_94, permute_96, permute_97, mul_437, unsqueeze_274, mul_449, unsqueeze_286, unsqueeze_298, mul_477, unsqueeze_310, mul_489, unsqueeze_322, unsqueeze_334, unsqueeze_346, mul_526, unsqueeze_358, mul_538, unsqueeze_370, mul_550, unsqueeze_382, permute_110, permute_111, alias_27, permute_115, permute_121, permute_123, permute_124, mul_565, unsqueeze_394, mul_577, unsqueeze_406, unsqueeze_418, mul_605, unsqueeze_430, mul_617, unsqueeze_442, unsqueeze_454, unsqueeze_466, mul_654, unsqueeze_478, mul_666, unsqueeze_490, unsqueeze_502, mul_694, unsqueeze_514, mul_706, unsqueeze_526, unsqueeze_538, unsqueeze_550, mul_743, unsqueeze_562, mul_755, unsqueeze_574, mul_767, unsqueeze_586, mul_779, unsqueeze_598]
    