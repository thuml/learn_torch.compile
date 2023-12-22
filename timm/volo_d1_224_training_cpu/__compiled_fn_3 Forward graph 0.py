from __future__ import annotations



def forward(self, primals_1: "f32[1, 14, 14, 384]", primals_2: "f32[1, 1, 384]", primals_3: "f32[64, 3, 7, 7]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64, 64, 3, 3]", primals_7: "f32[64]", primals_8: "f32[64]", primals_9: "f32[64, 64, 3, 3]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[192, 64, 4, 4]", primals_13: "f32[192]", primals_14: "f32[192]", primals_15: "f32[192]", primals_16: "f32[192, 192]", primals_17: "f32[486, 192]", primals_18: "f32[486]", primals_19: "f32[192, 192]", primals_20: "f32[192]", primals_21: "f32[192]", primals_22: "f32[192]", primals_23: "f32[576, 192]", primals_24: "f32[576]", primals_25: "f32[192, 576]", primals_26: "f32[192]", primals_27: "f32[192]", primals_28: "f32[192]", primals_29: "f32[192, 192]", primals_30: "f32[486, 192]", primals_31: "f32[486]", primals_32: "f32[192, 192]", primals_33: "f32[192]", primals_34: "f32[192]", primals_35: "f32[192]", primals_36: "f32[576, 192]", primals_37: "f32[576]", primals_38: "f32[192, 576]", primals_39: "f32[192]", primals_40: "f32[192]", primals_41: "f32[192]", primals_42: "f32[192, 192]", primals_43: "f32[486, 192]", primals_44: "f32[486]", primals_45: "f32[192, 192]", primals_46: "f32[192]", primals_47: "f32[192]", primals_48: "f32[192]", primals_49: "f32[576, 192]", primals_50: "f32[576]", primals_51: "f32[192, 576]", primals_52: "f32[192]", primals_53: "f32[192]", primals_54: "f32[192]", primals_55: "f32[192, 192]", primals_56: "f32[486, 192]", primals_57: "f32[486]", primals_58: "f32[192, 192]", primals_59: "f32[192]", primals_60: "f32[192]", primals_61: "f32[192]", primals_62: "f32[576, 192]", primals_63: "f32[576]", primals_64: "f32[192, 576]", primals_65: "f32[192]", primals_66: "f32[384, 192, 2, 2]", primals_67: "f32[384]", primals_68: "f32[384]", primals_69: "f32[384]", primals_70: "f32[1152, 384]", primals_71: "f32[384, 384]", primals_72: "f32[384]", primals_73: "f32[384]", primals_74: "f32[384]", primals_75: "f32[1152, 384]", primals_76: "f32[1152]", primals_77: "f32[384, 1152]", primals_78: "f32[384]", primals_79: "f32[384]", primals_80: "f32[384]", primals_81: "f32[1152, 384]", primals_82: "f32[384, 384]", primals_83: "f32[384]", primals_84: "f32[384]", primals_85: "f32[384]", primals_86: "f32[1152, 384]", primals_87: "f32[1152]", primals_88: "f32[384, 1152]", primals_89: "f32[384]", primals_90: "f32[384]", primals_91: "f32[384]", primals_92: "f32[1152, 384]", primals_93: "f32[384, 384]", primals_94: "f32[384]", primals_95: "f32[384]", primals_96: "f32[384]", primals_97: "f32[1152, 384]", primals_98: "f32[1152]", primals_99: "f32[384, 1152]", primals_100: "f32[384]", primals_101: "f32[384]", primals_102: "f32[384]", primals_103: "f32[1152, 384]", primals_104: "f32[384, 384]", primals_105: "f32[384]", primals_106: "f32[384]", primals_107: "f32[384]", primals_108: "f32[1152, 384]", primals_109: "f32[1152]", primals_110: "f32[384, 1152]", primals_111: "f32[384]", primals_112: "f32[384]", primals_113: "f32[384]", primals_114: "f32[1152, 384]", primals_115: "f32[384, 384]", primals_116: "f32[384]", primals_117: "f32[384]", primals_118: "f32[384]", primals_119: "f32[1152, 384]", primals_120: "f32[1152]", primals_121: "f32[384, 1152]", primals_122: "f32[384]", primals_123: "f32[384]", primals_124: "f32[384]", primals_125: "f32[1152, 384]", primals_126: "f32[384, 384]", primals_127: "f32[384]", primals_128: "f32[384]", primals_129: "f32[384]", primals_130: "f32[1152, 384]", primals_131: "f32[1152]", primals_132: "f32[384, 1152]", primals_133: "f32[384]", primals_134: "f32[384]", primals_135: "f32[384]", primals_136: "f32[1152, 384]", primals_137: "f32[384, 384]", primals_138: "f32[384]", primals_139: "f32[384]", primals_140: "f32[384]", primals_141: "f32[1152, 384]", primals_142: "f32[1152]", primals_143: "f32[384, 1152]", primals_144: "f32[384]", primals_145: "f32[384]", primals_146: "f32[384]", primals_147: "f32[1152, 384]", primals_148: "f32[384, 384]", primals_149: "f32[384]", primals_150: "f32[384]", primals_151: "f32[384]", primals_152: "f32[1152, 384]", primals_153: "f32[1152]", primals_154: "f32[384, 1152]", primals_155: "f32[384]", primals_156: "f32[384]", primals_157: "f32[384]", primals_158: "f32[1152, 384]", primals_159: "f32[384, 384]", primals_160: "f32[384]", primals_161: "f32[384]", primals_162: "f32[384]", primals_163: "f32[1152, 384]", primals_164: "f32[1152]", primals_165: "f32[384, 1152]", primals_166: "f32[384]", primals_167: "f32[384]", primals_168: "f32[384]", primals_169: "f32[1152, 384]", primals_170: "f32[384, 384]", primals_171: "f32[384]", primals_172: "f32[384]", primals_173: "f32[384]", primals_174: "f32[1152, 384]", primals_175: "f32[1152]", primals_176: "f32[384, 1152]", primals_177: "f32[384]", primals_178: "f32[384]", primals_179: "f32[384]", primals_180: "f32[1152, 384]", primals_181: "f32[384, 384]", primals_182: "f32[384]", primals_183: "f32[384]", primals_184: "f32[384]", primals_185: "f32[1152, 384]", primals_186: "f32[1152]", primals_187: "f32[384, 1152]", primals_188: "f32[384]", primals_189: "f32[384]", primals_190: "f32[384]", primals_191: "f32[1152, 384]", primals_192: "f32[384, 384]", primals_193: "f32[384]", primals_194: "f32[384]", primals_195: "f32[384]", primals_196: "f32[1152, 384]", primals_197: "f32[1152]", primals_198: "f32[384, 1152]", primals_199: "f32[384]", primals_200: "f32[384]", primals_201: "f32[384]", primals_202: "f32[1152, 384]", primals_203: "f32[384, 384]", primals_204: "f32[384]", primals_205: "f32[384]", primals_206: "f32[384]", primals_207: "f32[1152, 384]", primals_208: "f32[1152]", primals_209: "f32[384, 1152]", primals_210: "f32[384]", primals_211: "f32[384]", primals_212: "f32[384]", primals_213: "f32[1152, 384]", primals_214: "f32[384, 384]", primals_215: "f32[384]", primals_216: "f32[384]", primals_217: "f32[384]", primals_218: "f32[1152, 384]", primals_219: "f32[1152]", primals_220: "f32[384, 1152]", primals_221: "f32[384]", primals_222: "f32[384]", primals_223: "f32[384]", primals_224: "f32[768, 384]", primals_225: "f32[384, 384]", primals_226: "f32[384, 384]", primals_227: "f32[384]", primals_228: "f32[384]", primals_229: "f32[384]", primals_230: "f32[1152, 384]", primals_231: "f32[1152]", primals_232: "f32[384, 1152]", primals_233: "f32[384]", primals_234: "f32[384]", primals_235: "f32[384]", primals_236: "f32[768, 384]", primals_237: "f32[384, 384]", primals_238: "f32[384, 384]", primals_239: "f32[384]", primals_240: "f32[384]", primals_241: "f32[384]", primals_242: "f32[1152, 384]", primals_243: "f32[1152]", primals_244: "f32[384, 1152]", primals_245: "f32[384]", primals_246: "f32[384]", primals_247: "f32[384]", primals_248: "f32[1000, 384]", primals_249: "f32[1000]", primals_250: "f32[1000, 384]", primals_251: "f32[1000]", primals_252: "f32[64]", primals_253: "f32[64]", primals_254: "i64[]", primals_255: "f32[64]", primals_256: "f32[64]", primals_257: "i64[]", primals_258: "f32[64]", primals_259: "f32[64]", primals_260: "i64[]", primals_261: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_261, primals_3, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_254, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 64, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1);  primals_5 = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    convolution_1: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_6, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_257, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    relu_1: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_2: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1);  primals_11 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    relu_2: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:358, code: x = self.proj(x)  # B, C, H, W
    convolution_3: "f32[8, 192, 28, 28]" = torch.ops.aten.convolution.default(relu_2, primals_12, primals_13, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:695, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
    permute: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone, [3], correction = 0, keepdim = True)
    getitem_6: "f32[8, 28, 28, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 28, 28, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_3: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone, getitem_7);  clone = getitem_7 = None
    mul_21: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_22: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_21, primals_14)
    add_16: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_22, primals_15);  mul_22 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_1: "f32[192, 192]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    view: "f32[6272, 192]" = torch.ops.aten.view.default(add_16, [6272, 192])
    mm: "f32[6272, 192]" = torch.ops.aten.mm.default(view, permute_1)
    view_1: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm, [8, 28, 28, 192]);  mm = None
    permute_2: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_1, [0, 3, 1, 2]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 2, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_12: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_13: "i64[3, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add_17: "i64[3, 14]" = torch.ops.aten.add.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None
    constant_pad_nd: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_2, [1, 1, 1, 1], 0.0);  permute_2 = None
    unsqueeze_16: "i64[3, 14, 1]" = torch.ops.aten.unsqueeze.default(add_17, -1)
    unsqueeze_17: "i64[3, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    slice_1: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 0, 0, 9223372036854775807);  constant_pad_nd = None
    slice_2: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    index: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_2, [None, None, unsqueeze_17, add_17]);  slice_2 = None
    permute_3: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    clone_1: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    view_2: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_1, [8, 1728, 196]);  clone_1 = None
    view_3: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_2, [8, 6, 32, 9, 196]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_4: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_3, [0, 1, 4, 3, 2]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_5: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
    avg_pool2d: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_5, [2, 2], [2, 2], [0, 0], True)
    permute_6: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d, [0, 2, 3, 1]);  avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_4: "f32[1568, 192]" = torch.ops.aten.view.default(permute_6, [1568, 192]);  permute_6 = None
    permute_7: "f32[192, 486]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_18, view_4, permute_7);  primals_18 = None
    view_5: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm, [8, 14, 14, 486]);  addmm = None
    view_6: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_5, [8, 196, 6, 9, 9]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_8: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3, 4]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_23: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_8, 0.1767766952966369);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_23, memory_format = torch.contiguous_format);  mul_23 = None
    amax: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_2, [-1], True)
    sub_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_2, amax);  clone_2 = amax = None
    exp: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_1: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_3, [8, 6, 196, 9, 9]);  clone_3 = None
    view_7: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand, [9408, 9, 9]);  expand = None
    expand_1: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_4, [8, 6, 196, 9, 32]);  permute_4 = None
    clone_4: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_8: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_4, [9408, 9, 32]);  clone_4 = None
    bmm: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_7, view_8)
    view_9: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm, [8, 6, 196, 9, 32]);  bmm = None
    permute_9: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_9, [0, 1, 4, 3, 2]);  view_9 = None
    clone_5: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_10: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_5, [8, 1728, 196]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_11: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_10, [8, 192, 3, 3, 14, 14]);  view_10 = None
    permute_10: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_11, [0, 1, 2, 4, 3, 5]);  view_11 = None
    full_default: "f32[8, 192, 30, 30]" = torch.ops.aten.full.default([8, 192, 30, 30], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_10, True);  permute_10 = None
    constant_pad_nd_1: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put, [-1, -1, -1, -1], 0.0);  _unsafe_index_put = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_11: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_1, [0, 2, 3, 1]);  constant_pad_nd_1 = None
    permute_12: "f32[192, 192]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    clone_6: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_12: "f32[6272, 192]" = torch.ops.aten.view.default(clone_6, [6272, 192]);  clone_6 = None
    mm_1: "f32[6272, 192]" = torch.ops.aten.mm.default(view_12, permute_12)
    view_13: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_1, [8, 28, 28, 192]);  mm_1 = None
    add_21: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_13, primals_20);  view_13 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_7: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_22: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(permute, clone_7);  permute = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_8: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_22, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_8, [3], correction = 0, keepdim = True)
    getitem_8: "f32[8, 28, 28, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 28, 28, 1]" = var_mean_4[1];  var_mean_4 = None
    add_23: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_5: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_8, getitem_9);  clone_8 = getitem_9 = None
    mul_24: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
    mul_25: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_24, primals_21)
    add_24: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_25, primals_22);  mul_25 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[6272, 192]" = torch.ops.aten.view.default(add_24, [6272, 192]);  add_24 = None
    permute_13: "f32[192, 576]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_1: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_24, view_14, permute_13);  primals_24 = None
    view_15: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_1, [8, 28, 28, 576])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_27: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
    erf: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_25: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_28: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_26, add_25);  mul_26 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[6272, 576]" = torch.ops.aten.view.default(clone_9, [6272, 576]);  clone_9 = None
    permute_14: "f32[576, 192]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_2: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_26, view_16, permute_14);  primals_26 = None
    view_17: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_2, [8, 28, 28, 192]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_26: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_22, clone_10);  add_22 = clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_11: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_11, [3], correction = 0, keepdim = True)
    getitem_10: "f32[8, 28, 28, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_6: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_11, getitem_11);  clone_11 = getitem_11 = None
    mul_29: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    mul_30: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_29, primals_27)
    add_28: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_30, primals_28);  mul_30 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_15: "f32[192, 192]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    view_18: "f32[6272, 192]" = torch.ops.aten.view.default(add_28, [6272, 192])
    mm_2: "f32[6272, 192]" = torch.ops.aten.mm.default(view_18, permute_15)
    view_19: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_2, [8, 28, 28, 192]);  mm_2 = None
    permute_16: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_19, [0, 3, 1, 2]);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd_2: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_16, [1, 1, 1, 1], 0.0);  permute_16 = None
    slice_3: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, 0, 0, 9223372036854775807);  constant_pad_nd_2 = None
    slice_4: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);  slice_3 = None
    index_1: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_4, [None, None, unsqueeze_17, add_17]);  slice_4 = None
    permute_17: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
    clone_12: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_20: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_12, [8, 1728, 196]);  clone_12 = None
    view_21: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_20, [8, 6, 32, 9, 196]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_18: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_21, [0, 1, 4, 3, 2]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_19: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_28, [0, 3, 1, 2]);  add_28 = None
    avg_pool2d_1: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_19, [2, 2], [2, 2], [0, 0], True)
    permute_20: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_1, [0, 2, 3, 1]);  avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_22: "f32[1568, 192]" = torch.ops.aten.view.default(permute_20, [1568, 192]);  permute_20 = None
    permute_21: "f32[192, 486]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_3: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_31, view_22, permute_21);  primals_31 = None
    view_23: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_3, [8, 14, 14, 486]);  addmm_3 = None
    view_24: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_23, [8, 196, 6, 9, 9]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_22: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3, 4]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_31: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_22, 0.1767766952966369);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_13: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_31, memory_format = torch.contiguous_format);  mul_31 = None
    amax_1: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_13, [-1], True)
    sub_7: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_13, amax_1);  clone_13 = amax_1 = None
    exp_1: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_14: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_14, [8, 6, 196, 9, 9]);  clone_14 = None
    view_25: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_2, [9408, 9, 9]);  expand_2 = None
    expand_3: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_18, [8, 6, 196, 9, 32]);  permute_18 = None
    clone_15: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_26: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_15, [9408, 9, 32]);  clone_15 = None
    bmm_1: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_1, [8, 6, 196, 9, 32]);  bmm_1 = None
    permute_23: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_27, [0, 1, 4, 3, 2]);  view_27 = None
    clone_16: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_28: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_16, [8, 1728, 196]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_29: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_28, [8, 192, 3, 3, 14, 14]);  view_28 = None
    permute_24: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_29, [0, 1, 2, 4, 3, 5]);  view_29 = None
    _unsafe_index_put_1: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_24, True);  permute_24 = None
    constant_pad_nd_3: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_1, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_25: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_3, [0, 2, 3, 1]);  constant_pad_nd_3 = None
    permute_26: "f32[192, 192]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    clone_17: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    view_30: "f32[6272, 192]" = torch.ops.aten.view.default(clone_17, [6272, 192]);  clone_17 = None
    mm_3: "f32[6272, 192]" = torch.ops.aten.mm.default(view_30, permute_26)
    view_31: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_3, [8, 28, 28, 192]);  mm_3 = None
    add_33: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_31, primals_33);  view_31 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_18: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_33);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_34: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_26, clone_18);  add_26 = clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_19: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_34, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_19, [3], correction = 0, keepdim = True)
    getitem_12: "f32[8, 28, 28, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
    add_35: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_8: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_19, getitem_13);  clone_19 = getitem_13 = None
    mul_32: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
    mul_33: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_32, primals_34)
    add_36: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_33, primals_35);  mul_33 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_32: "f32[6272, 192]" = torch.ops.aten.view.default(add_36, [6272, 192]);  add_36 = None
    permute_27: "f32[192, 576]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_4: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_37, view_32, permute_27);  primals_37 = None
    view_33: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_4, [8, 28, 28, 576])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_34: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_35: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf_1: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_37: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_36: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_34, add_37);  mul_34 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_34: "f32[6272, 576]" = torch.ops.aten.view.default(clone_20, [6272, 576]);  clone_20 = None
    permute_28: "f32[576, 192]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_5: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_39, view_34, permute_28);  primals_39 = None
    view_35: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_5, [8, 28, 28, 192]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_35);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_38: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_34, clone_21);  add_34 = clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_22: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_38, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_22, [3], correction = 0, keepdim = True)
    getitem_14: "f32[8, 28, 28, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 28, 28, 1]" = var_mean_7[1];  var_mean_7 = None
    add_39: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_9: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_22, getitem_15);  clone_22 = getitem_15 = None
    mul_37: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
    mul_38: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_37, primals_40)
    add_40: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_38, primals_41);  mul_38 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_29: "f32[192, 192]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_36: "f32[6272, 192]" = torch.ops.aten.view.default(add_40, [6272, 192])
    mm_4: "f32[6272, 192]" = torch.ops.aten.mm.default(view_36, permute_29)
    view_37: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_4, [8, 28, 28, 192]);  mm_4 = None
    permute_30: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_37, [0, 3, 1, 2]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd_4: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_30, [1, 1, 1, 1], 0.0);  permute_30 = None
    slice_5: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_4, 0, 0, 9223372036854775807);  constant_pad_nd_4 = None
    slice_6: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    index_2: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_6, [None, None, unsqueeze_17, add_17]);  slice_6 = None
    permute_31: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
    clone_23: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    view_38: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_23, [8, 1728, 196]);  clone_23 = None
    view_39: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_38, [8, 6, 32, 9, 196]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_32: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_39, [0, 1, 4, 3, 2]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_33: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_40, [0, 3, 1, 2]);  add_40 = None
    avg_pool2d_2: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_33, [2, 2], [2, 2], [0, 0], True)
    permute_34: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_2, [0, 2, 3, 1]);  avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_40: "f32[1568, 192]" = torch.ops.aten.view.default(permute_34, [1568, 192]);  permute_34 = None
    permute_35: "f32[192, 486]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_6: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_44, view_40, permute_35);  primals_44 = None
    view_41: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_6, [8, 14, 14, 486]);  addmm_6 = None
    view_42: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_41, [8, 196, 6, 9, 9]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_36: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3, 4]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_39: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_36, 0.1767766952966369);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_24: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_39, memory_format = torch.contiguous_format);  mul_39 = None
    amax_2: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_24, [-1], True)
    sub_10: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_24, amax_2);  clone_24 = amax_2 = None
    exp_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_3: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_5: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_25: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_4: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_25, [8, 6, 196, 9, 9]);  clone_25 = None
    view_43: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_4, [9408, 9, 9]);  expand_4 = None
    expand_5: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_32, [8, 6, 196, 9, 32]);  permute_32 = None
    clone_26: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_44: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_26, [9408, 9, 32]);  clone_26 = None
    bmm_2: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_2, [8, 6, 196, 9, 32]);  bmm_2 = None
    permute_37: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_45, [0, 1, 4, 3, 2]);  view_45 = None
    clone_27: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_46: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_27, [8, 1728, 196]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_47: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_46, [8, 192, 3, 3, 14, 14]);  view_46 = None
    permute_38: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_47, [0, 1, 2, 4, 3, 5]);  view_47 = None
    _unsafe_index_put_2: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_38, True);  permute_38 = None
    constant_pad_nd_5: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_2, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_39: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_5, [0, 2, 3, 1]);  constant_pad_nd_5 = None
    permute_40: "f32[192, 192]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_28: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    view_48: "f32[6272, 192]" = torch.ops.aten.view.default(clone_28, [6272, 192]);  clone_28 = None
    mm_5: "f32[6272, 192]" = torch.ops.aten.mm.default(view_48, permute_40)
    view_49: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_5, [8, 28, 28, 192]);  mm_5 = None
    add_45: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_49, primals_46);  view_49 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_29: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_46: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_38, clone_29);  add_38 = clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_30: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_46, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_30, [3], correction = 0, keepdim = True)
    getitem_16: "f32[8, 28, 28, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
    add_47: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_11: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_30, getitem_17);  clone_30 = getitem_17 = None
    mul_40: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
    mul_41: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_40, primals_47)
    add_48: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_41, primals_48);  mul_41 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[6272, 192]" = torch.ops.aten.view.default(add_48, [6272, 192]);  add_48 = None
    permute_41: "f32[192, 576]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_7: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_50, view_50, permute_41);  primals_50 = None
    view_51: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_7, [8, 28, 28, 576])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_42: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_43: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_2: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_49: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_44: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_42, add_49);  mul_42 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_52: "f32[6272, 576]" = torch.ops.aten.view.default(clone_31, [6272, 576]);  clone_31 = None
    permute_42: "f32[576, 192]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_8: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_52, view_52, permute_42);  primals_52 = None
    view_53: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_8, [8, 28, 28, 192]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_53);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_46, clone_32);  add_46 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_33: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_33, [3], correction = 0, keepdim = True)
    getitem_18: "f32[8, 28, 28, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 28, 28, 1]" = var_mean_9[1];  var_mean_9 = None
    add_51: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_12: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_33, getitem_19);  clone_33 = getitem_19 = None
    mul_45: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = None
    mul_46: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_45, primals_53)
    add_52: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_46, primals_54);  mul_46 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_43: "f32[192, 192]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_54: "f32[6272, 192]" = torch.ops.aten.view.default(add_52, [6272, 192])
    mm_6: "f32[6272, 192]" = torch.ops.aten.mm.default(view_54, permute_43)
    view_55: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_6, [8, 28, 28, 192]);  mm_6 = None
    permute_44: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(view_55, [0, 3, 1, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    constant_pad_nd_6: "f32[8, 192, 30, 30]" = torch.ops.aten.constant_pad_nd.default(permute_44, [1, 1, 1, 1], 0.0);  permute_44 = None
    slice_7: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(constant_pad_nd_6, 0, 0, 9223372036854775807);  constant_pad_nd_6 = None
    slice_8: "f32[8, 192, 30, 30]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    index_3: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.index.Tensor(slice_8, [None, None, unsqueeze_17, add_17]);  slice_8 = None
    permute_45: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
    clone_34: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_56: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_34, [8, 1728, 196]);  clone_34 = None
    view_57: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.view.default(view_56, [8, 6, 32, 9, 196]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    permute_46: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.permute.default(view_57, [0, 1, 4, 3, 2]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_47: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_52, [0, 3, 1, 2]);  add_52 = None
    avg_pool2d_3: "f32[8, 192, 14, 14]" = torch.ops.aten.avg_pool2d.default(permute_47, [2, 2], [2, 2], [0, 0], True)
    permute_48: "f32[8, 14, 14, 192]" = torch.ops.aten.permute.default(avg_pool2d_3, [0, 2, 3, 1]);  avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    view_58: "f32[1568, 192]" = torch.ops.aten.view.default(permute_48, [1568, 192]);  permute_48 = None
    permute_49: "f32[192, 486]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_9: "f32[1568, 486]" = torch.ops.aten.addmm.default(primals_57, view_58, permute_49);  primals_57 = None
    view_59: "f32[8, 14, 14, 486]" = torch.ops.aten.view.default(addmm_9, [8, 14, 14, 486]);  addmm_9 = None
    view_60: "f32[8, 196, 6, 9, 9]" = torch.ops.aten.view.default(view_59, [8, 196, 6, 9, 9]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    permute_50: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3, 4]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    mul_47: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.mul.Tensor(permute_50, 0.1767766952966369);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    clone_35: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(mul_47, memory_format = torch.contiguous_format);  mul_47 = None
    amax_3: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.amax.default(clone_35, [-1], True)
    sub_13: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.sub.Tensor(clone_35, amax_3);  clone_35 = amax_3 = None
    exp_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_4: "f32[8, 6, 196, 9, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    clone_36: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    expand_6: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.expand.default(clone_36, [8, 6, 196, 9, 9]);  clone_36 = None
    view_61: "f32[9408, 9, 9]" = torch.ops.aten.view.default(expand_6, [9408, 9, 9]);  expand_6 = None
    expand_7: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.expand.default(permute_46, [8, 6, 196, 9, 32]);  permute_46 = None
    clone_37: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_62: "f32[9408, 9, 32]" = torch.ops.aten.view.default(clone_37, [9408, 9, 32]);  clone_37 = None
    bmm_3: "f32[9408, 9, 32]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[8, 6, 196, 9, 32]" = torch.ops.aten.view.default(bmm_3, [8, 6, 196, 9, 32]);  bmm_3 = None
    permute_51: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.permute.default(view_63, [0, 1, 4, 3, 2]);  view_63 = None
    clone_38: "f32[8, 6, 32, 9, 196]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_64: "f32[8, 1728, 196]" = torch.ops.aten.view.default(clone_38, [8, 1728, 196]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    view_65: "f32[8, 192, 3, 3, 14, 14]" = torch.ops.aten.view.default(view_64, [8, 192, 3, 3, 14, 14]);  view_64 = None
    permute_52: "f32[8, 192, 3, 14, 3, 14]" = torch.ops.aten.permute.default(view_65, [0, 1, 2, 4, 3, 5]);  view_65 = None
    _unsafe_index_put_3: "f32[8, 192, 30, 30]" = torch.ops.aten._unsafe_index_put.default(full_default, [None, None, unsqueeze_17, add_17], permute_52, True);  permute_52 = None
    constant_pad_nd_7: "f32[8, 192, 28, 28]" = torch.ops.aten.constant_pad_nd.default(_unsafe_index_put_3, [-1, -1, -1, -1], 0.0);  _unsafe_index_put_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_53: "f32[8, 28, 28, 192]" = torch.ops.aten.permute.default(constant_pad_nd_7, [0, 2, 3, 1]);  constant_pad_nd_7 = None
    permute_54: "f32[192, 192]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    clone_39: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    view_66: "f32[6272, 192]" = torch.ops.aten.view.default(clone_39, [6272, 192]);  clone_39 = None
    mm_7: "f32[6272, 192]" = torch.ops.aten.mm.default(view_66, permute_54)
    view_67: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(mm_7, [8, 28, 28, 192]);  mm_7 = None
    add_57: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(view_67, primals_59);  view_67 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    clone_40: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_58: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_50, clone_40);  add_50 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_41: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(add_58, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_41, [3], correction = 0, keepdim = True)
    getitem_20: "f32[8, 28, 28, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 28, 28, 1]" = var_mean_10[1];  var_mean_10 = None
    add_59: "f32[8, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[8, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_14: "f32[8, 28, 28, 192]" = torch.ops.aten.sub.Tensor(clone_41, getitem_21);  clone_41 = getitem_21 = None
    mul_48: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
    mul_49: "f32[8, 28, 28, 192]" = torch.ops.aten.mul.Tensor(mul_48, primals_60)
    add_60: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(mul_49, primals_61);  mul_49 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_68: "f32[6272, 192]" = torch.ops.aten.view.default(add_60, [6272, 192]);  add_60 = None
    permute_55: "f32[192, 576]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_10: "f32[6272, 576]" = torch.ops.aten.addmm.default(primals_63, view_68, permute_55);  primals_63 = None
    view_69: "f32[8, 28, 28, 576]" = torch.ops.aten.view.default(addmm_10, [8, 28, 28, 576])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_51: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
    erf_3: "f32[8, 28, 28, 576]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_61: "f32[8, 28, 28, 576]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_52: "f32[8, 28, 28, 576]" = torch.ops.aten.mul.Tensor(mul_50, add_61);  mul_50 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_42: "f32[8, 28, 28, 576]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_70: "f32[6272, 576]" = torch.ops.aten.view.default(clone_42, [6272, 576]);  clone_42 = None
    permute_56: "f32[576, 192]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_11: "f32[6272, 192]" = torch.ops.aten.addmm.default(primals_65, view_70, permute_56);  primals_65 = None
    view_71: "f32[8, 28, 28, 192]" = torch.ops.aten.view.default(addmm_11, [8, 28, 28, 192]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_43: "f32[8, 28, 28, 192]" = torch.ops.aten.clone.default(view_71);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_62: "f32[8, 28, 28, 192]" = torch.ops.aten.add.Tensor(add_58, clone_43);  add_58 = clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:371, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 192, 28, 28]" = torch.ops.aten.permute.default(add_62, [0, 3, 1, 2]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:372, code: x = self.proj(x)  # B, C, H, W
    convolution_4: "f32[8, 384, 14, 14]" = torch.ops.aten.convolution.default(permute_57, primals_66, primals_67, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:373, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 14, 14, 384]" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:620, code: x = x + self.pos_embed
    add_63: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(permute_58, primals_1);  permute_58 = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:621, code: x = self.pos_drop(x)
    clone_44: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_45: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(clone_44, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_45, [3], correction = 0, keepdim = True)
    getitem_22: "f32[8, 14, 14, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
    add_64: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_15: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_45, getitem_23);  clone_45 = getitem_23 = None
    mul_53: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
    mul_54: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_53, primals_68)
    add_65: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_54, primals_69);  mul_54 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_59: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_72: "f32[1568, 384]" = torch.ops.aten.view.default(add_65, [1568, 384]);  add_65 = None
    mm_8: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_72, permute_59)
    view_73: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_8, [8, 14, 14, 1152]);  mm_8 = None
    view_74: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_73, [8, 196, 3, 12, 32]);  view_73 = None
    permute_60: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_74, [2, 0, 3, 1, 4]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_60);  permute_60 = None
    getitem_24: "f32[8, 12, 196, 32]" = unbind[0]
    getitem_25: "f32[8, 12, 196, 32]" = unbind[1]
    getitem_26: "f32[8, 12, 196, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_61: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
    expand_8: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_24, [8, 12, 196, 32]);  getitem_24 = None
    clone_46: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_75: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_46, [96, 196, 32]);  clone_46 = None
    expand_9: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_61, [8, 12, 32, 196]);  permute_61 = None
    clone_47: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_76: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_47, [96, 32, 196]);  clone_47 = None
    bmm_4: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_4, [8, 12, 196, 196]);  bmm_4 = None
    mul_55: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_77, 0.1767766952966369);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_55, [-1], True)
    sub_16: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_55, amax_4);  mul_55 = amax_4 = None
    exp_4: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_5: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_7: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_48: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_10: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_48, [8, 12, 196, 196]);  clone_48 = None
    view_78: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_10, [96, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_26, [8, 12, 196, 32]);  getitem_26 = None
    clone_49: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_79: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_49, [96, 196, 32]);  clone_49 = None
    bmm_5: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_5, [8, 12, 196, 32]);  bmm_5 = None
    permute_62: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_50: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_81: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_50, [8, 14, 14, 384]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_82: "f32[1568, 384]" = torch.ops.aten.view.default(view_81, [1568, 384]);  view_81 = None
    permute_63: "f32[384, 384]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_12: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_72, view_82, permute_63);  primals_72 = None
    view_83: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_12, [8, 14, 14, 384]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_51: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_66: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(clone_44, clone_51);  clone_44 = clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_52: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_66, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_52, [3], correction = 0, keepdim = True)
    getitem_27: "f32[8, 14, 14, 1]" = var_mean_12[0]
    getitem_28: "f32[8, 14, 14, 1]" = var_mean_12[1];  var_mean_12 = None
    add_67: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_27, 1e-05);  getitem_27 = None
    rsqrt_12: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_17: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_52, getitem_28);  clone_52 = getitem_28 = None
    mul_56: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
    mul_57: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_56, primals_73)
    add_68: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_57, primals_74);  mul_57 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_84: "f32[1568, 384]" = torch.ops.aten.view.default(add_68, [1568, 384]);  add_68 = None
    permute_64: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_13: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_76, view_84, permute_64);  primals_76 = None
    view_85: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_13, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_58: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_59: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_4: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
    add_69: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_60: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_58, add_69);  mul_58 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_53, [1568, 1152]);  clone_53 = None
    permute_65: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_14: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_78, view_86, permute_65);  primals_78 = None
    view_87: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_14, [8, 14, 14, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_70: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_66, clone_54);  add_66 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_55: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_70, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_55, [3], correction = 0, keepdim = True)
    getitem_29: "f32[8, 14, 14, 1]" = var_mean_13[0]
    getitem_30: "f32[8, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
    add_71: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_29, 1e-05);  getitem_29 = None
    rsqrt_13: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_18: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_55, getitem_30);  clone_55 = getitem_30 = None
    mul_61: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
    mul_62: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_61, primals_79)
    add_72: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_62, primals_80);  mul_62 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_66: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_88: "f32[1568, 384]" = torch.ops.aten.view.default(add_72, [1568, 384]);  add_72 = None
    mm_9: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_88, permute_66)
    view_89: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_9, [8, 14, 14, 1152]);  mm_9 = None
    view_90: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_89, [8, 196, 3, 12, 32]);  view_89 = None
    permute_67: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_90, [2, 0, 3, 1, 4]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_31: "f32[8, 12, 196, 32]" = unbind_1[0]
    getitem_32: "f32[8, 12, 196, 32]" = unbind_1[1]
    getitem_33: "f32[8, 12, 196, 32]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_68: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_32, [0, 1, 3, 2]);  getitem_32 = None
    expand_12: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_31, [8, 12, 196, 32]);  getitem_31 = None
    clone_56: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_91: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_56, [96, 196, 32]);  clone_56 = None
    expand_13: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_68, [8, 12, 32, 196]);  permute_68 = None
    clone_57: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_92: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_57, [96, 32, 196]);  clone_57 = None
    bmm_6: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_91, view_92)
    view_93: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_6, [8, 12, 196, 196]);  bmm_6 = None
    mul_63: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_93, 0.1767766952966369);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_63, [-1], True)
    sub_19: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_63, amax_5);  mul_63 = amax_5 = None
    exp_5: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_6: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_8: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_58: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_14: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_58, [8, 12, 196, 196]);  clone_58 = None
    view_94: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_14, [96, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_33, [8, 12, 196, 32]);  getitem_33 = None
    clone_59: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_95: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_59, [96, 196, 32]);  clone_59 = None
    bmm_7: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_94, view_95)
    view_96: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_7, [8, 12, 196, 32]);  bmm_7 = None
    permute_69: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    clone_60: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_97: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_60, [8, 14, 14, 384]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_98: "f32[1568, 384]" = torch.ops.aten.view.default(view_97, [1568, 384]);  view_97 = None
    permute_70: "f32[384, 384]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_15: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_83, view_98, permute_70);  primals_83 = None
    view_99: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_15, [8, 14, 14, 384]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_61: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_73: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_70, clone_61);  add_70 = clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_62: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_62, [3], correction = 0, keepdim = True)
    getitem_34: "f32[8, 14, 14, 1]" = var_mean_14[0]
    getitem_35: "f32[8, 14, 14, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_14: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_20: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_62, getitem_35);  clone_62 = getitem_35 = None
    mul_64: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
    mul_65: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_64, primals_84)
    add_75: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_65, primals_85);  mul_65 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_100: "f32[1568, 384]" = torch.ops.aten.view.default(add_75, [1568, 384]);  add_75 = None
    permute_71: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_16: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_87, view_100, permute_71);  primals_87 = None
    view_101: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_16, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_66: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
    mul_67: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476);  view_101 = None
    erf_5: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_76: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_68: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_66, add_76);  mul_66 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_102: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_63, [1568, 1152]);  clone_63 = None
    permute_72: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_17: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_89, view_102, permute_72);  primals_89 = None
    view_103: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_17, [8, 14, 14, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_103);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_77: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_73, clone_64);  add_73 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_65: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_65, [3], correction = 0, keepdim = True)
    getitem_36: "f32[8, 14, 14, 1]" = var_mean_15[0]
    getitem_37: "f32[8, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_15: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_21: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_65, getitem_37);  clone_65 = getitem_37 = None
    mul_69: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
    mul_70: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_69, primals_90)
    add_79: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_70, primals_91);  mul_70 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_73: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    view_104: "f32[1568, 384]" = torch.ops.aten.view.default(add_79, [1568, 384]);  add_79 = None
    mm_10: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_104, permute_73)
    view_105: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_10, [8, 14, 14, 1152]);  mm_10 = None
    view_106: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_105, [8, 196, 3, 12, 32]);  view_105 = None
    permute_74: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_106, [2, 0, 3, 1, 4]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_38: "f32[8, 12, 196, 32]" = unbind_2[0]
    getitem_39: "f32[8, 12, 196, 32]" = unbind_2[1]
    getitem_40: "f32[8, 12, 196, 32]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_75: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_39, [0, 1, 3, 2]);  getitem_39 = None
    expand_16: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_38, [8, 12, 196, 32]);  getitem_38 = None
    clone_66: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_107: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_66, [96, 196, 32]);  clone_66 = None
    expand_17: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_75, [8, 12, 32, 196]);  permute_75 = None
    clone_67: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_108: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_67, [96, 32, 196]);  clone_67 = None
    bmm_8: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_107, view_108)
    view_109: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_8, [8, 12, 196, 196]);  bmm_8 = None
    mul_71: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_109, 0.1767766952966369);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_71, [-1], True)
    sub_22: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_71, amax_6);  mul_71 = amax_6 = None
    exp_6: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_7: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_9: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_68: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_18: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_68, [8, 12, 196, 196]);  clone_68 = None
    view_110: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_18, [96, 196, 196]);  expand_18 = None
    expand_19: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_40, [8, 12, 196, 32]);  getitem_40 = None
    clone_69: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_111: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_69, [96, 196, 32]);  clone_69 = None
    bmm_9: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_110, view_111)
    view_112: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_9, [8, 12, 196, 32]);  bmm_9 = None
    permute_76: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    clone_70: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    view_113: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_70, [8, 14, 14, 384]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_114: "f32[1568, 384]" = torch.ops.aten.view.default(view_113, [1568, 384]);  view_113 = None
    permute_77: "f32[384, 384]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_18: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_94, view_114, permute_77);  primals_94 = None
    view_115: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_18, [8, 14, 14, 384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_71: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_115);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_80: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_77, clone_71);  add_77 = clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_72: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_80, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_72, [3], correction = 0, keepdim = True)
    getitem_41: "f32[8, 14, 14, 1]" = var_mean_16[0]
    getitem_42: "f32[8, 14, 14, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
    rsqrt_16: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_72, getitem_42);  clone_72 = getitem_42 = None
    mul_72: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
    mul_73: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_72, primals_95)
    add_82: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_73, primals_96);  mul_73 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[1568, 384]" = torch.ops.aten.view.default(add_82, [1568, 384]);  add_82 = None
    permute_78: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_19: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_98, view_116, permute_78);  primals_98 = None
    view_117: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_19, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
    mul_75: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476);  view_117 = None
    erf_6: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_76: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_74, add_83);  mul_74 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_73: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_73, [1568, 1152]);  clone_73 = None
    permute_79: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_20: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_100, view_118, permute_79);  primals_100 = None
    view_119: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_20, [8, 14, 14, 384]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_74: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_119);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_84: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_80, clone_74);  add_80 = clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_75: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_84, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_75, [3], correction = 0, keepdim = True)
    getitem_43: "f32[8, 14, 14, 1]" = var_mean_17[0]
    getitem_44: "f32[8, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
    add_85: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_43, 1e-05);  getitem_43 = None
    rsqrt_17: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_24: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_75, getitem_44);  clone_75 = getitem_44 = None
    mul_77: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
    mul_78: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_77, primals_101)
    add_86: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_78, primals_102);  mul_78 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_80: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    view_120: "f32[1568, 384]" = torch.ops.aten.view.default(add_86, [1568, 384]);  add_86 = None
    mm_11: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_120, permute_80)
    view_121: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_11, [8, 14, 14, 1152]);  mm_11 = None
    view_122: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_121, [8, 196, 3, 12, 32]);  view_121 = None
    permute_81: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_122, [2, 0, 3, 1, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_81);  permute_81 = None
    getitem_45: "f32[8, 12, 196, 32]" = unbind_3[0]
    getitem_46: "f32[8, 12, 196, 32]" = unbind_3[1]
    getitem_47: "f32[8, 12, 196, 32]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_82: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_46, [0, 1, 3, 2]);  getitem_46 = None
    expand_20: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_45, [8, 12, 196, 32]);  getitem_45 = None
    clone_76: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_123: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_76, [96, 196, 32]);  clone_76 = None
    expand_21: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_82, [8, 12, 32, 196]);  permute_82 = None
    clone_77: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_124: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_77, [96, 32, 196]);  clone_77 = None
    bmm_10: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_123, view_124)
    view_125: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_10, [8, 12, 196, 196]);  bmm_10 = None
    mul_79: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_125, 0.1767766952966369);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_79, [-1], True)
    sub_25: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_79, amax_7);  mul_79 = amax_7 = None
    exp_7: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_8: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_10: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_78: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_22: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_78, [8, 12, 196, 196]);  clone_78 = None
    view_126: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_22, [96, 196, 196]);  expand_22 = None
    expand_23: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_47, [8, 12, 196, 32]);  getitem_47 = None
    clone_79: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_127: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_79, [96, 196, 32]);  clone_79 = None
    bmm_11: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_126, view_127)
    view_128: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_11, [8, 12, 196, 32]);  bmm_11 = None
    permute_83: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_80: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    view_129: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_80, [8, 14, 14, 384]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_130: "f32[1568, 384]" = torch.ops.aten.view.default(view_129, [1568, 384]);  view_129 = None
    permute_84: "f32[384, 384]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_21: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_105, view_130, permute_84);  primals_105 = None
    view_131: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_21, [8, 14, 14, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_81: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_131);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_87: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_84, clone_81);  add_84 = clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_82: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_87, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_82, [3], correction = 0, keepdim = True)
    getitem_48: "f32[8, 14, 14, 1]" = var_mean_18[0]
    getitem_49: "f32[8, 14, 14, 1]" = var_mean_18[1];  var_mean_18 = None
    add_88: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_18: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_26: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_82, getitem_49);  clone_82 = getitem_49 = None
    mul_80: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
    mul_81: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_80, primals_106)
    add_89: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_81, primals_107);  mul_81 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_132: "f32[1568, 384]" = torch.ops.aten.view.default(add_89, [1568, 384]);  add_89 = None
    permute_85: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_22: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_109, view_132, permute_85);  primals_109 = None
    view_133: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_22, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.5)
    mul_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476);  view_133 = None
    erf_7: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_90: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_84: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_82, add_90);  mul_82 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_134: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_83, [1568, 1152]);  clone_83 = None
    permute_86: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_23: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_111, view_134, permute_86);  primals_111 = None
    view_135: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_23, [8, 14, 14, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_135);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_91: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_87, clone_84);  add_87 = clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_85: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_91, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_85, [3], correction = 0, keepdim = True)
    getitem_50: "f32[8, 14, 14, 1]" = var_mean_19[0]
    getitem_51: "f32[8, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
    add_92: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_19: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_27: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_85, getitem_51);  clone_85 = getitem_51 = None
    mul_85: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
    mul_86: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_85, primals_112)
    add_93: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_86, primals_113);  mul_86 = primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_87: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_136: "f32[1568, 384]" = torch.ops.aten.view.default(add_93, [1568, 384]);  add_93 = None
    mm_12: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_136, permute_87)
    view_137: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_12, [8, 14, 14, 1152]);  mm_12 = None
    view_138: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_137, [8, 196, 3, 12, 32]);  view_137 = None
    permute_88: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_138, [2, 0, 3, 1, 4]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_88);  permute_88 = None
    getitem_52: "f32[8, 12, 196, 32]" = unbind_4[0]
    getitem_53: "f32[8, 12, 196, 32]" = unbind_4[1]
    getitem_54: "f32[8, 12, 196, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_89: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_53, [0, 1, 3, 2]);  getitem_53 = None
    expand_24: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_52, [8, 12, 196, 32]);  getitem_52 = None
    clone_86: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_139: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_86, [96, 196, 32]);  clone_86 = None
    expand_25: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_89, [8, 12, 32, 196]);  permute_89 = None
    clone_87: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_140: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_87, [96, 32, 196]);  clone_87 = None
    bmm_12: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_139, view_140)
    view_141: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_12, [8, 12, 196, 196]);  bmm_12 = None
    mul_87: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_141, 0.1767766952966369);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_87, [-1], True)
    sub_28: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_87, amax_8);  mul_87 = amax_8 = None
    exp_8: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_9: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_11: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_88: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_26: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_88, [8, 12, 196, 196]);  clone_88 = None
    view_142: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_26, [96, 196, 196]);  expand_26 = None
    expand_27: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_54, [8, 12, 196, 32]);  getitem_54 = None
    clone_89: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_143: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_89, [96, 196, 32]);  clone_89 = None
    bmm_13: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_142, view_143)
    view_144: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_13, [8, 12, 196, 32]);  bmm_13 = None
    permute_90: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
    clone_90: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_145: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_90, [8, 14, 14, 384]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_146: "f32[1568, 384]" = torch.ops.aten.view.default(view_145, [1568, 384]);  view_145 = None
    permute_91: "f32[384, 384]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_24: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_116, view_146, permute_91);  primals_116 = None
    view_147: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_24, [8, 14, 14, 384]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_91: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_147);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_94: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_91, clone_91);  add_91 = clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_92: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_92, [3], correction = 0, keepdim = True)
    getitem_55: "f32[8, 14, 14, 1]" = var_mean_20[0]
    getitem_56: "f32[8, 14, 14, 1]" = var_mean_20[1];  var_mean_20 = None
    add_95: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
    rsqrt_20: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_29: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_92, getitem_56);  clone_92 = getitem_56 = None
    mul_88: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
    mul_89: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_88, primals_117)
    add_96: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_89, primals_118);  mul_89 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_148: "f32[1568, 384]" = torch.ops.aten.view.default(add_96, [1568, 384]);  add_96 = None
    permute_92: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_25: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_120, view_148, permute_92);  primals_120 = None
    view_149: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_25, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_90: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
    mul_91: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_149, 0.7071067811865476);  view_149 = None
    erf_8: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_97: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_92: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_90, add_97);  mul_90 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_93: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_150: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_93, [1568, 1152]);  clone_93 = None
    permute_93: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_26: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_122, view_150, permute_93);  primals_122 = None
    view_151: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_26, [8, 14, 14, 384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_94: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_98: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_94, clone_94);  add_94 = clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_95: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_98, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_95, [3], correction = 0, keepdim = True)
    getitem_57: "f32[8, 14, 14, 1]" = var_mean_21[0]
    getitem_58: "f32[8, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
    add_99: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05);  getitem_57 = None
    rsqrt_21: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_30: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_95, getitem_58);  clone_95 = getitem_58 = None
    mul_93: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = None
    mul_94: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_93, primals_123)
    add_100: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_94, primals_124);  mul_94 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_94: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    view_152: "f32[1568, 384]" = torch.ops.aten.view.default(add_100, [1568, 384]);  add_100 = None
    mm_13: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_152, permute_94)
    view_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_13, [8, 14, 14, 1152]);  mm_13 = None
    view_154: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_153, [8, 196, 3, 12, 32]);  view_153 = None
    permute_95: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_154, [2, 0, 3, 1, 4]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
    getitem_59: "f32[8, 12, 196, 32]" = unbind_5[0]
    getitem_60: "f32[8, 12, 196, 32]" = unbind_5[1]
    getitem_61: "f32[8, 12, 196, 32]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_96: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_60, [0, 1, 3, 2]);  getitem_60 = None
    expand_28: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_59, [8, 12, 196, 32]);  getitem_59 = None
    clone_96: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_155: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_96, [96, 196, 32]);  clone_96 = None
    expand_29: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_96, [8, 12, 32, 196]);  permute_96 = None
    clone_97: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_156: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_97, [96, 32, 196]);  clone_97 = None
    bmm_14: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_155, view_156)
    view_157: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_14, [8, 12, 196, 196]);  bmm_14 = None
    mul_95: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_157, 0.1767766952966369);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_95, [-1], True)
    sub_31: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_95, amax_9);  mul_95 = amax_9 = None
    exp_9: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_10: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_12: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_98: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_30: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_98, [8, 12, 196, 196]);  clone_98 = None
    view_158: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_30, [96, 196, 196]);  expand_30 = None
    expand_31: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_61, [8, 12, 196, 32]);  getitem_61 = None
    clone_99: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_159: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_99, [96, 196, 32]);  clone_99 = None
    bmm_15: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_158, view_159)
    view_160: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_15, [8, 12, 196, 32]);  bmm_15 = None
    permute_97: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_100: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    view_161: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_100, [8, 14, 14, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_162: "f32[1568, 384]" = torch.ops.aten.view.default(view_161, [1568, 384]);  view_161 = None
    permute_98: "f32[384, 384]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_27: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_127, view_162, permute_98);  primals_127 = None
    view_163: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_27, [8, 14, 14, 384]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_101: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_101: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_98, clone_101);  add_98 = clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_102: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_101, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_102, [3], correction = 0, keepdim = True)
    getitem_62: "f32[8, 14, 14, 1]" = var_mean_22[0]
    getitem_63: "f32[8, 14, 14, 1]" = var_mean_22[1];  var_mean_22 = None
    add_102: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_22: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_32: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_102, getitem_63);  clone_102 = getitem_63 = None
    mul_96: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
    mul_97: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_96, primals_128)
    add_103: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_97, primals_129);  mul_97 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_164: "f32[1568, 384]" = torch.ops.aten.view.default(add_103, [1568, 384]);  add_103 = None
    permute_99: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_28: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_131, view_164, permute_99);  primals_131 = None
    view_165: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_28, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_98: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
    mul_99: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_165, 0.7071067811865476);  view_165 = None
    erf_9: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_104: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_100: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_98, add_104);  mul_98 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_103: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_100);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_166: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_103, [1568, 1152]);  clone_103 = None
    permute_100: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_29: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_133, view_166, permute_100);  primals_133 = None
    view_167: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_29, [8, 14, 14, 384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_104: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_167);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_105: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_101, clone_104);  add_101 = clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_105: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_105, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_105, [3], correction = 0, keepdim = True)
    getitem_64: "f32[8, 14, 14, 1]" = var_mean_23[0]
    getitem_65: "f32[8, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
    add_106: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_23: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_33: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_105, getitem_65);  clone_105 = getitem_65 = None
    mul_101: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = None
    mul_102: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_101, primals_134)
    add_107: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_102, primals_135);  mul_102 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_101: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    view_168: "f32[1568, 384]" = torch.ops.aten.view.default(add_107, [1568, 384]);  add_107 = None
    mm_14: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_168, permute_101)
    view_169: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_14, [8, 14, 14, 1152]);  mm_14 = None
    view_170: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_169, [8, 196, 3, 12, 32]);  view_169 = None
    permute_102: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_170, [2, 0, 3, 1, 4]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_102);  permute_102 = None
    getitem_66: "f32[8, 12, 196, 32]" = unbind_6[0]
    getitem_67: "f32[8, 12, 196, 32]" = unbind_6[1]
    getitem_68: "f32[8, 12, 196, 32]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_103: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_67, [0, 1, 3, 2]);  getitem_67 = None
    expand_32: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_66, [8, 12, 196, 32]);  getitem_66 = None
    clone_106: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_171: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_106, [96, 196, 32]);  clone_106 = None
    expand_33: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_103, [8, 12, 32, 196]);  permute_103 = None
    clone_107: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_172: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_107, [96, 32, 196]);  clone_107 = None
    bmm_16: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_171, view_172)
    view_173: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_16, [8, 12, 196, 196]);  bmm_16 = None
    mul_103: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_173, 0.1767766952966369);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_103, [-1], True)
    sub_34: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_103, amax_10);  mul_103 = amax_10 = None
    exp_10: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_11: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_13: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_108: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_34: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_108, [8, 12, 196, 196]);  clone_108 = None
    view_174: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_34, [96, 196, 196]);  expand_34 = None
    expand_35: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_68, [8, 12, 196, 32]);  getitem_68 = None
    clone_109: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_175: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_109, [96, 196, 32]);  clone_109 = None
    bmm_17: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_174, view_175)
    view_176: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_17, [8, 12, 196, 32]);  bmm_17 = None
    permute_104: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    clone_110: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_177: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_110, [8, 14, 14, 384]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_178: "f32[1568, 384]" = torch.ops.aten.view.default(view_177, [1568, 384]);  view_177 = None
    permute_105: "f32[384, 384]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_30: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_138, view_178, permute_105);  primals_138 = None
    view_179: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_30, [8, 14, 14, 384]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_111: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_108: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_105, clone_111);  add_105 = clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_112: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_112, [3], correction = 0, keepdim = True)
    getitem_69: "f32[8, 14, 14, 1]" = var_mean_24[0]
    getitem_70: "f32[8, 14, 14, 1]" = var_mean_24[1];  var_mean_24 = None
    add_109: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
    rsqrt_24: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_35: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_112, getitem_70);  clone_112 = getitem_70 = None
    mul_104: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
    mul_105: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_104, primals_139)
    add_110: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_105, primals_140);  mul_105 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[1568, 384]" = torch.ops.aten.view.default(add_110, [1568, 384]);  add_110 = None
    permute_106: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_31: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_142, view_180, permute_106);  primals_142 = None
    view_181: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_31, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_107: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_10: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_111: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_108: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_106, add_111);  mul_106 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_113: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_113, [1568, 1152]);  clone_113 = None
    permute_107: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_32: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_144, view_182, permute_107);  primals_144 = None
    view_183: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_32, [8, 14, 14, 384]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_114: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_112: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_108, clone_114);  add_108 = clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_115: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_115, [3], correction = 0, keepdim = True)
    getitem_71: "f32[8, 14, 14, 1]" = var_mean_25[0]
    getitem_72: "f32[8, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
    add_113: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05);  getitem_71 = None
    rsqrt_25: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_36: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_115, getitem_72);  clone_115 = getitem_72 = None
    mul_109: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = None
    mul_110: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_109, primals_145)
    add_114: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_110, primals_146);  mul_110 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_108: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_184: "f32[1568, 384]" = torch.ops.aten.view.default(add_114, [1568, 384]);  add_114 = None
    mm_15: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_184, permute_108)
    view_185: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_15, [8, 14, 14, 1152]);  mm_15 = None
    view_186: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_185, [8, 196, 3, 12, 32]);  view_185 = None
    permute_109: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_109);  permute_109 = None
    getitem_73: "f32[8, 12, 196, 32]" = unbind_7[0]
    getitem_74: "f32[8, 12, 196, 32]" = unbind_7[1]
    getitem_75: "f32[8, 12, 196, 32]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_110: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_74, [0, 1, 3, 2]);  getitem_74 = None
    expand_36: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_73, [8, 12, 196, 32]);  getitem_73 = None
    clone_116: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_187: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_116, [96, 196, 32]);  clone_116 = None
    expand_37: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_110, [8, 12, 32, 196]);  permute_110 = None
    clone_117: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_188: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_117, [96, 32, 196]);  clone_117 = None
    bmm_18: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_18, [8, 12, 196, 196]);  bmm_18 = None
    mul_111: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_189, 0.1767766952966369);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_111, [-1], True)
    sub_37: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_111, amax_11);  mul_111 = amax_11 = None
    exp_11: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_12: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_14: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_118: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_38: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_118, [8, 12, 196, 196]);  clone_118 = None
    view_190: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_38, [96, 196, 196]);  expand_38 = None
    expand_39: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_75, [8, 12, 196, 32]);  getitem_75 = None
    clone_119: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_191: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_119, [96, 196, 32]);  clone_119 = None
    bmm_19: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_19, [8, 12, 196, 32]);  bmm_19 = None
    permute_111: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_120: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_193: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_120, [8, 14, 14, 384]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_194: "f32[1568, 384]" = torch.ops.aten.view.default(view_193, [1568, 384]);  view_193 = None
    permute_112: "f32[384, 384]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_33: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_149, view_194, permute_112);  primals_149 = None
    view_195: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_33, [8, 14, 14, 384]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_121: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_115: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_112, clone_121);  add_112 = clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_122: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_115, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_122, [3], correction = 0, keepdim = True)
    getitem_76: "f32[8, 14, 14, 1]" = var_mean_26[0]
    getitem_77: "f32[8, 14, 14, 1]" = var_mean_26[1];  var_mean_26 = None
    add_116: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_26: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_38: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_122, getitem_77);  clone_122 = getitem_77 = None
    mul_112: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
    mul_113: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_112, primals_150)
    add_117: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_113, primals_151);  mul_113 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_196: "f32[1568, 384]" = torch.ops.aten.view.default(add_117, [1568, 384]);  add_117 = None
    permute_113: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_34: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_153, view_196, permute_113);  primals_153 = None
    view_197: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_34, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_114: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_115: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_11: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_118: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_116: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_114, add_118);  mul_114 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_123: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_123, [1568, 1152]);  clone_123 = None
    permute_114: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_35: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_155, view_198, permute_114);  primals_155 = None
    view_199: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_35, [8, 14, 14, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_124: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_119: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_115, clone_124);  add_115 = clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_125: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_119, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_125, [3], correction = 0, keepdim = True)
    getitem_78: "f32[8, 14, 14, 1]" = var_mean_27[0]
    getitem_79: "f32[8, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
    add_120: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_27: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_39: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_125, getitem_79);  clone_125 = getitem_79 = None
    mul_117: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = None
    mul_118: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_117, primals_156)
    add_121: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_118, primals_157);  mul_118 = primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_115: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    view_200: "f32[1568, 384]" = torch.ops.aten.view.default(add_121, [1568, 384]);  add_121 = None
    mm_16: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_200, permute_115)
    view_201: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_16, [8, 14, 14, 1152]);  mm_16 = None
    view_202: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_201, [8, 196, 3, 12, 32]);  view_201 = None
    permute_116: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_202, [2, 0, 3, 1, 4]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_116);  permute_116 = None
    getitem_80: "f32[8, 12, 196, 32]" = unbind_8[0]
    getitem_81: "f32[8, 12, 196, 32]" = unbind_8[1]
    getitem_82: "f32[8, 12, 196, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_117: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_81, [0, 1, 3, 2]);  getitem_81 = None
    expand_40: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_80, [8, 12, 196, 32]);  getitem_80 = None
    clone_126: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_203: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_126, [96, 196, 32]);  clone_126 = None
    expand_41: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_117, [8, 12, 32, 196]);  permute_117 = None
    clone_127: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_204: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_127, [96, 32, 196]);  clone_127 = None
    bmm_20: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_203, view_204)
    view_205: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_20, [8, 12, 196, 196]);  bmm_20 = None
    mul_119: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_205, 0.1767766952966369);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_119, [-1], True)
    sub_40: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_119, amax_12);  mul_119 = amax_12 = None
    exp_12: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_13: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_15: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_128: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_42: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_128, [8, 12, 196, 196]);  clone_128 = None
    view_206: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_42, [96, 196, 196]);  expand_42 = None
    expand_43: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_82, [8, 12, 196, 32]);  getitem_82 = None
    clone_129: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_207: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_129, [96, 196, 32]);  clone_129 = None
    bmm_21: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_206, view_207)
    view_208: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_21, [8, 12, 196, 32]);  bmm_21 = None
    permute_118: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_130: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_209: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_130, [8, 14, 14, 384]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_210: "f32[1568, 384]" = torch.ops.aten.view.default(view_209, [1568, 384]);  view_209 = None
    permute_119: "f32[384, 384]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_36: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_160, view_210, permute_119);  primals_160 = None
    view_211: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_36, [8, 14, 14, 384]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_131: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_211);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_122: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_119, clone_131);  add_119 = clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_132: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_132, [3], correction = 0, keepdim = True)
    getitem_83: "f32[8, 14, 14, 1]" = var_mean_28[0]
    getitem_84: "f32[8, 14, 14, 1]" = var_mean_28[1];  var_mean_28 = None
    add_123: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
    rsqrt_28: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_41: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_132, getitem_84);  clone_132 = getitem_84 = None
    mul_120: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
    mul_121: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_120, primals_161)
    add_124: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_121, primals_162);  mul_121 = primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_212: "f32[1568, 384]" = torch.ops.aten.view.default(add_124, [1568, 384]);  add_124 = None
    permute_120: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_37: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_164, view_212, permute_120);  primals_164 = None
    view_213: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_37, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_122: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_123: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
    erf_12: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_125: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_124: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_122, add_125);  mul_122 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_133: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_124);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_214: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_133, [1568, 1152]);  clone_133 = None
    permute_121: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_38: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_166, view_214, permute_121);  primals_166 = None
    view_215: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_38, [8, 14, 14, 384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_134: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_215);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_126: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_122, clone_134);  add_122 = clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_135: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_126, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_135, [3], correction = 0, keepdim = True)
    getitem_85: "f32[8, 14, 14, 1]" = var_mean_29[0]
    getitem_86: "f32[8, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
    add_127: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_85, 1e-05);  getitem_85 = None
    rsqrt_29: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_42: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_135, getitem_86);  clone_135 = getitem_86 = None
    mul_125: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = None
    mul_126: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_125, primals_167)
    add_128: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_126, primals_168);  mul_126 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_122: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    view_216: "f32[1568, 384]" = torch.ops.aten.view.default(add_128, [1568, 384]);  add_128 = None
    mm_17: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_216, permute_122)
    view_217: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_17, [8, 14, 14, 1152]);  mm_17 = None
    view_218: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_217, [8, 196, 3, 12, 32]);  view_217 = None
    permute_123: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_218, [2, 0, 3, 1, 4]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_123);  permute_123 = None
    getitem_87: "f32[8, 12, 196, 32]" = unbind_9[0]
    getitem_88: "f32[8, 12, 196, 32]" = unbind_9[1]
    getitem_89: "f32[8, 12, 196, 32]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_124: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_88, [0, 1, 3, 2]);  getitem_88 = None
    expand_44: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_87, [8, 12, 196, 32]);  getitem_87 = None
    clone_136: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_219: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_136, [96, 196, 32]);  clone_136 = None
    expand_45: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_124, [8, 12, 32, 196]);  permute_124 = None
    clone_137: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_220: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_137, [96, 32, 196]);  clone_137 = None
    bmm_22: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_219, view_220)
    view_221: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_22, [8, 12, 196, 196]);  bmm_22 = None
    mul_127: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_221, 0.1767766952966369);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_127, [-1], True)
    sub_43: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_127, amax_13);  mul_127 = amax_13 = None
    exp_13: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_14: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_16: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_138: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_46: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_138, [8, 12, 196, 196]);  clone_138 = None
    view_222: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_46, [96, 196, 196]);  expand_46 = None
    expand_47: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_89, [8, 12, 196, 32]);  getitem_89 = None
    clone_139: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_223: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_139, [96, 196, 32]);  clone_139 = None
    bmm_23: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_222, view_223)
    view_224: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_23, [8, 12, 196, 32]);  bmm_23 = None
    permute_125: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    clone_140: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    view_225: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_140, [8, 14, 14, 384]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_226: "f32[1568, 384]" = torch.ops.aten.view.default(view_225, [1568, 384]);  view_225 = None
    permute_126: "f32[384, 384]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_39: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_171, view_226, permute_126);  primals_171 = None
    view_227: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_39, [8, 14, 14, 384]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_141: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_227);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_129: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_126, clone_141);  add_126 = clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_142: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_142, [3], correction = 0, keepdim = True)
    getitem_90: "f32[8, 14, 14, 1]" = var_mean_30[0]
    getitem_91: "f32[8, 14, 14, 1]" = var_mean_30[1];  var_mean_30 = None
    add_130: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_30: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_44: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_142, getitem_91);  clone_142 = getitem_91 = None
    mul_128: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
    mul_129: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_128, primals_172)
    add_131: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_129, primals_173);  mul_129 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_228: "f32[1568, 384]" = torch.ops.aten.view.default(add_131, [1568, 384]);  add_131 = None
    permute_127: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_40: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_175, view_228, permute_127);  primals_175 = None
    view_229: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_40, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_130: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.5)
    mul_131: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_229, 0.7071067811865476);  view_229 = None
    erf_13: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_132: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_132: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_130, add_132);  mul_130 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_143: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_230: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_143, [1568, 1152]);  clone_143 = None
    permute_128: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_41: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_177, view_230, permute_128);  primals_177 = None
    view_231: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_41, [8, 14, 14, 384]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_144: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_231);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_133: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_129, clone_144);  add_129 = clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_145: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_145, [3], correction = 0, keepdim = True)
    getitem_92: "f32[8, 14, 14, 1]" = var_mean_31[0]
    getitem_93: "f32[8, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
    add_134: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_31: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_45: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_145, getitem_93);  clone_145 = getitem_93 = None
    mul_133: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = None
    mul_134: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_133, primals_178)
    add_135: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_134, primals_179);  mul_134 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_129: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    view_232: "f32[1568, 384]" = torch.ops.aten.view.default(add_135, [1568, 384]);  add_135 = None
    mm_18: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_232, permute_129)
    view_233: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_18, [8, 14, 14, 1152]);  mm_18 = None
    view_234: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_233, [8, 196, 3, 12, 32]);  view_233 = None
    permute_130: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_234, [2, 0, 3, 1, 4]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_130);  permute_130 = None
    getitem_94: "f32[8, 12, 196, 32]" = unbind_10[0]
    getitem_95: "f32[8, 12, 196, 32]" = unbind_10[1]
    getitem_96: "f32[8, 12, 196, 32]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_131: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_95, [0, 1, 3, 2]);  getitem_95 = None
    expand_48: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_94, [8, 12, 196, 32]);  getitem_94 = None
    clone_146: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_235: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_146, [96, 196, 32]);  clone_146 = None
    expand_49: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_131, [8, 12, 32, 196]);  permute_131 = None
    clone_147: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_236: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_147, [96, 32, 196]);  clone_147 = None
    bmm_24: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_235, view_236)
    view_237: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_24, [8, 12, 196, 196]);  bmm_24 = None
    mul_135: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_237, 0.1767766952966369);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_135, [-1], True)
    sub_46: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_135, amax_14);  mul_135 = amax_14 = None
    exp_14: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_15: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_17: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_148: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_50: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_148, [8, 12, 196, 196]);  clone_148 = None
    view_238: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_50, [96, 196, 196]);  expand_50 = None
    expand_51: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_96, [8, 12, 196, 32]);  getitem_96 = None
    clone_149: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_239: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_149, [96, 196, 32]);  clone_149 = None
    bmm_25: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_238, view_239)
    view_240: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_25, [8, 12, 196, 32]);  bmm_25 = None
    permute_132: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    clone_150: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_241: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_150, [8, 14, 14, 384]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_242: "f32[1568, 384]" = torch.ops.aten.view.default(view_241, [1568, 384]);  view_241 = None
    permute_133: "f32[384, 384]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_42: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_182, view_242, permute_133);  primals_182 = None
    view_243: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_42, [8, 14, 14, 384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_151: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_243);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_136: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_133, clone_151);  add_133 = clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_152: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_136, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_152, [3], correction = 0, keepdim = True)
    getitem_97: "f32[8, 14, 14, 1]" = var_mean_32[0]
    getitem_98: "f32[8, 14, 14, 1]" = var_mean_32[1];  var_mean_32 = None
    add_137: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
    rsqrt_32: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_47: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_152, getitem_98);  clone_152 = getitem_98 = None
    mul_136: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
    mul_137: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_136, primals_183)
    add_138: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_137, primals_184);  mul_137 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_244: "f32[1568, 384]" = torch.ops.aten.view.default(add_138, [1568, 384]);  add_138 = None
    permute_134: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_43: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_186, view_244, permute_134);  primals_186 = None
    view_245: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_43, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_138: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.5)
    mul_139: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_245, 0.7071067811865476);  view_245 = None
    erf_14: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_139: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_140: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_138, add_139);  mul_138 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_153, [1568, 1152]);  clone_153 = None
    permute_135: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_44: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_188, view_246, permute_135);  primals_188 = None
    view_247: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_44, [8, 14, 14, 384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_154: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_247);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_140: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_136, clone_154);  add_136 = clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_155: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_155, [3], correction = 0, keepdim = True)
    getitem_99: "f32[8, 14, 14, 1]" = var_mean_33[0]
    getitem_100: "f32[8, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
    add_141: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
    rsqrt_33: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_48: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_155, getitem_100);  clone_155 = getitem_100 = None
    mul_141: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = None
    mul_142: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_141, primals_189)
    add_142: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_142, primals_190);  mul_142 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_136: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    view_248: "f32[1568, 384]" = torch.ops.aten.view.default(add_142, [1568, 384]);  add_142 = None
    mm_19: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_248, permute_136)
    view_249: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_19, [8, 14, 14, 1152]);  mm_19 = None
    view_250: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_249, [8, 196, 3, 12, 32]);  view_249 = None
    permute_137: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_250, [2, 0, 3, 1, 4]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
    getitem_101: "f32[8, 12, 196, 32]" = unbind_11[0]
    getitem_102: "f32[8, 12, 196, 32]" = unbind_11[1]
    getitem_103: "f32[8, 12, 196, 32]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_138: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_102, [0, 1, 3, 2]);  getitem_102 = None
    expand_52: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_101, [8, 12, 196, 32]);  getitem_101 = None
    clone_156: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_251: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_156, [96, 196, 32]);  clone_156 = None
    expand_53: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_138, [8, 12, 32, 196]);  permute_138 = None
    clone_157: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_252: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_157, [96, 32, 196]);  clone_157 = None
    bmm_26: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_26, [8, 12, 196, 196]);  bmm_26 = None
    mul_143: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_253, 0.1767766952966369);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_143, [-1], True)
    sub_49: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_143, amax_15);  mul_143 = amax_15 = None
    exp_15: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_16: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_18: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_158: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_54: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_158, [8, 12, 196, 196]);  clone_158 = None
    view_254: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_54, [96, 196, 196]);  expand_54 = None
    expand_55: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_103, [8, 12, 196, 32]);  getitem_103 = None
    clone_159: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_255: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_159, [96, 196, 32]);  clone_159 = None
    bmm_27: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_27, [8, 12, 196, 32]);  bmm_27 = None
    permute_139: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_160: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_257: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_160, [8, 14, 14, 384]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_258: "f32[1568, 384]" = torch.ops.aten.view.default(view_257, [1568, 384]);  view_257 = None
    permute_140: "f32[384, 384]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_45: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_193, view_258, permute_140);  primals_193 = None
    view_259: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_45, [8, 14, 14, 384]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_161: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_259);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_143: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_140, clone_161);  add_140 = clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_162: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_143, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_162, [3], correction = 0, keepdim = True)
    getitem_104: "f32[8, 14, 14, 1]" = var_mean_34[0]
    getitem_105: "f32[8, 14, 14, 1]" = var_mean_34[1];  var_mean_34 = None
    add_144: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_34: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_50: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_162, getitem_105);  clone_162 = getitem_105 = None
    mul_144: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
    mul_145: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_144, primals_194)
    add_145: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_145, primals_195);  mul_145 = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_260: "f32[1568, 384]" = torch.ops.aten.view.default(add_145, [1568, 384]);  add_145 = None
    permute_141: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_46: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_197, view_260, permute_141);  primals_197 = None
    view_261: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_46, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_146: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_147: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_15: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_146: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_148: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_146, add_146);  mul_146 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_163: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_262: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_163, [1568, 1152]);  clone_163 = None
    permute_142: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_47: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_199, view_262, permute_142);  primals_199 = None
    view_263: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_47, [8, 14, 14, 384]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_164: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_263);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_147: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_143, clone_164);  add_143 = clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_165: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_147, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_165, [3], correction = 0, keepdim = True)
    getitem_106: "f32[8, 14, 14, 1]" = var_mean_35[0]
    getitem_107: "f32[8, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
    add_148: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_35: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_51: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_165, getitem_107);  clone_165 = getitem_107 = None
    mul_149: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = None
    mul_150: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_149, primals_200)
    add_149: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_150, primals_201);  mul_150 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_143: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    view_264: "f32[1568, 384]" = torch.ops.aten.view.default(add_149, [1568, 384]);  add_149 = None
    mm_20: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_264, permute_143)
    view_265: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_20, [8, 14, 14, 1152]);  mm_20 = None
    view_266: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_265, [8, 196, 3, 12, 32]);  view_265 = None
    permute_144: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_266, [2, 0, 3, 1, 4]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_144);  permute_144 = None
    getitem_108: "f32[8, 12, 196, 32]" = unbind_12[0]
    getitem_109: "f32[8, 12, 196, 32]" = unbind_12[1]
    getitem_110: "f32[8, 12, 196, 32]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_145: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_109, [0, 1, 3, 2]);  getitem_109 = None
    expand_56: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_108, [8, 12, 196, 32]);  getitem_108 = None
    clone_166: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_267: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_166, [96, 196, 32]);  clone_166 = None
    expand_57: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_145, [8, 12, 32, 196]);  permute_145 = None
    clone_167: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_268: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_167, [96, 32, 196]);  clone_167 = None
    bmm_28: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_267, view_268)
    view_269: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_28, [8, 12, 196, 196]);  bmm_28 = None
    mul_151: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_269, 0.1767766952966369);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_151, [-1], True)
    sub_52: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_151, amax_16);  mul_151 = amax_16 = None
    exp_16: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_17: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_19: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_168: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_58: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_168, [8, 12, 196, 196]);  clone_168 = None
    view_270: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_58, [96, 196, 196]);  expand_58 = None
    expand_59: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_110, [8, 12, 196, 32]);  getitem_110 = None
    clone_169: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_271: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_169, [96, 196, 32]);  clone_169 = None
    bmm_29: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_270, view_271)
    view_272: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_29, [8, 12, 196, 32]);  bmm_29 = None
    permute_146: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_170: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_273: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_170, [8, 14, 14, 384]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_274: "f32[1568, 384]" = torch.ops.aten.view.default(view_273, [1568, 384]);  view_273 = None
    permute_147: "f32[384, 384]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_48: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_204, view_274, permute_147);  primals_204 = None
    view_275: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_48, [8, 14, 14, 384]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_171: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_275);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_150: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_147, clone_171);  add_147 = clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_172: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_172, [3], correction = 0, keepdim = True)
    getitem_111: "f32[8, 14, 14, 1]" = var_mean_36[0]
    getitem_112: "f32[8, 14, 14, 1]" = var_mean_36[1];  var_mean_36 = None
    add_151: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
    rsqrt_36: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_53: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_172, getitem_112);  clone_172 = getitem_112 = None
    mul_152: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
    mul_153: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_152, primals_205)
    add_152: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_153, primals_206);  mul_153 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[1568, 384]" = torch.ops.aten.view.default(add_152, [1568, 384]);  add_152 = None
    permute_148: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_49: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_208, view_276, permute_148);  primals_208 = None
    view_277: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_49, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
    mul_155: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_277, 0.7071067811865476);  view_277 = None
    erf_16: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_153: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_156: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_154, add_153);  mul_154 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_173: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_278: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_173, [1568, 1152]);  clone_173 = None
    permute_149: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_50: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_210, view_278, permute_149);  primals_210 = None
    view_279: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_50, [8, 14, 14, 384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_174: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_279);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_154: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_150, clone_174);  add_150 = clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    clone_175: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_154, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_175, [3], correction = 0, keepdim = True)
    getitem_113: "f32[8, 14, 14, 1]" = var_mean_37[0]
    getitem_114: "f32[8, 14, 14, 1]" = var_mean_37[1];  var_mean_37 = None
    add_155: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_113, 1e-05);  getitem_113 = None
    rsqrt_37: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_54: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_175, getitem_114);  clone_175 = getitem_114 = None
    mul_157: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = None
    mul_158: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_157, primals_211)
    add_156: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_158, primals_212);  mul_158 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_150: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    view_280: "f32[1568, 384]" = torch.ops.aten.view.default(add_156, [1568, 384]);  add_156 = None
    mm_21: "f32[1568, 1152]" = torch.ops.aten.mm.default(view_280, permute_150)
    view_281: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(mm_21, [8, 14, 14, 1152]);  mm_21 = None
    view_282: "f32[8, 196, 3, 12, 32]" = torch.ops.aten.view.default(view_281, [8, 196, 3, 12, 32]);  view_281 = None
    permute_151: "f32[3, 8, 12, 196, 32]" = torch.ops.aten.permute.default(view_282, [2, 0, 3, 1, 4]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_13 = torch.ops.aten.unbind.int(permute_151);  permute_151 = None
    getitem_115: "f32[8, 12, 196, 32]" = unbind_13[0]
    getitem_116: "f32[8, 12, 196, 32]" = unbind_13[1]
    getitem_117: "f32[8, 12, 196, 32]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_152: "f32[8, 12, 32, 196]" = torch.ops.aten.permute.default(getitem_116, [0, 1, 3, 2]);  getitem_116 = None
    expand_60: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_115, [8, 12, 196, 32]);  getitem_115 = None
    clone_176: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_283: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_176, [96, 196, 32]);  clone_176 = None
    expand_61: "f32[8, 12, 32, 196]" = torch.ops.aten.expand.default(permute_152, [8, 12, 32, 196]);  permute_152 = None
    clone_177: "f32[8, 12, 32, 196]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_284: "f32[96, 32, 196]" = torch.ops.aten.view.default(clone_177, [96, 32, 196]);  clone_177 = None
    bmm_30: "f32[96, 196, 196]" = torch.ops.aten.bmm.default(view_283, view_284)
    view_285: "f32[8, 12, 196, 196]" = torch.ops.aten.view.default(bmm_30, [8, 12, 196, 196]);  bmm_30 = None
    mul_159: "f32[8, 12, 196, 196]" = torch.ops.aten.mul.Tensor(view_285, 0.1767766952966369);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 12, 196, 1]" = torch.ops.aten.amax.default(mul_159, [-1], True)
    sub_55: "f32[8, 12, 196, 196]" = torch.ops.aten.sub.Tensor(mul_159, amax_17);  mul_159 = amax_17 = None
    exp_17: "f32[8, 12, 196, 196]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_18: "f32[8, 12, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 12, 196, 196]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_20: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    clone_178: "f32[8, 12, 196, 196]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    expand_62: "f32[8, 12, 196, 196]" = torch.ops.aten.expand.default(clone_178, [8, 12, 196, 196]);  clone_178 = None
    view_286: "f32[96, 196, 196]" = torch.ops.aten.view.default(expand_62, [96, 196, 196]);  expand_62 = None
    expand_63: "f32[8, 12, 196, 32]" = torch.ops.aten.expand.default(getitem_117, [8, 12, 196, 32]);  getitem_117 = None
    clone_179: "f32[8, 12, 196, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_287: "f32[96, 196, 32]" = torch.ops.aten.view.default(clone_179, [96, 196, 32]);  clone_179 = None
    bmm_31: "f32[96, 196, 32]" = torch.ops.aten.bmm.default(view_286, view_287)
    view_288: "f32[8, 12, 196, 32]" = torch.ops.aten.view.default(bmm_31, [8, 12, 196, 32]);  bmm_31 = None
    permute_153: "f32[8, 196, 12, 32]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    clone_180: "f32[8, 196, 12, 32]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_289: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(clone_180, [8, 14, 14, 384]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    view_290: "f32[1568, 384]" = torch.ops.aten.view.default(view_289, [1568, 384]);  view_289 = None
    permute_154: "f32[384, 384]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    addmm_51: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_215, view_290, permute_154);  primals_215 = None
    view_291: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_51, [8, 14, 14, 384]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    clone_181: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_291);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_157: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_154, clone_181);  add_154 = clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    clone_182: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(add_157, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_182, [3], correction = 0, keepdim = True)
    getitem_118: "f32[8, 14, 14, 1]" = var_mean_38[0]
    getitem_119: "f32[8, 14, 14, 1]" = var_mean_38[1];  var_mean_38 = None
    add_158: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_38: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_56: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(clone_182, getitem_119);  clone_182 = getitem_119 = None
    mul_160: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
    mul_161: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_160, primals_216)
    add_159: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_161, primals_217);  mul_161 = primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_292: "f32[1568, 384]" = torch.ops.aten.view.default(add_159, [1568, 384]);  add_159 = None
    permute_155: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
    addmm_52: "f32[1568, 1152]" = torch.ops.aten.addmm.default(primals_219, view_292, permute_155);  primals_219 = None
    view_293: "f32[8, 14, 14, 1152]" = torch.ops.aten.view.default(addmm_52, [8, 14, 14, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_162: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.5)
    mul_163: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(view_293, 0.7071067811865476);  view_293 = None
    erf_17: "f32[8, 14, 14, 1152]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
    add_160: "f32[8, 14, 14, 1152]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_164: "f32[8, 14, 14, 1152]" = torch.ops.aten.mul.Tensor(mul_162, add_160);  mul_162 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_183: "f32[8, 14, 14, 1152]" = torch.ops.aten.clone.default(mul_164);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_294: "f32[1568, 1152]" = torch.ops.aten.view.default(clone_183, [1568, 1152]);  clone_183 = None
    permute_156: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_53: "f32[1568, 384]" = torch.ops.aten.addmm.default(primals_221, view_294, permute_156);  primals_221 = None
    view_295: "f32[8, 14, 14, 384]" = torch.ops.aten.view.default(addmm_53, [8, 14, 14, 384]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_184: "f32[8, 14, 14, 384]" = torch.ops.aten.clone.default(view_295);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_161: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(add_157, clone_184);  add_157 = clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:628, code: x = x.reshape(B, -1, C)
    view_296: "f32[8, 196, 384]" = torch.ops.aten.view.default(add_161, [8, 196, 384]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:633, code: cls_tokens = self.cls_token.expand(B, -1, -1)
    expand_64: "f32[8, 1, 384]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:634, code: x = torch.cat([cls_tokens, x], dim=1)
    cat: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand_64, view_296], 1);  expand_64 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_9: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807)
    slice_10: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    var_mean_39 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 197, 1]" = var_mean_39[0]
    getitem_121: "f32[8, 197, 1]" = var_mean_39[1];  var_mean_39 = None
    add_162: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_39: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_57: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat, getitem_121)
    mul_165: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
    mul_166: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_165, primals_222);  mul_165 = None
    add_163: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_166, primals_223);  mul_166 = primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_157: "f32[384, 768]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
    view_297: "f32[1576, 384]" = torch.ops.aten.view.default(add_163, [1576, 384])
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_297, permute_157)
    view_298: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_22, [8, 197, 768]);  mm_22 = None
    view_299: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.view.default(view_298, [8, 197, 2, 12, 32]);  view_298 = None
    permute_158: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_299, [2, 0, 3, 1, 4]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_14 = torch.ops.aten.unbind.int(permute_158);  permute_158 = None
    getitem_122: "f32[8, 12, 197, 32]" = unbind_14[0]
    getitem_123: "f32[8, 12, 197, 32]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    slice_11: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(add_163, 0, 0, 9223372036854775807);  add_163 = None
    slice_12: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 1);  slice_11 = None
    slice_13: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_12, 2, 0, 9223372036854775807);  slice_12 = None
    permute_159: "f32[384, 384]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
    view_300: "f32[8, 384]" = torch.ops.aten.view.default(slice_13, [8, 384]);  slice_13 = None
    mm_23: "f32[8, 384]" = torch.ops.aten.mm.default(view_300, permute_159)
    view_301: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_23, [8, 1, 384]);  mm_23 = None
    view_302: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(view_301, [8, 12, 1, 32]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_167: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_302, 0.1767766952966369);  view_302 = None
    permute_160: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_122, [0, 1, 3, 2]);  getitem_122 = None
    expand_65: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_167, [8, 12, 1, 32]);  mul_167 = None
    view_303: "f32[96, 1, 32]" = torch.ops.aten.view.default(expand_65, [96, 1, 32]);  expand_65 = None
    expand_66: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_160, [8, 12, 32, 197]);  permute_160 = None
    clone_185: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_304: "f32[96, 32, 197]" = torch.ops.aten.view.default(clone_185, [96, 32, 197]);  clone_185 = None
    bmm_32: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_303, view_304)
    view_305: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_32, [8, 12, 1, 197]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_305, [-1], True)
    sub_58: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_305, amax_18);  view_305 = amax_18 = None
    exp_18: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_19: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_21: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:240, code: attn = self.attn_drop(attn)
    clone_186: "f32[8, 12, 1, 197]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    expand_67: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(clone_186, [8, 12, 1, 197]);  clone_186 = None
    view_306: "f32[96, 1, 197]" = torch.ops.aten.view.default(expand_67, [96, 1, 197]);  expand_67 = None
    expand_68: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_123, [8, 12, 197, 32]);  getitem_123 = None
    clone_187: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_307: "f32[96, 197, 32]" = torch.ops.aten.view.default(clone_187, [96, 197, 32]);  clone_187 = None
    bmm_33: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_306, view_307)
    view_308: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_33, [8, 12, 1, 32]);  bmm_33 = None
    permute_161: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    view_309: "f32[8, 1, 384]" = torch.ops.aten.view.default(permute_161, [8, 1, 384]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_310: "f32[8, 384]" = torch.ops.aten.view.default(view_309, [8, 384]);  view_309 = None
    permute_162: "f32[384, 384]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm_54: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_227, view_310, permute_162);  primals_227 = None
    view_311: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_54, [8, 1, 384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:244, code: cls_embed = self.proj_drop(cls_embed)
    clone_188: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_311);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_164: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_10, clone_188);  slice_10 = clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    var_mean_40 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 1, 1]" = var_mean_40[0]
    getitem_125: "f32[8, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_165: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_40: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_59: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_164, getitem_125);  getitem_125 = None
    mul_168: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
    mul_169: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_168, primals_228)
    add_166: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_169, primals_229);  mul_169 = primals_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_312: "f32[8, 384]" = torch.ops.aten.view.default(add_166, [8, 384]);  add_166 = None
    permute_163: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
    addmm_55: "f32[8, 1152]" = torch.ops.aten.addmm.default(primals_231, view_312, permute_163);  primals_231 = None
    view_313: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_55, [8, 1, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_170: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.5)
    mul_171: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476);  view_313 = None
    erf_18: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_167: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_172: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_170, add_167);  mul_170 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_189: "f32[8, 1, 1152]" = torch.ops.aten.clone.default(mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_314: "f32[8, 1152]" = torch.ops.aten.view.default(clone_189, [8, 1152]);  clone_189 = None
    permute_164: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
    addmm_56: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_233, view_314, permute_164);  primals_233 = None
    view_315: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_56, [8, 1, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_190: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_315);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_168: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_164, clone_190);  add_164 = clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_15: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(slice_9, 1, 1, 9223372036854775807);  slice_9 = None
    cat_1: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_168, slice_15], 1);  add_168 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    slice_16: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
    slice_17: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    var_mean_41 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 197, 1]" = var_mean_41[0]
    getitem_127: "f32[8, 197, 1]" = var_mean_41[1];  var_mean_41 = None
    add_169: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
    rsqrt_41: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_60: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_127)
    mul_173: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
    mul_174: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_173, primals_234);  mul_173 = None
    add_170: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_174, primals_235);  mul_174 = primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_165: "f32[384, 768]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
    view_316: "f32[1576, 384]" = torch.ops.aten.view.default(add_170, [1576, 384])
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_316, permute_165)
    view_317: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_24, [8, 197, 768]);  mm_24 = None
    view_318: "f32[8, 197, 2, 12, 32]" = torch.ops.aten.view.default(view_317, [8, 197, 2, 12, 32]);  view_317 = None
    permute_166: "f32[2, 8, 12, 197, 32]" = torch.ops.aten.permute.default(view_318, [2, 0, 3, 1, 4]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_15 = torch.ops.aten.unbind.int(permute_166);  permute_166 = None
    getitem_128: "f32[8, 12, 197, 32]" = unbind_15[0]
    getitem_129: "f32[8, 12, 197, 32]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    slice_18: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(add_170, 0, 0, 9223372036854775807);  add_170 = None
    slice_19: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_18, 1, 0, 1);  slice_18 = None
    slice_20: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(slice_19, 2, 0, 9223372036854775807);  slice_19 = None
    permute_167: "f32[384, 384]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    view_319: "f32[8, 384]" = torch.ops.aten.view.default(slice_20, [8, 384]);  slice_20 = None
    mm_25: "f32[8, 384]" = torch.ops.aten.mm.default(view_319, permute_167)
    view_320: "f32[8, 1, 384]" = torch.ops.aten.view.default(mm_25, [8, 1, 384]);  mm_25 = None
    view_321: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(view_320, [8, 12, 1, 32]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_175: "f32[8, 12, 1, 32]" = torch.ops.aten.mul.Tensor(view_321, 0.1767766952966369);  view_321 = None
    permute_168: "f32[8, 12, 32, 197]" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
    expand_69: "f32[8, 12, 1, 32]" = torch.ops.aten.expand.default(mul_175, [8, 12, 1, 32]);  mul_175 = None
    view_322: "f32[96, 1, 32]" = torch.ops.aten.view.default(expand_69, [96, 1, 32]);  expand_69 = None
    expand_70: "f32[8, 12, 32, 197]" = torch.ops.aten.expand.default(permute_168, [8, 12, 32, 197]);  permute_168 = None
    clone_191: "f32[8, 12, 32, 197]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_323: "f32[96, 32, 197]" = torch.ops.aten.view.default(clone_191, [96, 32, 197]);  clone_191 = None
    bmm_34: "f32[96, 1, 197]" = torch.ops.aten.bmm.default(view_322, view_323)
    view_324: "f32[8, 12, 1, 197]" = torch.ops.aten.view.default(bmm_34, [8, 12, 1, 197]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 12, 1, 1]" = torch.ops.aten.amax.default(view_324, [-1], True)
    sub_61: "f32[8, 12, 1, 197]" = torch.ops.aten.sub.Tensor(view_324, amax_19);  view_324 = amax_19 = None
    exp_19: "f32[8, 12, 1, 197]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_20: "f32[8, 12, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 12, 1, 197]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_22: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:240, code: attn = self.attn_drop(attn)
    clone_192: "f32[8, 12, 1, 197]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    expand_71: "f32[8, 12, 1, 197]" = torch.ops.aten.expand.default(clone_192, [8, 12, 1, 197]);  clone_192 = None
    view_325: "f32[96, 1, 197]" = torch.ops.aten.view.default(expand_71, [96, 1, 197]);  expand_71 = None
    expand_72: "f32[8, 12, 197, 32]" = torch.ops.aten.expand.default(getitem_129, [8, 12, 197, 32]);  getitem_129 = None
    clone_193: "f32[8, 12, 197, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_326: "f32[96, 197, 32]" = torch.ops.aten.view.default(clone_193, [96, 197, 32]);  clone_193 = None
    bmm_35: "f32[96, 1, 32]" = torch.ops.aten.bmm.default(view_325, view_326)
    view_327: "f32[8, 12, 1, 32]" = torch.ops.aten.view.default(bmm_35, [8, 12, 1, 32]);  bmm_35 = None
    permute_169: "f32[8, 1, 12, 32]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    view_328: "f32[8, 1, 384]" = torch.ops.aten.view.default(permute_169, [8, 1, 384]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    view_329: "f32[8, 384]" = torch.ops.aten.view.default(view_328, [8, 384]);  view_328 = None
    permute_170: "f32[384, 384]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
    addmm_57: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_239, view_329, permute_170);  primals_239 = None
    view_330: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_57, [8, 1, 384]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:244, code: cls_embed = self.proj_drop(cls_embed)
    clone_194: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_330);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    add_171: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(slice_17, clone_194);  slice_17 = clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    var_mean_42 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 1, 1]" = var_mean_42[0]
    getitem_131: "f32[8, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_172: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_42: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_62: "f32[8, 1, 384]" = torch.ops.aten.sub.Tensor(add_171, getitem_131);  getitem_131 = None
    mul_176: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
    mul_177: "f32[8, 1, 384]" = torch.ops.aten.mul.Tensor(mul_176, primals_240)
    add_173: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(mul_177, primals_241);  mul_177 = primals_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_331: "f32[8, 384]" = torch.ops.aten.view.default(add_173, [8, 384]);  add_173 = None
    permute_171: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_58: "f32[8, 1152]" = torch.ops.aten.addmm.default(primals_243, view_331, permute_171);  primals_243 = None
    view_332: "f32[8, 1, 1152]" = torch.ops.aten.view.default(addmm_58, [8, 1, 1152])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_178: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.5)
    mul_179: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(view_332, 0.7071067811865476);  view_332 = None
    erf_19: "f32[8, 1, 1152]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_174: "f32[8, 1, 1152]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_180: "f32[8, 1, 1152]" = torch.ops.aten.mul.Tensor(mul_178, add_174);  mul_178 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_195: "f32[8, 1, 1152]" = torch.ops.aten.clone.default(mul_180);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_333: "f32[8, 1152]" = torch.ops.aten.view.default(clone_195, [8, 1152]);  clone_195 = None
    permute_172: "f32[1152, 384]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
    addmm_59: "f32[8, 384]" = torch.ops.aten.addmm.default(primals_245, view_333, permute_172);  primals_245 = None
    view_334: "f32[8, 1, 384]" = torch.ops.aten.view.default(addmm_59, [8, 1, 384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_196: "f32[8, 1, 384]" = torch.ops.aten.clone.default(view_334);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    add_175: "f32[8, 1, 384]" = torch.ops.aten.add.Tensor(add_171, clone_196);  add_171 = clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    slice_22: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(slice_16, 1, 1, 9223372036854775807);  slice_16 = None
    cat_2: "f32[8, 197, 384]" = torch.ops.aten.cat.default([add_175, slice_22], 1);  add_175 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 197, 1]" = var_mean_43[0]
    getitem_133: "f32[8, 197, 1]" = var_mean_43[1];  var_mean_43 = None
    add_176: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_43: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_63: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_133)
    mul_181: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
    mul_182: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_181, primals_246);  mul_181 = None
    add_177: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_182, primals_247);  mul_182 = primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    slice_23: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(add_177, 0, 0, 9223372036854775807)
    select: "f32[8, 384]" = torch.ops.aten.select.int(slice_23, 1, 0);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:713, code: x = self.head_drop(x)
    clone_197: "f32[8, 197, 384]" = torch.ops.aten.clone.default(add_177);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    permute_173: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_60: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_249, select, permute_173);  primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    slice_24: "f32[8, 197, 384]" = torch.ops.aten.slice.Tensor(clone_197, 0, 0, 9223372036854775807);  clone_197 = None
    slice_25: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(slice_24, 1, 1, 9223372036854775807);  slice_24 = None
    permute_174: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
    clone_198: "f32[8, 196, 384]" = torch.ops.aten.clone.default(slice_25, memory_format = torch.contiguous_format);  slice_25 = None
    view_335: "f32[1568, 384]" = torch.ops.aten.view.default(clone_198, [1568, 384]);  clone_198 = None
    mm_26: "f32[1568, 1000]" = torch.ops.aten.mm.default(view_335, permute_174)
    view_336: "f32[8, 196, 1000]" = torch.ops.aten.view.default(mm_26, [8, 196, 1000]);  mm_26 = None
    add_178: "f32[8, 196, 1000]" = torch.ops.aten.add.Tensor(view_336, primals_251);  view_336 = primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:720, code: out = out + 0.5 * aux.max(1)[0]
    max_1 = torch.ops.aten.max.dim(add_178, 1);  add_178 = None
    getitem_134: "f32[8, 1000]" = max_1[0]
    getitem_135: "i64[8, 1000]" = max_1[1];  max_1 = None
    mul_183: "f32[8, 1000]" = torch.ops.aten.mul.Tensor(getitem_134, 0.5);  getitem_134 = None
    add_179: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm_60, mul_183);  addmm_60 = mul_183 = None
    unsqueeze_61: "i64[8, 1, 1000]" = torch.ops.aten.unsqueeze.default(getitem_135, 1);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    permute_177: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    permute_179: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_183: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_187: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    div_21: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 384);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    permute_191: "f32[384, 384]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    permute_196: "f32[96, 197, 1]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    permute_197: "f32[96, 32, 197]" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    alias_23: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    permute_198: "f32[96, 32, 1]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    permute_199: "f32[96, 197, 32]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    permute_203: "f32[384, 384]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_208: "f32[768, 384]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_210: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_214: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    div_23: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 384);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    permute_218: "f32[384, 384]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    permute_223: "f32[96, 197, 1]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
    permute_224: "f32[96, 32, 197]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 12, 1, 197]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    permute_225: "f32[96, 32, 1]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    permute_226: "f32[96, 197, 32]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    permute_230: "f32[384, 384]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_235: "f32[768, 384]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_237: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_241: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_25: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 384);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_245: "f32[384, 384]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_250: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    permute_251: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_252: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    permute_253: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_258: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_26: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 384);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_260: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_264: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_27: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 384);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_268: "f32[384, 384]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_273: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    permute_274: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_26: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_275: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    permute_276: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_281: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_28: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 384);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_283: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_287: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_29: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 384);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_291: "f32[384, 384]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_296: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    permute_297: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_298: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    permute_299: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_304: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_30: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 384);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_306: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_310: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_31: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 384);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_314: "f32[384, 384]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_319: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
    permute_320: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_28: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_321: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    permute_322: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_327: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_32: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 384);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_329: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_333: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_33: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 384);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_337: "f32[384, 384]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_342: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_222, [0, 2, 1]);  view_222 = None
    permute_343: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_29: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_344: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    permute_345: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_350: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_34: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 384);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_352: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_356: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_35: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 384);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_360: "f32[384, 384]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_365: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    permute_366: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_30: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_367: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
    permute_368: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_373: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_36: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 384);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_375: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_379: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_37: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 384);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_383: "f32[384, 384]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_388: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    permute_389: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_31: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_390: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    permute_391: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_396: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_38: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 384);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_398: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_402: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_39: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 384);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_406: "f32[384, 384]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_411: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_174, [0, 2, 1]);  view_174 = None
    permute_412: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_32: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_413: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    permute_414: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_419: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_40: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 384);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_421: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_425: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_41: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 384);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_429: "f32[384, 384]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_434: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    permute_435: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_33: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_436: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    permute_437: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_442: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_42: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 384);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_444: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_448: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_43: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 384);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_452: "f32[384, 384]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_457: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    permute_458: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_34: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_459: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    permute_460: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_465: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_44: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 384);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_467: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_471: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_45: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 384);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_475: "f32[384, 384]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_480: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
    permute_481: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_35: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_482: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    permute_483: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_488: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_46: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 384);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_490: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_494: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_47: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 384);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_498: "f32[384, 384]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_503: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    permute_504: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_36: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_505: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    permute_506: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_511: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_48: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 384);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_513: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_517: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_49: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 384);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_521: "f32[384, 384]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_526: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    permute_527: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_37: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_528: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    permute_529: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_534: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_50: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 384);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_536: "f32[384, 1152]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_540: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_51: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 384);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    permute_544: "f32[384, 384]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    permute_549: "f32[96, 196, 196]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    permute_550: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    alias_38: "f32[8, 12, 196, 196]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    permute_551: "f32[96, 32, 196]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    permute_552: "f32[96, 196, 32]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_557: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_52: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 384);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_561: "f32[192, 576]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_565: "f32[576, 192]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_53: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 192);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_571: "f32[192, 192]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    permute_576: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    permute_577: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_39: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    permute_579: "f32[486, 192]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_590: "f32[192, 192]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_54: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 192);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_592: "f32[192, 576]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_596: "f32[576, 192]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_55: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 192);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_602: "f32[192, 192]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    permute_607: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    permute_608: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_40: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    permute_610: "f32[486, 192]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_621: "f32[192, 192]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_56: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 192);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_623: "f32[192, 576]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_627: "f32[576, 192]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_57: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 192);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_633: "f32[192, 192]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    permute_638: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    permute_639: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_41: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    permute_641: "f32[486, 192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_652: "f32[192, 192]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_58: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 192);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_654: "f32[192, 576]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_658: "f32[576, 192]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    div_59: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 192);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_664: "f32[192, 192]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    permute_669: "f32[9408, 9, 9]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    permute_670: "f32[9408, 32, 9]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    alias_42: "f32[8, 6, 196, 9, 9]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    permute_672: "f32[486, 192]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    permute_683: "f32[192, 192]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    div_60: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 192);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    unsqueeze_110: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_111: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
    unsqueeze_112: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
    unsqueeze_122: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_123: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
    unsqueeze_124: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
    unsqueeze_134: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_135: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
    unsqueeze_136: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_252, add_2);  primals_252 = add_2 = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_253, add_3);  primals_253 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_254, add);  primals_254 = add = None
    copy__3: "f32[64]" = torch.ops.aten.copy_.default(primals_255, add_7);  primals_255 = add_7 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_256, add_8);  primals_256 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_257, add_5);  primals_257 = add_5 = None
    copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_258, add_12);  primals_258 = add_12 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_259, add_13);  primals_259 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_260, add_10);  primals_260 = add_10 = None
    return [add_179, primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_14, primals_21, primals_27, primals_34, primals_40, primals_47, primals_53, primals_60, primals_66, primals_68, primals_73, primals_79, primals_84, primals_90, primals_95, primals_101, primals_106, primals_112, primals_117, primals_123, primals_128, primals_134, primals_139, primals_145, primals_150, primals_156, primals_161, primals_167, primals_172, primals_178, primals_183, primals_189, primals_194, primals_200, primals_205, primals_211, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_261, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mul_21, view, add_17, unsqueeze_17, permute_5, view_4, full_default, view_12, mul_24, view_14, addmm_1, view_16, mul_29, view_18, permute_19, view_22, view_30, mul_32, view_32, addmm_4, view_34, mul_37, view_36, permute_33, view_40, view_48, mul_40, view_50, addmm_7, view_52, mul_45, view_54, permute_47, view_58, view_66, mul_48, view_68, addmm_10, view_70, permute_57, mul_53, view_72, view_82, mul_56, view_84, addmm_13, view_86, mul_61, view_88, view_98, mul_64, view_100, addmm_16, view_102, mul_69, view_104, view_114, mul_72, view_116, addmm_19, view_118, mul_77, view_120, view_130, mul_80, view_132, addmm_22, view_134, mul_85, view_136, view_146, mul_88, view_148, addmm_25, view_150, mul_93, view_152, view_162, mul_96, view_164, addmm_28, view_166, mul_101, view_168, view_178, mul_104, view_180, addmm_31, view_182, mul_109, view_184, view_194, mul_112, view_196, addmm_34, view_198, mul_117, view_200, view_210, mul_120, view_212, addmm_37, view_214, mul_125, view_216, view_226, mul_128, view_228, addmm_40, view_230, mul_133, view_232, view_242, mul_136, view_244, addmm_43, view_246, mul_141, view_248, view_258, mul_144, view_260, addmm_46, view_262, mul_149, view_264, view_274, mul_152, view_276, addmm_49, view_278, mul_157, view_280, view_290, mul_160, view_292, addmm_52, view_294, cat, getitem_121, rsqrt_39, view_297, view_300, view_310, mul_168, view_312, addmm_55, view_314, cat_1, getitem_127, rsqrt_41, view_316, view_319, view_329, mul_176, view_331, addmm_58, view_333, cat_2, getitem_133, rsqrt_43, select, view_335, unsqueeze_61, permute_177, permute_179, permute_183, permute_187, div_21, permute_191, permute_196, permute_197, alias_23, permute_198, permute_199, permute_203, permute_208, permute_210, permute_214, div_23, permute_218, permute_223, permute_224, alias_24, permute_225, permute_226, permute_230, permute_235, permute_237, permute_241, div_25, permute_245, permute_250, permute_251, alias_25, permute_252, permute_253, permute_258, div_26, permute_260, permute_264, div_27, permute_268, permute_273, permute_274, alias_26, permute_275, permute_276, permute_281, div_28, permute_283, permute_287, div_29, permute_291, permute_296, permute_297, alias_27, permute_298, permute_299, permute_304, div_30, permute_306, permute_310, div_31, permute_314, permute_319, permute_320, alias_28, permute_321, permute_322, permute_327, div_32, permute_329, permute_333, div_33, permute_337, permute_342, permute_343, alias_29, permute_344, permute_345, permute_350, div_34, permute_352, permute_356, div_35, permute_360, permute_365, permute_366, alias_30, permute_367, permute_368, permute_373, div_36, permute_375, permute_379, div_37, permute_383, permute_388, permute_389, alias_31, permute_390, permute_391, permute_396, div_38, permute_398, permute_402, div_39, permute_406, permute_411, permute_412, alias_32, permute_413, permute_414, permute_419, div_40, permute_421, permute_425, div_41, permute_429, permute_434, permute_435, alias_33, permute_436, permute_437, permute_442, div_42, permute_444, permute_448, div_43, permute_452, permute_457, permute_458, alias_34, permute_459, permute_460, permute_465, div_44, permute_467, permute_471, div_45, permute_475, permute_480, permute_481, alias_35, permute_482, permute_483, permute_488, div_46, permute_490, permute_494, div_47, permute_498, permute_503, permute_504, alias_36, permute_505, permute_506, permute_511, div_48, permute_513, permute_517, div_49, permute_521, permute_526, permute_527, alias_37, permute_528, permute_529, permute_534, div_50, permute_536, permute_540, div_51, permute_544, permute_549, permute_550, alias_38, permute_551, permute_552, permute_557, div_52, permute_561, permute_565, div_53, permute_571, permute_576, permute_577, alias_39, permute_579, permute_590, div_54, permute_592, permute_596, div_55, permute_602, permute_607, permute_608, alias_40, permute_610, permute_621, div_56, permute_623, permute_627, div_57, permute_633, permute_638, permute_639, alias_41, permute_641, permute_652, div_58, permute_654, permute_658, div_59, permute_664, permute_669, permute_670, alias_42, permute_672, permute_683, div_60, unsqueeze_112, unsqueeze_124, unsqueeze_136]
    