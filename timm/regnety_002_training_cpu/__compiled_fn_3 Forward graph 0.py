from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[24]", primals_4: "f32[24]", primals_5: "f32[24]", primals_6: "f32[24]", primals_7: "f32[24]", primals_8: "f32[24]", primals_9: "f32[24]", primals_10: "f32[24]", primals_11: "f32[56]", primals_12: "f32[56]", primals_13: "f32[56]", primals_14: "f32[56]", primals_15: "f32[56]", primals_16: "f32[56]", primals_17: "f32[56]", primals_18: "f32[56]", primals_19: "f32[152]", primals_20: "f32[152]", primals_21: "f32[152]", primals_22: "f32[152]", primals_23: "f32[152]", primals_24: "f32[152]", primals_25: "f32[152]", primals_26: "f32[152]", primals_27: "f32[152]", primals_28: "f32[152]", primals_29: "f32[152]", primals_30: "f32[152]", primals_31: "f32[152]", primals_32: "f32[152]", primals_33: "f32[152]", primals_34: "f32[152]", primals_35: "f32[152]", primals_36: "f32[152]", primals_37: "f32[152]", primals_38: "f32[152]", primals_39: "f32[152]", primals_40: "f32[152]", primals_41: "f32[152]", primals_42: "f32[152]", primals_43: "f32[152]", primals_44: "f32[152]", primals_45: "f32[368]", primals_46: "f32[368]", primals_47: "f32[368]", primals_48: "f32[368]", primals_49: "f32[368]", primals_50: "f32[368]", primals_51: "f32[368]", primals_52: "f32[368]", primals_53: "f32[368]", primals_54: "f32[368]", primals_55: "f32[368]", primals_56: "f32[368]", primals_57: "f32[368]", primals_58: "f32[368]", primals_59: "f32[368]", primals_60: "f32[368]", primals_61: "f32[368]", primals_62: "f32[368]", primals_63: "f32[368]", primals_64: "f32[368]", primals_65: "f32[368]", primals_66: "f32[368]", primals_67: "f32[368]", primals_68: "f32[368]", primals_69: "f32[368]", primals_70: "f32[368]", primals_71: "f32[368]", primals_72: "f32[368]", primals_73: "f32[368]", primals_74: "f32[368]", primals_75: "f32[368]", primals_76: "f32[368]", primals_77: "f32[368]", primals_78: "f32[368]", primals_79: "f32[368]", primals_80: "f32[368]", primals_81: "f32[368]", primals_82: "f32[368]", primals_83: "f32[368]", primals_84: "f32[368]", primals_85: "f32[368]", primals_86: "f32[368]", primals_87: "f32[368]", primals_88: "f32[368]", primals_89: "f32[32, 3, 3, 3]", primals_90: "f32[24, 32, 1, 1]", primals_91: "f32[24, 8, 3, 3]", primals_92: "f32[8, 24, 1, 1]", primals_93: "f32[8]", primals_94: "f32[24, 8, 1, 1]", primals_95: "f32[24]", primals_96: "f32[24, 24, 1, 1]", primals_97: "f32[24, 32, 1, 1]", primals_98: "f32[56, 24, 1, 1]", primals_99: "f32[56, 8, 3, 3]", primals_100: "f32[6, 56, 1, 1]", primals_101: "f32[6]", primals_102: "f32[56, 6, 1, 1]", primals_103: "f32[56]", primals_104: "f32[56, 56, 1, 1]", primals_105: "f32[56, 24, 1, 1]", primals_106: "f32[152, 56, 1, 1]", primals_107: "f32[152, 8, 3, 3]", primals_108: "f32[14, 152, 1, 1]", primals_109: "f32[14]", primals_110: "f32[152, 14, 1, 1]", primals_111: "f32[152]", primals_112: "f32[152, 152, 1, 1]", primals_113: "f32[152, 56, 1, 1]", primals_114: "f32[152, 152, 1, 1]", primals_115: "f32[152, 8, 3, 3]", primals_116: "f32[38, 152, 1, 1]", primals_117: "f32[38]", primals_118: "f32[152, 38, 1, 1]", primals_119: "f32[152]", primals_120: "f32[152, 152, 1, 1]", primals_121: "f32[152, 152, 1, 1]", primals_122: "f32[152, 8, 3, 3]", primals_123: "f32[38, 152, 1, 1]", primals_124: "f32[38]", primals_125: "f32[152, 38, 1, 1]", primals_126: "f32[152]", primals_127: "f32[152, 152, 1, 1]", primals_128: "f32[152, 152, 1, 1]", primals_129: "f32[152, 8, 3, 3]", primals_130: "f32[38, 152, 1, 1]", primals_131: "f32[38]", primals_132: "f32[152, 38, 1, 1]", primals_133: "f32[152]", primals_134: "f32[152, 152, 1, 1]", primals_135: "f32[368, 152, 1, 1]", primals_136: "f32[368, 8, 3, 3]", primals_137: "f32[38, 368, 1, 1]", primals_138: "f32[38]", primals_139: "f32[368, 38, 1, 1]", primals_140: "f32[368]", primals_141: "f32[368, 368, 1, 1]", primals_142: "f32[368, 152, 1, 1]", primals_143: "f32[368, 368, 1, 1]", primals_144: "f32[368, 8, 3, 3]", primals_145: "f32[92, 368, 1, 1]", primals_146: "f32[92]", primals_147: "f32[368, 92, 1, 1]", primals_148: "f32[368]", primals_149: "f32[368, 368, 1, 1]", primals_150: "f32[368, 368, 1, 1]", primals_151: "f32[368, 8, 3, 3]", primals_152: "f32[92, 368, 1, 1]", primals_153: "f32[92]", primals_154: "f32[368, 92, 1, 1]", primals_155: "f32[368]", primals_156: "f32[368, 368, 1, 1]", primals_157: "f32[368, 368, 1, 1]", primals_158: "f32[368, 8, 3, 3]", primals_159: "f32[92, 368, 1, 1]", primals_160: "f32[92]", primals_161: "f32[368, 92, 1, 1]", primals_162: "f32[368]", primals_163: "f32[368, 368, 1, 1]", primals_164: "f32[368, 368, 1, 1]", primals_165: "f32[368, 8, 3, 3]", primals_166: "f32[92, 368, 1, 1]", primals_167: "f32[92]", primals_168: "f32[368, 92, 1, 1]", primals_169: "f32[368]", primals_170: "f32[368, 368, 1, 1]", primals_171: "f32[368, 368, 1, 1]", primals_172: "f32[368, 8, 3, 3]", primals_173: "f32[92, 368, 1, 1]", primals_174: "f32[92]", primals_175: "f32[368, 92, 1, 1]", primals_176: "f32[368]", primals_177: "f32[368, 368, 1, 1]", primals_178: "f32[368, 368, 1, 1]", primals_179: "f32[368, 8, 3, 3]", primals_180: "f32[92, 368, 1, 1]", primals_181: "f32[92]", primals_182: "f32[368, 92, 1, 1]", primals_183: "f32[368]", primals_184: "f32[368, 368, 1, 1]", primals_185: "f32[1000, 368]", primals_186: "f32[1000]", primals_187: "i64[]", primals_188: "f32[32]", primals_189: "f32[32]", primals_190: "i64[]", primals_191: "f32[24]", primals_192: "f32[24]", primals_193: "i64[]", primals_194: "f32[24]", primals_195: "f32[24]", primals_196: "i64[]", primals_197: "f32[24]", primals_198: "f32[24]", primals_199: "i64[]", primals_200: "f32[24]", primals_201: "f32[24]", primals_202: "i64[]", primals_203: "f32[56]", primals_204: "f32[56]", primals_205: "i64[]", primals_206: "f32[56]", primals_207: "f32[56]", primals_208: "i64[]", primals_209: "f32[56]", primals_210: "f32[56]", primals_211: "i64[]", primals_212: "f32[56]", primals_213: "f32[56]", primals_214: "i64[]", primals_215: "f32[152]", primals_216: "f32[152]", primals_217: "i64[]", primals_218: "f32[152]", primals_219: "f32[152]", primals_220: "i64[]", primals_221: "f32[152]", primals_222: "f32[152]", primals_223: "i64[]", primals_224: "f32[152]", primals_225: "f32[152]", primals_226: "i64[]", primals_227: "f32[152]", primals_228: "f32[152]", primals_229: "i64[]", primals_230: "f32[152]", primals_231: "f32[152]", primals_232: "i64[]", primals_233: "f32[152]", primals_234: "f32[152]", primals_235: "i64[]", primals_236: "f32[152]", primals_237: "f32[152]", primals_238: "i64[]", primals_239: "f32[152]", primals_240: "f32[152]", primals_241: "i64[]", primals_242: "f32[152]", primals_243: "f32[152]", primals_244: "i64[]", primals_245: "f32[152]", primals_246: "f32[152]", primals_247: "i64[]", primals_248: "f32[152]", primals_249: "f32[152]", primals_250: "i64[]", primals_251: "f32[152]", primals_252: "f32[152]", primals_253: "i64[]", primals_254: "f32[368]", primals_255: "f32[368]", primals_256: "i64[]", primals_257: "f32[368]", primals_258: "f32[368]", primals_259: "i64[]", primals_260: "f32[368]", primals_261: "f32[368]", primals_262: "i64[]", primals_263: "f32[368]", primals_264: "f32[368]", primals_265: "i64[]", primals_266: "f32[368]", primals_267: "f32[368]", primals_268: "i64[]", primals_269: "f32[368]", primals_270: "f32[368]", primals_271: "i64[]", primals_272: "f32[368]", primals_273: "f32[368]", primals_274: "i64[]", primals_275: "f32[368]", primals_276: "f32[368]", primals_277: "i64[]", primals_278: "f32[368]", primals_279: "f32[368]", primals_280: "i64[]", primals_281: "f32[368]", primals_282: "f32[368]", primals_283: "i64[]", primals_284: "f32[368]", primals_285: "f32[368]", primals_286: "i64[]", primals_287: "f32[368]", primals_288: "f32[368]", primals_289: "i64[]", primals_290: "f32[368]", primals_291: "f32[368]", primals_292: "i64[]", primals_293: "f32[368]", primals_294: "f32[368]", primals_295: "i64[]", primals_296: "f32[368]", primals_297: "f32[368]", primals_298: "i64[]", primals_299: "f32[368]", primals_300: "f32[368]", primals_301: "i64[]", primals_302: "f32[368]", primals_303: "f32[368]", primals_304: "i64[]", primals_305: "f32[368]", primals_306: "f32[368]", primals_307: "i64[]", primals_308: "f32[368]", primals_309: "f32[368]", primals_310: "i64[]", primals_311: "f32[368]", primals_312: "f32[368]", primals_313: "i64[]", primals_314: "f32[368]", primals_315: "f32[368]", primals_316: "i64[]", primals_317: "f32[368]", primals_318: "f32[368]", primals_319: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_319, primals_89, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_187, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
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
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_190, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 24, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 24, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[24]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_7: "f32[24]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[24]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[24]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
    add_8: "f32[24]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_91, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_193, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 24, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 24, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[24]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_12: "f32[24]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[24]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[24]" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
    add_13: "f32[24]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 24, 1, 1]" = torch.ops.aten.mean.dim(relu_2, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_3: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_92, primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_3: "f32[8, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_4: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_94, primals_95, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[8, 24, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_21: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(relu_2, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(mul_21, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_196, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 24, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 24, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
    mul_22: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_23: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_24: "f32[24]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
    add_17: "f32[24]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    squeeze_11: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_25: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_26: "f32[24]" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
    mul_27: "f32[24]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
    add_18: "f32[24]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    unsqueeze_12: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_28: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
    unsqueeze_14: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(relu, primals_97, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_199, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 24, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 24, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
    mul_29: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_30: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_31: "f32[24]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_22: "f32[24]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
    squeeze_14: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_32: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_33: "f32[24]" = torch.ops.aten.mul.Tensor(mul_32, 0.1);  mul_32 = None
    mul_34: "f32[24]" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
    add_23: "f32[24]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    unsqueeze_16: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_35: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
    unsqueeze_18: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_25: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_4: "f32[8, 24, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 56, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_98, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_202, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 56, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 56, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_11)
    mul_36: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_37: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_38: "f32[56]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_28: "f32[56]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
    squeeze_17: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_39: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_40: "f32[56]" = torch.ops.aten.mul.Tensor(mul_39, 0.1);  mul_39 = None
    mul_41: "f32[56]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
    add_29: "f32[56]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
    unsqueeze_20: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_42: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
    unsqueeze_22: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 56, 56, 56]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_23);  mul_42 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 56, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(relu_5, primals_99, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_205, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 56, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 56, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
    mul_43: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_44: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_45: "f32[56]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_33: "f32[56]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    squeeze_20: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_46: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001594642002871);  squeeze_20 = None
    mul_47: "f32[56]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
    mul_48: "f32[56]" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
    add_34: "f32[56]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    unsqueeze_24: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_49: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
    unsqueeze_26: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 56, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 56, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_9: "f32[8, 6, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_100, primals_101, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_7: "f32[8, 6, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_10: "f32[8, 56, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_102, primals_103, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[8, 56, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_50: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(relu_6, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_50, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_208, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 56, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 56, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
    mul_51: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_52: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_53: "f32[56]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_38: "f32[56]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    squeeze_23: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_54: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001594642002871);  squeeze_23 = None
    mul_55: "f32[56]" = torch.ops.aten.mul.Tensor(mul_54, 0.1);  mul_54 = None
    mul_56: "f32[56]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
    add_39: "f32[56]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
    unsqueeze_28: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_57: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
    unsqueeze_30: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(relu_4, primals_105, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_211, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 56, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 56, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_17)
    mul_58: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_59: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_60: "f32[56]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_43: "f32[56]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
    squeeze_26: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_61: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
    mul_62: "f32[56]" = torch.ops.aten.mul.Tensor(mul_61, 0.1);  mul_61 = None
    mul_63: "f32[56]" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
    add_44: "f32[56]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    unsqueeze_32: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_64: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_33);  mul_58 = unsqueeze_33 = None
    unsqueeze_34: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_35);  mul_64 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_46: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_40, add_45);  add_40 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_8: "f32[8, 56, 28, 28]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 152, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_214, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 152, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 152, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_19)
    mul_65: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_66: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_67: "f32[152]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_49: "f32[152]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    squeeze_29: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_68: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001594642002871);  squeeze_29 = None
    mul_69: "f32[152]" = torch.ops.aten.mul.Tensor(mul_68, 0.1);  mul_68 = None
    mul_70: "f32[152]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
    add_50: "f32[152]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
    unsqueeze_36: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_71: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_37);  mul_65 = unsqueeze_37 = None
    unsqueeze_38: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 152, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_39);  mul_71 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 152, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_9, primals_107, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_217, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 152, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 152, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_21)
    mul_72: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_73: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_74: "f32[152]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_54: "f32[152]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    squeeze_32: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_75: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0006381620931717);  squeeze_32 = None
    mul_76: "f32[152]" = torch.ops.aten.mul.Tensor(mul_75, 0.1);  mul_75 = None
    mul_77: "f32[152]" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
    add_55: "f32[152]" = torch.ops.aten.add.Tensor(mul_76, mul_77);  mul_76 = mul_77 = None
    unsqueeze_40: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_78: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_41);  mul_72 = unsqueeze_41 = None
    unsqueeze_42: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_78, unsqueeze_43);  mul_78 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_10, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_15: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_108, primals_109, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_11: "f32[8, 14, 1, 1]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_16: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_110, primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_79: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_10, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_79, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_220, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 152, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 152, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_23)
    mul_80: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_81: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_82: "f32[152]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_59: "f32[152]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    squeeze_35: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_83: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0006381620931717);  squeeze_35 = None
    mul_84: "f32[152]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
    mul_85: "f32[152]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
    add_60: "f32[152]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    unsqueeze_44: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_86: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_45);  mul_80 = unsqueeze_45 = None
    unsqueeze_46: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_47);  mul_86 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_8, primals_113, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_223, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 152, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 152, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_25)
    mul_87: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_88: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_89: "f32[152]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_64: "f32[152]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    squeeze_38: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_90: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0006381620931717);  squeeze_38 = None
    mul_91: "f32[152]" = torch.ops.aten.mul.Tensor(mul_90, 0.1);  mul_90 = None
    mul_92: "f32[152]" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
    add_65: "f32[152]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
    unsqueeze_48: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_93: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_49);  mul_87 = unsqueeze_49 = None
    unsqueeze_50: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_51);  mul_93 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_67: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_61, add_66);  add_61 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_12: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_12, primals_114, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_226, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 152, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 152, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_27)
    mul_94: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_95: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_96: "f32[152]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_70: "f32[152]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    squeeze_41: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_97: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_98: "f32[152]" = torch.ops.aten.mul.Tensor(mul_97, 0.1);  mul_97 = None
    mul_99: "f32[152]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
    add_71: "f32[152]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
    unsqueeze_52: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_100: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_53);  mul_94 = unsqueeze_53 = None
    unsqueeze_54: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_55);  mul_100 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_13, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_229, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 152, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 152, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_29)
    mul_101: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_102: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_103: "f32[152]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_75: "f32[152]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    squeeze_44: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_104: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_105: "f32[152]" = torch.ops.aten.mul.Tensor(mul_104, 0.1);  mul_104 = None
    mul_106: "f32[152]" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
    add_76: "f32[152]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    unsqueeze_56: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_107: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_101, unsqueeze_57);  mul_101 = unsqueeze_57 = None
    unsqueeze_58: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_59);  mul_107 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_14, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_21: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_116, primals_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_15: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_22: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_118, primals_119, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_3: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_108: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_14, sigmoid_3);  sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_108, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_232, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 152, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 152, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_31)
    mul_109: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_110: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_111: "f32[152]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_80: "f32[152]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
    squeeze_47: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_112: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_113: "f32[152]" = torch.ops.aten.mul.Tensor(mul_112, 0.1);  mul_112 = None
    mul_114: "f32[152]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
    add_81: "f32[152]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    unsqueeze_60: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_115: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_61);  mul_109 = unsqueeze_61 = None
    unsqueeze_62: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_115, unsqueeze_63);  mul_115 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_83: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_82, relu_12);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_16: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_83);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_16, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_235, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 152, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 152, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_85: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_16: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
    mul_116: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_117: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_118: "f32[152]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_86: "f32[152]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    squeeze_50: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_119: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_120: "f32[152]" = torch.ops.aten.mul.Tensor(mul_119, 0.1);  mul_119 = None
    mul_121: "f32[152]" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
    add_87: "f32[152]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    unsqueeze_64: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_122: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_65);  mul_116 = unsqueeze_65 = None
    unsqueeze_66: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_88: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_67);  mul_122 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_122, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_238, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 152, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 152, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_35)
    mul_123: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_124: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_125: "f32[152]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_91: "f32[152]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    squeeze_53: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_126: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_127: "f32[152]" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
    mul_128: "f32[152]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
    add_92: "f32[152]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    unsqueeze_68: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_129: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_69);  mul_123 = unsqueeze_69 = None
    unsqueeze_70: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_71);  mul_129 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_26: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_123, primals_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_19: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_26);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_27: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_19, primals_125, primals_126, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_130: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_18, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_130, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_241, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 152, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 152, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_95: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_18: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_37)
    mul_131: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_132: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_133: "f32[152]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_96: "f32[152]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    squeeze_56: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_134: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_135: "f32[152]" = torch.ops.aten.mul.Tensor(mul_134, 0.1);  mul_134 = None
    mul_136: "f32[152]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
    add_97: "f32[152]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    unsqueeze_72: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_137: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_73);  mul_131 = unsqueeze_73 = None
    unsqueeze_74: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_98: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_75);  mul_137 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_99: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_98, relu_16);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_20: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_20, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_244, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 152, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 152, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_101: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_19: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_39)
    mul_138: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_139: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_140: "f32[152]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_102: "f32[152]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_59: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_141: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_142: "f32[152]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[152]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
    add_103: "f32[152]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_76: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_144: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
    unsqueeze_78: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_104: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_247, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 152, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 152, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_106: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_20: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_41)
    mul_145: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_146: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_147: "f32[152]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_107: "f32[152]" = torch.ops.aten.add.Tensor(mul_146, mul_147);  mul_146 = mul_147 = None
    squeeze_62: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_148: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_149: "f32[152]" = torch.ops.aten.mul.Tensor(mul_148, 0.1);  mul_148 = None
    mul_150: "f32[152]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
    add_108: "f32[152]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    unsqueeze_80: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_151: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_81);  mul_145 = unsqueeze_81 = None
    unsqueeze_82: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_109: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_151, unsqueeze_83);  mul_151 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 152, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_31: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_130, primals_131, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_23: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_32: "f32[8, 152, 1, 1]" = torch.ops.aten.convolution.default(relu_23, primals_132, primals_133, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_5: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_152: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(relu_22, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 152, 14, 14]" = torch.ops.aten.convolution.default(mul_152, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_250, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 152, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 152, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 152, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_43)
    mul_153: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[152]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_154: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_155: "f32[152]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_112: "f32[152]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    squeeze_65: "f32[152]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_156: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
    mul_157: "f32[152]" = torch.ops.aten.mul.Tensor(mul_156, 0.1);  mul_156 = None
    mul_158: "f32[152]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
    add_113: "f32[152]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    unsqueeze_84: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_159: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_85);  mul_153 = unsqueeze_85 = None
    unsqueeze_86: "f32[152, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_87);  mul_159 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_115: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(add_114, relu_20);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_24: "f32[8, 152, 14, 14]" = torch.ops.aten.relu.default(add_115);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 368, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_253, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 368, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 368, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_117: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_22: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_45)
    mul_160: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_161: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_162: "f32[368]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_118: "f32[368]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    squeeze_68: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_163: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_164: "f32[368]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
    mul_165: "f32[368]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
    add_119: "f32[368]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    unsqueeze_88: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_166: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_89);  mul_160 = unsqueeze_89 = None
    unsqueeze_90: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_120: "f32[8, 368, 14, 14]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_91);  mul_166 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 368, 14, 14]" = torch.ops.aten.relu.default(add_120);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_25, primals_136, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_256, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 368, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 368, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_122: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_23: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_47)
    mul_167: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_168: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_169: "f32[368]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_123: "f32[368]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
    squeeze_71: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_170: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0025575447570332);  squeeze_71 = None
    mul_171: "f32[368]" = torch.ops.aten.mul.Tensor(mul_170, 0.1);  mul_170 = None
    mul_172: "f32[368]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_124: "f32[368]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    unsqueeze_92: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_173: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_93);  mul_167 = unsqueeze_93 = None
    unsqueeze_94: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_125: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_95);  mul_173 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_125);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_26, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_36: "f32[8, 38, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_27: "f32[8, 38, 1, 1]" = torch.ops.aten.relu.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_37: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_27, primals_139, primals_140, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_174: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_26, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_174, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_259, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 368, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 368, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_127: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_24: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_49)
    mul_175: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_176: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_177: "f32[368]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_128: "f32[368]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_74: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_178: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0025575447570332);  squeeze_74 = None
    mul_179: "f32[368]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[368]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_129: "f32[368]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_96: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_181: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_97);  mul_175 = unsqueeze_97 = None
    unsqueeze_98: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_130: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_99);  mul_181 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_24, primals_142, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_131: "i64[]" = torch.ops.aten.add.Tensor(primals_262, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 368, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 368, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_132: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_25: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_51)
    mul_182: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_183: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_184: "f32[368]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_133: "f32[368]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_77: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_185: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0025575447570332);  squeeze_77 = None
    mul_186: "f32[368]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[368]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_134: "f32[368]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_100: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_188: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_101);  mul_182 = unsqueeze_101 = None
    unsqueeze_102: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_135: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_103);  mul_188 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_136: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_130, add_135);  add_130 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_28: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_136);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_28, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_265, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 368, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 368, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_138: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_26: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_53)
    mul_189: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_190: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_191: "f32[368]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_139: "f32[368]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_80: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_192: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0025575447570332);  squeeze_80 = None
    mul_193: "f32[368]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[368]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_140: "f32[368]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_104: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_195: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_105);  mul_189 = unsqueeze_105 = None
    unsqueeze_106: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_141: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_107);  mul_195 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_141);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_29, primals_144, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_142: "i64[]" = torch.ops.aten.add.Tensor(primals_268, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 368, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 368, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_143: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_27: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_55)
    mul_196: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_197: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_198: "f32[368]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_144: "f32[368]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_83: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_199: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0025575447570332);  squeeze_83 = None
    mul_200: "f32[368]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[368]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_145: "f32[368]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_108: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_202: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_109);  mul_196 = unsqueeze_109 = None
    unsqueeze_110: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_146: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_111);  mul_202 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_146);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_30, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_42: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_31: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_43: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_31, primals_147, primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_203: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_30, sigmoid_7);  sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_203, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_147: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 368, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 368, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_148: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_28: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_57)
    mul_204: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_205: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_206: "f32[368]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_149: "f32[368]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    squeeze_86: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_207: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0025575447570332);  squeeze_86 = None
    mul_208: "f32[368]" = torch.ops.aten.mul.Tensor(mul_207, 0.1);  mul_207 = None
    mul_209: "f32[368]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_150: "f32[368]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    unsqueeze_112: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_210: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_113);  mul_204 = unsqueeze_113 = None
    unsqueeze_114: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_151: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_115);  mul_210 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_152: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_151, relu_28);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_32: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_152);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_274, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 368, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 368, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_154: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_29: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_59)
    mul_211: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_212: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_213: "f32[368]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_155: "f32[368]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    squeeze_89: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_214: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0025575447570332);  squeeze_89 = None
    mul_215: "f32[368]" = torch.ops.aten.mul.Tensor(mul_214, 0.1);  mul_214 = None
    mul_216: "f32[368]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_156: "f32[368]" = torch.ops.aten.add.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
    unsqueeze_116: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_217: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_117);  mul_211 = unsqueeze_117 = None
    unsqueeze_118: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_157: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_119);  mul_217 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_157);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_33, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 368, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 368, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_159: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_30: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_61)
    mul_218: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_219: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_220: "f32[368]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_160: "f32[368]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    squeeze_92: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_221: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0025575447570332);  squeeze_92 = None
    mul_222: "f32[368]" = torch.ops.aten.mul.Tensor(mul_221, 0.1);  mul_221 = None
    mul_223: "f32[368]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_161: "f32[368]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    unsqueeze_120: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_224: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_121);  mul_218 = unsqueeze_121 = None
    unsqueeze_122: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_162: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_123);  mul_224 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_162);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_34, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_47: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_152, primals_153, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_35: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_48: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_35, primals_154, primals_155, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_225: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_34, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_225, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_163: "i64[]" = torch.ops.aten.add.Tensor(primals_280, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 368, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 368, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_164: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_31: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_63)
    mul_226: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_227: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_228: "f32[368]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_165: "f32[368]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    squeeze_95: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_229: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0025575447570332);  squeeze_95 = None
    mul_230: "f32[368]" = torch.ops.aten.mul.Tensor(mul_229, 0.1);  mul_229 = None
    mul_231: "f32[368]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_166: "f32[368]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    unsqueeze_124: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_232: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_125);  mul_226 = unsqueeze_125 = None
    unsqueeze_126: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_167: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_127);  mul_232 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_168: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_167, relu_32);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_36: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_168);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_36, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 368, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 368, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_170: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_32: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_65)
    mul_233: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_234: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_235: "f32[368]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_171: "f32[368]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_98: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_236: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0025575447570332);  squeeze_98 = None
    mul_237: "f32[368]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[368]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_172: "f32[368]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    unsqueeze_128: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_239: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_129);  mul_233 = unsqueeze_129 = None
    unsqueeze_130: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_173: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_131);  mul_239 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_173);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_37, primals_158, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_286, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 368, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 368, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_175: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_33: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_67)
    mul_240: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_241: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_242: "f32[368]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_176: "f32[368]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_101: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_243: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0025575447570332);  squeeze_101 = None
    mul_244: "f32[368]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[368]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_177: "f32[368]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    unsqueeze_132: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_246: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_133);  mul_240 = unsqueeze_133 = None
    unsqueeze_134: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_178: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_135);  mul_246 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_178);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_52: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_159, primals_160, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_39: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_53: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_39, primals_161, primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_9: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_247: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_38, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_247, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_179: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 368, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 368, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_180: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_34: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_69)
    mul_248: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_249: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_250: "f32[368]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_181: "f32[368]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_104: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_251: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0025575447570332);  squeeze_104 = None
    mul_252: "f32[368]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[368]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_182: "f32[368]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_136: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_254: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_137);  mul_248 = unsqueeze_137 = None
    unsqueeze_138: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_183: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_139);  mul_254 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_184: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_183, relu_36);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_40: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_40, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_292, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 368, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 368, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_186: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_35: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_71)
    mul_255: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_256: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_257: "f32[368]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_187: "f32[368]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    squeeze_107: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_258: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0025575447570332);  squeeze_107 = None
    mul_259: "f32[368]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
    mul_260: "f32[368]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_188: "f32[368]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    unsqueeze_140: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_261: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_141);  mul_255 = unsqueeze_141 = None
    unsqueeze_142: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_189: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_143);  mul_261 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_41, primals_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 368, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 368, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_191: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_36: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_73)
    mul_262: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_263: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_264: "f32[368]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_192: "f32[368]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    squeeze_110: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_265: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0025575447570332);  squeeze_110 = None
    mul_266: "f32[368]" = torch.ops.aten.mul.Tensor(mul_265, 0.1);  mul_265 = None
    mul_267: "f32[368]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_193: "f32[368]" = torch.ops.aten.add.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    unsqueeze_144: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_268: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_145);  mul_262 = unsqueeze_145 = None
    unsqueeze_146: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_194: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_268, unsqueeze_147);  mul_268 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_194);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_42, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_57: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_166, primals_167, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_43: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_57);  convolution_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_58: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_43, primals_168, primals_169, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_269: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_42, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_269, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_195: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 368, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 368, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_196: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_37: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_75)
    mul_270: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_271: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_272: "f32[368]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_197: "f32[368]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    squeeze_113: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_273: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_274: "f32[368]" = torch.ops.aten.mul.Tensor(mul_273, 0.1);  mul_273 = None
    mul_275: "f32[368]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_198: "f32[368]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    unsqueeze_148: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_276: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_149);  mul_270 = unsqueeze_149 = None
    unsqueeze_150: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_199: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_151);  mul_276 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_200: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_199, relu_40);  add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_44: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_44, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 368, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 368, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_202: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_38: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_77)
    mul_277: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_278: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_279: "f32[368]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_203: "f32[368]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    squeeze_116: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_280: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_281: "f32[368]" = torch.ops.aten.mul.Tensor(mul_280, 0.1);  mul_280 = None
    mul_282: "f32[368]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_204: "f32[368]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    unsqueeze_152: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_283: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_153);  mul_277 = unsqueeze_153 = None
    unsqueeze_154: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_205: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_155);  mul_283 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_45: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_61: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_45, primals_172, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 368, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 368, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_207: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_39: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_79)
    mul_284: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_285: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_286: "f32[368]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_208: "f32[368]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    squeeze_119: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_287: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_288: "f32[368]" = torch.ops.aten.mul.Tensor(mul_287, 0.1);  mul_287 = None
    mul_289: "f32[368]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_209: "f32[368]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_156: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_290: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_157);  mul_284 = unsqueeze_157 = None
    unsqueeze_158: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_210: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_159);  mul_290 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_210);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_62: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_47: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_62);  convolution_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_63: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_47, primals_175, primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_11: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_291: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_46, sigmoid_11);  sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_291, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_211: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 368, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 368, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_212: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    sub_40: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_81)
    mul_292: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_293: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_294: "f32[368]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_213: "f32[368]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    squeeze_122: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_295: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_296: "f32[368]" = torch.ops.aten.mul.Tensor(mul_295, 0.1);  mul_295 = None
    mul_297: "f32[368]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_214: "f32[368]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    unsqueeze_160: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_298: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_161);  mul_292 = unsqueeze_161 = None
    unsqueeze_162: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_215: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_163);  mul_298 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_216: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_215, relu_44);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_48: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_48, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 368, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 368, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_218: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_41: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_83)
    mul_299: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_300: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_301: "f32[368]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_219: "f32[368]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    squeeze_125: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_302: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_303: "f32[368]" = torch.ops.aten.mul.Tensor(mul_302, 0.1);  mul_302 = None
    mul_304: "f32[368]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_220: "f32[368]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    unsqueeze_164: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_305: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_165);  mul_299 = unsqueeze_165 = None
    unsqueeze_166: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_221: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_167);  mul_305 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_66: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(relu_49, primals_179, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 368, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 368, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_223: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_42: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_85)
    mul_306: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_307: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_308: "f32[368]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_224: "f32[368]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    squeeze_128: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_309: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0025575447570332);  squeeze_128 = None
    mul_310: "f32[368]" = torch.ops.aten.mul.Tensor(mul_309, 0.1);  mul_309 = None
    mul_311: "f32[368]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_225: "f32[368]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    unsqueeze_168: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_312: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_169);  mul_306 = unsqueeze_169 = None
    unsqueeze_170: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_226: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_312, unsqueeze_171);  mul_312 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_50, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_67: "f32[8, 92, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    relu_51: "f32[8, 92, 1, 1]" = torch.ops.aten.relu.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_68: "f32[8, 368, 1, 1]" = torch.ops.aten.convolution.default(relu_51, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_313: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(relu_50, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_69: "f32[8, 368, 7, 7]" = torch.ops.aten.convolution.default(mul_313, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_316, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 368, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 368, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_228: "f32[1, 368, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 368, 1, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_43: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_87)
    mul_314: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[368]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_315: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_316: "f32[368]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_229: "f32[368]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    squeeze_131: "f32[368]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_317: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0025575447570332);  squeeze_131 = None
    mul_318: "f32[368]" = torch.ops.aten.mul.Tensor(mul_317, 0.1);  mul_317 = None
    mul_319: "f32[368]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_230: "f32[368]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    unsqueeze_172: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_320: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_173);  mul_314 = unsqueeze_173 = None
    unsqueeze_174: "f32[368, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_231: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_175);  mul_320 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:244, code: x = self.drop_path(x) + self.downsample(shortcut)
    add_232: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(add_231, relu_48);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    relu_52: "f32[8, 368, 7, 7]" = torch.ops.aten.relu.default(add_232);  add_232 = None
    alias_65: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(relu_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_13: "f32[8, 368, 1, 1]" = torch.ops.aten.mean.dim(relu_52, [-1, -2], True);  relu_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 368]" = torch.ops.aten.view.default(mean_13, [8, 368]);  mean_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 368]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[368, 1000]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_186, clone, permute);  primals_186 = None
    permute_1: "f32[1000, 368]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    alias_66: "f32[8, 368, 7, 7]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_176: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_177: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
    unsqueeze_178: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_188: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_189: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
    unsqueeze_190: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_200: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_201: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
    unsqueeze_202: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_212: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_213: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
    unsqueeze_214: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_224: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_225: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
    unsqueeze_226: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_236: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_237: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
    unsqueeze_238: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_249: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_261: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_272: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_273: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_285: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_297: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_309: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_321: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_333: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_345: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_357: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_368: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_369: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_381: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_393: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_405: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_416: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_417: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_429: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_441: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_453: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_464: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_465: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_477: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_489: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_501: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_512: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_513: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_524: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_525: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_537: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_548: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_549: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_560: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_561: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_572: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_573: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_585: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_596: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_597: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_608: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_609: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_620: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_621: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_632: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_633: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_644: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_645: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_656: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_657: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_668: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_669: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_680: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_681: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_692: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_693: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_187, add);  primals_187 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_188, add_2);  primals_188 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_189, add_3);  primals_189 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_190, add_5);  primals_190 = add_5 = None
    copy__4: "f32[24]" = torch.ops.aten.copy_.default(primals_191, add_7);  primals_191 = add_7 = None
    copy__5: "f32[24]" = torch.ops.aten.copy_.default(primals_192, add_8);  primals_192 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_193, add_10);  primals_193 = add_10 = None
    copy__7: "f32[24]" = torch.ops.aten.copy_.default(primals_194, add_12);  primals_194 = add_12 = None
    copy__8: "f32[24]" = torch.ops.aten.copy_.default(primals_195, add_13);  primals_195 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_196, add_15);  primals_196 = add_15 = None
    copy__10: "f32[24]" = torch.ops.aten.copy_.default(primals_197, add_17);  primals_197 = add_17 = None
    copy__11: "f32[24]" = torch.ops.aten.copy_.default(primals_198, add_18);  primals_198 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_199, add_20);  primals_199 = add_20 = None
    copy__13: "f32[24]" = torch.ops.aten.copy_.default(primals_200, add_22);  primals_200 = add_22 = None
    copy__14: "f32[24]" = torch.ops.aten.copy_.default(primals_201, add_23);  primals_201 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_202, add_26);  primals_202 = add_26 = None
    copy__16: "f32[56]" = torch.ops.aten.copy_.default(primals_203, add_28);  primals_203 = add_28 = None
    copy__17: "f32[56]" = torch.ops.aten.copy_.default(primals_204, add_29);  primals_204 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_205, add_31);  primals_205 = add_31 = None
    copy__19: "f32[56]" = torch.ops.aten.copy_.default(primals_206, add_33);  primals_206 = add_33 = None
    copy__20: "f32[56]" = torch.ops.aten.copy_.default(primals_207, add_34);  primals_207 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_208, add_36);  primals_208 = add_36 = None
    copy__22: "f32[56]" = torch.ops.aten.copy_.default(primals_209, add_38);  primals_209 = add_38 = None
    copy__23: "f32[56]" = torch.ops.aten.copy_.default(primals_210, add_39);  primals_210 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_211, add_41);  primals_211 = add_41 = None
    copy__25: "f32[56]" = torch.ops.aten.copy_.default(primals_212, add_43);  primals_212 = add_43 = None
    copy__26: "f32[56]" = torch.ops.aten.copy_.default(primals_213, add_44);  primals_213 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_214, add_47);  primals_214 = add_47 = None
    copy__28: "f32[152]" = torch.ops.aten.copy_.default(primals_215, add_49);  primals_215 = add_49 = None
    copy__29: "f32[152]" = torch.ops.aten.copy_.default(primals_216, add_50);  primals_216 = add_50 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_217, add_52);  primals_217 = add_52 = None
    copy__31: "f32[152]" = torch.ops.aten.copy_.default(primals_218, add_54);  primals_218 = add_54 = None
    copy__32: "f32[152]" = torch.ops.aten.copy_.default(primals_219, add_55);  primals_219 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_220, add_57);  primals_220 = add_57 = None
    copy__34: "f32[152]" = torch.ops.aten.copy_.default(primals_221, add_59);  primals_221 = add_59 = None
    copy__35: "f32[152]" = torch.ops.aten.copy_.default(primals_222, add_60);  primals_222 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_223, add_62);  primals_223 = add_62 = None
    copy__37: "f32[152]" = torch.ops.aten.copy_.default(primals_224, add_64);  primals_224 = add_64 = None
    copy__38: "f32[152]" = torch.ops.aten.copy_.default(primals_225, add_65);  primals_225 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_226, add_68);  primals_226 = add_68 = None
    copy__40: "f32[152]" = torch.ops.aten.copy_.default(primals_227, add_70);  primals_227 = add_70 = None
    copy__41: "f32[152]" = torch.ops.aten.copy_.default(primals_228, add_71);  primals_228 = add_71 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_229, add_73);  primals_229 = add_73 = None
    copy__43: "f32[152]" = torch.ops.aten.copy_.default(primals_230, add_75);  primals_230 = add_75 = None
    copy__44: "f32[152]" = torch.ops.aten.copy_.default(primals_231, add_76);  primals_231 = add_76 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_232, add_78);  primals_232 = add_78 = None
    copy__46: "f32[152]" = torch.ops.aten.copy_.default(primals_233, add_80);  primals_233 = add_80 = None
    copy__47: "f32[152]" = torch.ops.aten.copy_.default(primals_234, add_81);  primals_234 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_235, add_84);  primals_235 = add_84 = None
    copy__49: "f32[152]" = torch.ops.aten.copy_.default(primals_236, add_86);  primals_236 = add_86 = None
    copy__50: "f32[152]" = torch.ops.aten.copy_.default(primals_237, add_87);  primals_237 = add_87 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_238, add_89);  primals_238 = add_89 = None
    copy__52: "f32[152]" = torch.ops.aten.copy_.default(primals_239, add_91);  primals_239 = add_91 = None
    copy__53: "f32[152]" = torch.ops.aten.copy_.default(primals_240, add_92);  primals_240 = add_92 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_241, add_94);  primals_241 = add_94 = None
    copy__55: "f32[152]" = torch.ops.aten.copy_.default(primals_242, add_96);  primals_242 = add_96 = None
    copy__56: "f32[152]" = torch.ops.aten.copy_.default(primals_243, add_97);  primals_243 = add_97 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_244, add_100);  primals_244 = add_100 = None
    copy__58: "f32[152]" = torch.ops.aten.copy_.default(primals_245, add_102);  primals_245 = add_102 = None
    copy__59: "f32[152]" = torch.ops.aten.copy_.default(primals_246, add_103);  primals_246 = add_103 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_247, add_105);  primals_247 = add_105 = None
    copy__61: "f32[152]" = torch.ops.aten.copy_.default(primals_248, add_107);  primals_248 = add_107 = None
    copy__62: "f32[152]" = torch.ops.aten.copy_.default(primals_249, add_108);  primals_249 = add_108 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_250, add_110);  primals_250 = add_110 = None
    copy__64: "f32[152]" = torch.ops.aten.copy_.default(primals_251, add_112);  primals_251 = add_112 = None
    copy__65: "f32[152]" = torch.ops.aten.copy_.default(primals_252, add_113);  primals_252 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_253, add_116);  primals_253 = add_116 = None
    copy__67: "f32[368]" = torch.ops.aten.copy_.default(primals_254, add_118);  primals_254 = add_118 = None
    copy__68: "f32[368]" = torch.ops.aten.copy_.default(primals_255, add_119);  primals_255 = add_119 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_256, add_121);  primals_256 = add_121 = None
    copy__70: "f32[368]" = torch.ops.aten.copy_.default(primals_257, add_123);  primals_257 = add_123 = None
    copy__71: "f32[368]" = torch.ops.aten.copy_.default(primals_258, add_124);  primals_258 = add_124 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_259, add_126);  primals_259 = add_126 = None
    copy__73: "f32[368]" = torch.ops.aten.copy_.default(primals_260, add_128);  primals_260 = add_128 = None
    copy__74: "f32[368]" = torch.ops.aten.copy_.default(primals_261, add_129);  primals_261 = add_129 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_262, add_131);  primals_262 = add_131 = None
    copy__76: "f32[368]" = torch.ops.aten.copy_.default(primals_263, add_133);  primals_263 = add_133 = None
    copy__77: "f32[368]" = torch.ops.aten.copy_.default(primals_264, add_134);  primals_264 = add_134 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_265, add_137);  primals_265 = add_137 = None
    copy__79: "f32[368]" = torch.ops.aten.copy_.default(primals_266, add_139);  primals_266 = add_139 = None
    copy__80: "f32[368]" = torch.ops.aten.copy_.default(primals_267, add_140);  primals_267 = add_140 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_268, add_142);  primals_268 = add_142 = None
    copy__82: "f32[368]" = torch.ops.aten.copy_.default(primals_269, add_144);  primals_269 = add_144 = None
    copy__83: "f32[368]" = torch.ops.aten.copy_.default(primals_270, add_145);  primals_270 = add_145 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_147);  primals_271 = add_147 = None
    copy__85: "f32[368]" = torch.ops.aten.copy_.default(primals_272, add_149);  primals_272 = add_149 = None
    copy__86: "f32[368]" = torch.ops.aten.copy_.default(primals_273, add_150);  primals_273 = add_150 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_274, add_153);  primals_274 = add_153 = None
    copy__88: "f32[368]" = torch.ops.aten.copy_.default(primals_275, add_155);  primals_275 = add_155 = None
    copy__89: "f32[368]" = torch.ops.aten.copy_.default(primals_276, add_156);  primals_276 = add_156 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_158);  primals_277 = add_158 = None
    copy__91: "f32[368]" = torch.ops.aten.copy_.default(primals_278, add_160);  primals_278 = add_160 = None
    copy__92: "f32[368]" = torch.ops.aten.copy_.default(primals_279, add_161);  primals_279 = add_161 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_280, add_163);  primals_280 = add_163 = None
    copy__94: "f32[368]" = torch.ops.aten.copy_.default(primals_281, add_165);  primals_281 = add_165 = None
    copy__95: "f32[368]" = torch.ops.aten.copy_.default(primals_282, add_166);  primals_282 = add_166 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_169);  primals_283 = add_169 = None
    copy__97: "f32[368]" = torch.ops.aten.copy_.default(primals_284, add_171);  primals_284 = add_171 = None
    copy__98: "f32[368]" = torch.ops.aten.copy_.default(primals_285, add_172);  primals_285 = add_172 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_286, add_174);  primals_286 = add_174 = None
    copy__100: "f32[368]" = torch.ops.aten.copy_.default(primals_287, add_176);  primals_287 = add_176 = None
    copy__101: "f32[368]" = torch.ops.aten.copy_.default(primals_288, add_177);  primals_288 = add_177 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_179);  primals_289 = add_179 = None
    copy__103: "f32[368]" = torch.ops.aten.copy_.default(primals_290, add_181);  primals_290 = add_181 = None
    copy__104: "f32[368]" = torch.ops.aten.copy_.default(primals_291, add_182);  primals_291 = add_182 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_292, add_185);  primals_292 = add_185 = None
    copy__106: "f32[368]" = torch.ops.aten.copy_.default(primals_293, add_187);  primals_293 = add_187 = None
    copy__107: "f32[368]" = torch.ops.aten.copy_.default(primals_294, add_188);  primals_294 = add_188 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_190);  primals_295 = add_190 = None
    copy__109: "f32[368]" = torch.ops.aten.copy_.default(primals_296, add_192);  primals_296 = add_192 = None
    copy__110: "f32[368]" = torch.ops.aten.copy_.default(primals_297, add_193);  primals_297 = add_193 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_195);  primals_298 = add_195 = None
    copy__112: "f32[368]" = torch.ops.aten.copy_.default(primals_299, add_197);  primals_299 = add_197 = None
    copy__113: "f32[368]" = torch.ops.aten.copy_.default(primals_300, add_198);  primals_300 = add_198 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_201);  primals_301 = add_201 = None
    copy__115: "f32[368]" = torch.ops.aten.copy_.default(primals_302, add_203);  primals_302 = add_203 = None
    copy__116: "f32[368]" = torch.ops.aten.copy_.default(primals_303, add_204);  primals_303 = add_204 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_206);  primals_304 = add_206 = None
    copy__118: "f32[368]" = torch.ops.aten.copy_.default(primals_305, add_208);  primals_305 = add_208 = None
    copy__119: "f32[368]" = torch.ops.aten.copy_.default(primals_306, add_209);  primals_306 = add_209 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_211);  primals_307 = add_211 = None
    copy__121: "f32[368]" = torch.ops.aten.copy_.default(primals_308, add_213);  primals_308 = add_213 = None
    copy__122: "f32[368]" = torch.ops.aten.copy_.default(primals_309, add_214);  primals_309 = add_214 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_217);  primals_310 = add_217 = None
    copy__124: "f32[368]" = torch.ops.aten.copy_.default(primals_311, add_219);  primals_311 = add_219 = None
    copy__125: "f32[368]" = torch.ops.aten.copy_.default(primals_312, add_220);  primals_312 = add_220 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_222);  primals_313 = add_222 = None
    copy__127: "f32[368]" = torch.ops.aten.copy_.default(primals_314, add_224);  primals_314 = add_224 = None
    copy__128: "f32[368]" = torch.ops.aten.copy_.default(primals_315, add_225);  primals_315 = add_225 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_316, add_227);  primals_316 = add_227 = None
    copy__130: "f32[368]" = torch.ops.aten.copy_.default(primals_317, add_229);  primals_317 = add_229 = None
    copy__131: "f32[368]" = torch.ops.aten.copy_.default(primals_318, add_230);  primals_318 = add_230 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_90, primals_91, primals_92, primals_94, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_149, primals_150, primals_151, primals_152, primals_154, primals_156, primals_157, primals_158, primals_159, primals_161, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_319, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mean, relu_3, convolution_4, mul_21, convolution_5, squeeze_10, convolution_6, squeeze_13, relu_4, convolution_7, squeeze_16, relu_5, convolution_8, squeeze_19, relu_6, mean_1, relu_7, convolution_10, mul_50, convolution_11, squeeze_22, convolution_12, squeeze_25, relu_8, convolution_13, squeeze_28, relu_9, convolution_14, squeeze_31, relu_10, mean_2, relu_11, convolution_16, mul_79, convolution_17, squeeze_34, convolution_18, squeeze_37, relu_12, convolution_19, squeeze_40, relu_13, convolution_20, squeeze_43, relu_14, mean_3, relu_15, convolution_22, mul_108, convolution_23, squeeze_46, relu_16, convolution_24, squeeze_49, relu_17, convolution_25, squeeze_52, relu_18, mean_4, relu_19, convolution_27, mul_130, convolution_28, squeeze_55, relu_20, convolution_29, squeeze_58, relu_21, convolution_30, squeeze_61, relu_22, mean_5, relu_23, convolution_32, mul_152, convolution_33, squeeze_64, relu_24, convolution_34, squeeze_67, relu_25, convolution_35, squeeze_70, relu_26, mean_6, relu_27, convolution_37, mul_174, convolution_38, squeeze_73, convolution_39, squeeze_76, relu_28, convolution_40, squeeze_79, relu_29, convolution_41, squeeze_82, relu_30, mean_7, relu_31, convolution_43, mul_203, convolution_44, squeeze_85, relu_32, convolution_45, squeeze_88, relu_33, convolution_46, squeeze_91, relu_34, mean_8, relu_35, convolution_48, mul_225, convolution_49, squeeze_94, relu_36, convolution_50, squeeze_97, relu_37, convolution_51, squeeze_100, relu_38, mean_9, relu_39, convolution_53, mul_247, convolution_54, squeeze_103, relu_40, convolution_55, squeeze_106, relu_41, convolution_56, squeeze_109, relu_42, mean_10, relu_43, convolution_58, mul_269, convolution_59, squeeze_112, relu_44, convolution_60, squeeze_115, relu_45, convolution_61, squeeze_118, relu_46, mean_11, relu_47, convolution_63, mul_291, convolution_64, squeeze_121, relu_48, convolution_65, squeeze_124, relu_49, convolution_66, squeeze_127, relu_50, mean_12, relu_51, convolution_68, mul_313, convolution_69, squeeze_130, clone, permute_1, le, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694]
    