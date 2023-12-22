from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[192]", primals_10: "f32[192]", primals_11: "f32[192]", primals_12: "f32[192]", primals_13: "f32[192]", primals_14: "f32[192]", primals_15: "f32[192]", primals_16: "f32[192]", primals_17: "f32[192]", primals_18: "f32[192]", primals_19: "f32[160]", primals_20: "f32[160]", primals_21: "f32[160]", primals_22: "f32[160]", primals_23: "f32[640]", primals_24: "f32[640]", primals_25: "f32[640]", primals_26: "f32[640]", primals_27: "f32[160]", primals_28: "f32[160]", primals_29: "f32[160]", primals_30: "f32[160]", primals_31: "f32[640]", primals_32: "f32[640]", primals_33: "f32[160]", primals_34: "f32[160]", primals_35: "f32[160]", primals_36: "f32[160]", primals_37: "f32[640]", primals_38: "f32[640]", primals_39: "f32[160]", primals_40: "f32[160]", primals_41: "f32[160]", primals_42: "f32[160]", primals_43: "f32[640]", primals_44: "f32[640]", primals_45: "f32[160]", primals_46: "f32[160]", primals_47: "f32[160]", primals_48: "f32[160]", primals_49: "f32[640]", primals_50: "f32[640]", primals_51: "f32[160]", primals_52: "f32[160]", primals_53: "f32[160]", primals_54: "f32[160]", primals_55: "f32[640]", primals_56: "f32[640]", primals_57: "f32[1920]", primals_58: "f32[1920]", primals_59: "f32[1920]", primals_60: "f32[1920]", primals_61: "f32[640]", primals_62: "f32[640]", primals_63: "f32[640]", primals_64: "f32[640]", primals_65: "f32[1920]", primals_66: "f32[1920]", primals_67: "f32[1920]", primals_68: "f32[1920]", primals_69: "f32[640]", primals_70: "f32[640]", primals_71: "f32[1920]", primals_72: "f32[1920]", primals_73: "f32[1920]", primals_74: "f32[1920]", primals_75: "f32[640]", primals_76: "f32[640]", primals_77: "f32[1920]", primals_78: "f32[1920]", primals_79: "f32[1920]", primals_80: "f32[1920]", primals_81: "f32[640]", primals_82: "f32[640]", primals_83: "f32[1920]", primals_84: "f32[1920]", primals_85: "f32[1920]", primals_86: "f32[1920]", primals_87: "f32[640]", primals_88: "f32[640]", primals_89: "f32[1920]", primals_90: "f32[1920]", primals_91: "f32[1920]", primals_92: "f32[1920]", primals_93: "f32[640]", primals_94: "f32[640]", primals_95: "f32[1920]", primals_96: "f32[1920]", primals_97: "f32[1920]", primals_98: "f32[1920]", primals_99: "f32[640]", primals_100: "f32[640]", primals_101: "f32[1920]", primals_102: "f32[1920]", primals_103: "f32[1920]", primals_104: "f32[1920]", primals_105: "f32[640]", primals_106: "f32[640]", primals_107: "f32[1920]", primals_108: "f32[1920]", primals_109: "f32[1920]", primals_110: "f32[1920]", primals_111: "f32[640]", primals_112: "f32[640]", primals_113: "f32[2560]", primals_114: "f32[2560]", primals_115: "f32[32, 3, 3, 3]", primals_116: "f32[128, 32, 3, 3]", primals_117: "f32[128, 128, 3, 3]", primals_118: "f32[128, 32, 1, 1]", primals_119: "f32[192, 128, 3, 3]", primals_120: "f32[192, 192, 3, 3]", primals_121: "f32[192, 128, 1, 1]", primals_122: "f32[192, 192, 3, 3]", primals_123: "f32[192, 192, 3, 3]", primals_124: "f32[160, 192, 1, 1]", primals_125: "f32[160, 160, 3, 3]", primals_126: "f32[640, 160, 1, 1]", primals_127: "f32[640, 192, 1, 1]", primals_128: "f32[160, 640, 1, 1]", primals_129: "f32[160, 160, 3, 3]", primals_130: "f32[640, 160, 1, 1]", primals_131: "f32[160, 640, 1, 1]", primals_132: "f32[160, 160, 3, 3]", primals_133: "f32[640, 160, 1, 1]", primals_134: "f32[160, 640, 1, 1]", primals_135: "f32[160, 160, 3, 3]", primals_136: "f32[640, 160, 1, 1]", primals_137: "f32[160, 640, 1, 1]", primals_138: "f32[160, 160, 3, 3]", primals_139: "f32[640, 160, 1, 1]", primals_140: "f32[160, 640, 1, 1]", primals_141: "f32[160, 160, 3, 3]", primals_142: "f32[640, 160, 1, 1]", primals_143: "f32[1920, 640, 1, 1]", primals_144: "f32[1920, 1, 3, 3]", primals_145: "f32[640, 1920, 1, 1]", primals_146: "f32[640, 640, 1, 1]", primals_147: "f32[1920, 640, 1, 1]", primals_148: "f32[1920, 1, 3, 3]", primals_149: "f32[640, 1920, 1, 1]", primals_150: "f32[1920, 640, 1, 1]", primals_151: "f32[1920, 1, 3, 3]", primals_152: "f32[640, 1920, 1, 1]", primals_153: "f32[1920, 640, 1, 1]", primals_154: "f32[1920, 1, 3, 3]", primals_155: "f32[640, 1920, 1, 1]", primals_156: "f32[1920, 640, 1, 1]", primals_157: "f32[1920, 1, 3, 3]", primals_158: "f32[640, 1920, 1, 1]", primals_159: "f32[1920, 640, 1, 1]", primals_160: "f32[1920, 1, 3, 3]", primals_161: "f32[640, 1920, 1, 1]", primals_162: "f32[1920, 640, 1, 1]", primals_163: "f32[1920, 1, 3, 3]", primals_164: "f32[640, 1920, 1, 1]", primals_165: "f32[1920, 640, 1, 1]", primals_166: "f32[1920, 1, 3, 3]", primals_167: "f32[640, 1920, 1, 1]", primals_168: "f32[1920, 640, 1, 1]", primals_169: "f32[1920, 1, 3, 3]", primals_170: "f32[640, 1920, 1, 1]", primals_171: "f32[2560, 640, 1, 1]", primals_172: "f32[1000, 2560]", primals_173: "f32[1000]", primals_174: "i64[]", primals_175: "f32[32]", primals_176: "f32[32]", primals_177: "i64[]", primals_178: "f32[128]", primals_179: "f32[128]", primals_180: "i64[]", primals_181: "f32[128]", primals_182: "f32[128]", primals_183: "i64[]", primals_184: "f32[128]", primals_185: "f32[128]", primals_186: "i64[]", primals_187: "f32[192]", primals_188: "f32[192]", primals_189: "i64[]", primals_190: "f32[192]", primals_191: "f32[192]", primals_192: "i64[]", primals_193: "f32[192]", primals_194: "f32[192]", primals_195: "i64[]", primals_196: "f32[192]", primals_197: "f32[192]", primals_198: "i64[]", primals_199: "f32[192]", primals_200: "f32[192]", primals_201: "i64[]", primals_202: "f32[160]", primals_203: "f32[160]", primals_204: "i64[]", primals_205: "f32[160]", primals_206: "f32[160]", primals_207: "i64[]", primals_208: "f32[640]", primals_209: "f32[640]", primals_210: "i64[]", primals_211: "f32[640]", primals_212: "f32[640]", primals_213: "i64[]", primals_214: "f32[160]", primals_215: "f32[160]", primals_216: "i64[]", primals_217: "f32[160]", primals_218: "f32[160]", primals_219: "i64[]", primals_220: "f32[640]", primals_221: "f32[640]", primals_222: "i64[]", primals_223: "f32[160]", primals_224: "f32[160]", primals_225: "i64[]", primals_226: "f32[160]", primals_227: "f32[160]", primals_228: "i64[]", primals_229: "f32[640]", primals_230: "f32[640]", primals_231: "i64[]", primals_232: "f32[160]", primals_233: "f32[160]", primals_234: "i64[]", primals_235: "f32[160]", primals_236: "f32[160]", primals_237: "i64[]", primals_238: "f32[640]", primals_239: "f32[640]", primals_240: "i64[]", primals_241: "f32[160]", primals_242: "f32[160]", primals_243: "i64[]", primals_244: "f32[160]", primals_245: "f32[160]", primals_246: "i64[]", primals_247: "f32[640]", primals_248: "f32[640]", primals_249: "i64[]", primals_250: "f32[160]", primals_251: "f32[160]", primals_252: "i64[]", primals_253: "f32[160]", primals_254: "f32[160]", primals_255: "i64[]", primals_256: "f32[640]", primals_257: "f32[640]", primals_258: "i64[]", primals_259: "f32[1920]", primals_260: "f32[1920]", primals_261: "i64[]", primals_262: "f32[1920]", primals_263: "f32[1920]", primals_264: "i64[]", primals_265: "f32[640]", primals_266: "f32[640]", primals_267: "i64[]", primals_268: "f32[640]", primals_269: "f32[640]", primals_270: "i64[]", primals_271: "f32[1920]", primals_272: "f32[1920]", primals_273: "i64[]", primals_274: "f32[1920]", primals_275: "f32[1920]", primals_276: "i64[]", primals_277: "f32[640]", primals_278: "f32[640]", primals_279: "i64[]", primals_280: "f32[1920]", primals_281: "f32[1920]", primals_282: "i64[]", primals_283: "f32[1920]", primals_284: "f32[1920]", primals_285: "i64[]", primals_286: "f32[640]", primals_287: "f32[640]", primals_288: "i64[]", primals_289: "f32[1920]", primals_290: "f32[1920]", primals_291: "i64[]", primals_292: "f32[1920]", primals_293: "f32[1920]", primals_294: "i64[]", primals_295: "f32[640]", primals_296: "f32[640]", primals_297: "i64[]", primals_298: "f32[1920]", primals_299: "f32[1920]", primals_300: "i64[]", primals_301: "f32[1920]", primals_302: "f32[1920]", primals_303: "i64[]", primals_304: "f32[640]", primals_305: "f32[640]", primals_306: "i64[]", primals_307: "f32[1920]", primals_308: "f32[1920]", primals_309: "i64[]", primals_310: "f32[1920]", primals_311: "f32[1920]", primals_312: "i64[]", primals_313: "f32[640]", primals_314: "f32[640]", primals_315: "i64[]", primals_316: "f32[1920]", primals_317: "f32[1920]", primals_318: "i64[]", primals_319: "f32[1920]", primals_320: "f32[1920]", primals_321: "i64[]", primals_322: "f32[640]", primals_323: "f32[640]", primals_324: "i64[]", primals_325: "f32[1920]", primals_326: "f32[1920]", primals_327: "i64[]", primals_328: "f32[1920]", primals_329: "f32[1920]", primals_330: "i64[]", primals_331: "f32[640]", primals_332: "f32[640]", primals_333: "i64[]", primals_334: "f32[1920]", primals_335: "f32[1920]", primals_336: "i64[]", primals_337: "f32[1920]", primals_338: "f32[1920]", primals_339: "i64[]", primals_340: "f32[640]", primals_341: "f32[640]", primals_342: "i64[]", primals_343: "f32[2560]", primals_344: "f32[2560]", primals_345: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 128, 128]" = torch.ops.aten.convolution.default(primals_345, primals_115, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_174, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 128, 128]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu, primals_116, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
    add_7: "f32[128]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.000030518509476);  squeeze_5 = None
    mul_11: "f32[128]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_1, primals_117, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_180, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[128]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.000030518509476);  squeeze_8 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
    add_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu, primals_118, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_21: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[128]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
    add_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
    mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
    add_18: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:267, code: x = x + self.shortcut(shortcut)
    add_20: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(add_14, add_19);  add_14 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    relu_2: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 192, 32, 32]" = torch.ops.aten.convolution.default(relu_2, primals_119, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_186, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 192, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 192, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_22: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_4: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_28: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[192]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
    add_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0001220852154804);  squeeze_14 = None
    mul_32: "f32[192]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[192]" = torch.ops.aten.mul.Tensor(primals_188, 0.9)
    add_24: "f32[192]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_25: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 192, 32, 32]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 192, 32, 32]" = torch.ops.aten.convolution.default(relu_3, primals_120, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 192, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 192, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_35: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[192]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
    add_28: "f32[192]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001220852154804);  squeeze_17 = None
    mul_39: "f32[192]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[192]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
    add_29: "f32[192]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 192, 32, 32]" = torch.ops.aten.convolution.default(relu_2, primals_121, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_192, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 192, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 192, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_42: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[192]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
    add_33: "f32[192]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001220852154804);  squeeze_20 = None
    mul_46: "f32[192]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[192]" = torch.ops.aten.mul.Tensor(primals_194, 0.9)
    add_34: "f32[192]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:267, code: x = x + self.shortcut(shortcut)
    add_36: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(add_30, add_35);  add_30 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    relu_4: "f32[8, 192, 32, 32]" = torch.ops.aten.relu.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 192, 32, 32]" = torch.ops.aten.convolution.default(relu_4, primals_122, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_37: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 192, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 192, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_38: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_7: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_49: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[192]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
    add_39: "f32[192]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001220852154804);  squeeze_23 = None
    mul_53: "f32[192]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[192]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
    add_40: "f32[192]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_41: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 192, 32, 32]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 192, 32, 32]" = torch.ops.aten.convolution.default(relu_5, primals_123, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_198, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 192, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 192, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_43: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_8: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_56: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[192]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
    add_44: "f32[192]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001220852154804);  squeeze_26 = None
    mul_60: "f32[192]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[192]" = torch.ops.aten.mul.Tensor(primals_200, 0.9)
    add_45: "f32[192]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_46: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:267, code: x = x + self.shortcut(shortcut)
    add_47: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(add_46, relu_4);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    relu_6: "f32[8, 192, 32, 32]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 160, 32, 32]" = torch.ops.aten.convolution.default(relu_6, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 160, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 160, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_49: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_9: "f32[8, 160, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_63: "f32[8, 160, 32, 32]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[160]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
    add_50: "f32[160]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_66: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001220852154804);  squeeze_29 = None
    mul_67: "f32[160]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[160]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
    add_51: "f32[160]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 160, 32, 32]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_52: "f32[8, 160, 32, 32]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 160, 32, 32]" = torch.ops.aten.relu.default(add_52);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_7, primals_125, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_204, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 160, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 160, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_54: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_10: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_70: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[160]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
    add_55: "f32[160]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_73: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0004885197850513);  squeeze_32 = None
    mul_74: "f32[160]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[160]" = torch.ops.aten.mul.Tensor(primals_206, 0.9)
    add_56: "f32[160]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_57: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_8, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 640, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 640, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_59: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_11: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_77: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[640]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
    add_60: "f32[640]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_80: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0004885197850513);  squeeze_35 = None
    mul_81: "f32[640]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[640]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
    add_61: "f32[640]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_62: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_6, primals_127, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_210, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 640, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 640, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_64: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_12: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_84: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[640]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
    add_65: "f32[640]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_87: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0004885197850513);  squeeze_38 = None
    mul_88: "f32[640]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[640]" = torch.ops.aten.mul.Tensor(primals_212, 0.9)
    add_66: "f32[640]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_67: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_68: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_62, add_67);  add_62 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_9: "f32[8, 640, 16, 16]" = torch.ops.aten.relu.default(add_68);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_9, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_69: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 160, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 160, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_70: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_13: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_91: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[160]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
    add_71: "f32[160]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_94: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0004885197850513);  squeeze_41 = None
    mul_95: "f32[160]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[160]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
    add_72: "f32[160]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_73: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_10, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_74: "i64[]" = torch.ops.aten.add.Tensor(primals_216, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 160, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 160, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_75: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_14: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_98: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[160]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
    add_76: "f32[160]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_101: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0004885197850513);  squeeze_44 = None
    mul_102: "f32[160]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[160]" = torch.ops.aten.mul.Tensor(primals_218, 0.9)
    add_77: "f32[160]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_78: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_78);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_11, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 640, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 640, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_80: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_15: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_105: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[640]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
    add_81: "f32[640]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_108: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0004885197850513);  squeeze_47 = None
    mul_109: "f32[640]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[640]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
    add_82: "f32[640]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_83: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_84: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_83, relu_9);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_12: "f32[8, 640, 16, 16]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_12, primals_131, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_222, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 160, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 160, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_86: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_16: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_112: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[160]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_87: "f32[160]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_115: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0004885197850513);  squeeze_50 = None
    mul_116: "f32[160]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[160]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_88: "f32[160]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_89: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_89);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_13, primals_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 160, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 160, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_91: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_17: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_119: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[160]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_92: "f32[160]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_122: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0004885197850513);  squeeze_53 = None
    mul_123: "f32[160]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[160]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_93: "f32[160]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_94: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_94);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_14, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 640, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 640, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_96: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_18: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_126: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[640]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_97: "f32[640]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_129: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0004885197850513);  squeeze_56 = None
    mul_130: "f32[640]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[640]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_98: "f32[640]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_99: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_100: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_99, relu_12);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_15: "f32[8, 640, 16, 16]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_15, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 160, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 160, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_102: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_19: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
    mul_133: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[160]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_103: "f32[160]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_136: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0004885197850513);  squeeze_59 = None
    mul_137: "f32[160]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[160]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_104: "f32[160]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_105: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_16, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_106: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 160, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 160, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_107: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_20: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_41)
    mul_140: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[160]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_108: "f32[160]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_143: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0004885197850513);  squeeze_62 = None
    mul_144: "f32[160]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[160]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_109: "f32[160]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_110: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_110);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_17, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 640, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 640, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_112: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_21: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
    mul_147: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[640]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_113: "f32[640]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_150: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0004885197850513);  squeeze_65 = None
    mul_151: "f32[640]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[640]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_114: "f32[640]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_115: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_116: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_115, relu_15);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_18: "f32[8, 640, 16, 16]" = torch.ops.aten.relu.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_18, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 160, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 160, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_22: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
    mul_154: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[160]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_119: "f32[160]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_157: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0004885197850513);  squeeze_68 = None
    mul_158: "f32[160]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[160]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_120: "f32[160]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_19, primals_138, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 160, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 160, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_23: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_47)
    mul_161: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[160]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_124: "f32[160]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_164: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0004885197850513);  squeeze_71 = None
    mul_165: "f32[160]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[160]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_125: "f32[160]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_126);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_20, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 640, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 640, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_128: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_24: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_49)
    mul_168: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[640]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_129: "f32[640]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_171: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0004885197850513);  squeeze_74 = None
    mul_172: "f32[640]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[640]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_130: "f32[640]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_131: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_132: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_131, relu_18);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_21: "f32[8, 640, 16, 16]" = torch.ops.aten.relu.default(add_132);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_21, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 160, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 160, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_25: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_51)
    mul_175: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[160]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_135: "f32[160]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_178: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0004885197850513);  squeeze_77 = None
    mul_179: "f32[160]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[160]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_136: "f32[160]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_137: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_137);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 160, 16, 16]" = torch.ops.aten.convolution.default(relu_22, primals_141, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 160, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 160, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_139: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_26: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_53)
    mul_182: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[160]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_140: "f32[160]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_185: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0004885197850513);  squeeze_80 = None
    mul_186: "f32[160]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[160]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_141: "f32[160]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_142: "f32[8, 160, 16, 16]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_23: "f32[8, 160, 16, 16]" = torch.ops.aten.relu.default(add_142);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 640, 16, 16]" = torch.ops.aten.convolution.default(relu_23, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 640, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 640, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_144: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_27: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_55)
    mul_189: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[640]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_145: "f32[640]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_192: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0004885197850513);  squeeze_83 = None
    mul_193: "f32[640]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[640]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_146: "f32[640]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_147: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_148: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(add_147, relu_21);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_24: "f32[8, 640, 16, 16]" = torch.ops.aten.relu.default(add_148);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 1920, 16, 16]" = torch.ops.aten.convolution.default(relu_24, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_149: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 1920, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 1920, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_150: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_28: "f32[8, 1920, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_57)
    mul_196: "f32[8, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_151: "f32[1920]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_199: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0004885197850513);  squeeze_86 = None
    mul_200: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_152: "f32[1920]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_153: "f32[8, 1920, 16, 16]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 1920, 16, 16]" = torch.ops.aten.relu.default(add_153);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_25, primals_144, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 1920, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 1920, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_155: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_29: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_59)
    mul_203: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_156: "f32[1920]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_206: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0019569471624266);  squeeze_89 = None
    mul_207: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_157: "f32[1920]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_158: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_158);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_26, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 640, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 640, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_160: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_30: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_61)
    mul_210: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[640]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_161: "f32[640]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_213: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0019569471624266);  squeeze_92 = None
    mul_214: "f32[640]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[640]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_162: "f32[640]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_163: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_24, primals_146, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_164: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 640, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 640, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_165: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_31: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_63)
    mul_217: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[640]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_166: "f32[640]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_220: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0019569471624266);  squeeze_95 = None
    mul_221: "f32[640]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[640]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_167: "f32[640]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_168: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_169: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_163, add_168);  add_163 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_27: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_169);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_27, primals_147, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 1920, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 1920, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_171: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_32: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_65)
    mul_224: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_172: "f32[1920]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_227: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0019569471624266);  squeeze_98 = None
    mul_228: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_173: "f32[1920]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_174: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_28: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_28, primals_148, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 1920, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 1920, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_176: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_33: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_67)
    mul_231: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_177: "f32[1920]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_234: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0019569471624266);  squeeze_101 = None
    mul_235: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_178: "f32[1920]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_179: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_29, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 640, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 640, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_181: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_34: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_69)
    mul_238: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[640]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_182: "f32[640]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_241: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0019569471624266);  squeeze_104 = None
    mul_242: "f32[640]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[640]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_183: "f32[640]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_184: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_185: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_184, relu_27);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_30: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_185);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_30, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 1920, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 1920, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_187: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_35: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_71)
    mul_245: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_188: "f32[1920]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_248: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0019569471624266);  squeeze_107 = None
    mul_249: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_189: "f32[1920]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_190: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_31: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_190);  add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_36: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_31, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 1920, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 1920, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_192: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_36: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_73)
    mul_252: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_193: "f32[1920]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_255: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0019569471624266);  squeeze_110 = None
    mul_256: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_194: "f32[1920]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_195: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_32: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_195);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_37: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_32, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 640, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 640, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_197: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_37: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_75)
    mul_259: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[640]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_198: "f32[640]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_262: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0019569471624266);  squeeze_113 = None
    mul_263: "f32[640]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[640]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_199: "f32[640]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_200: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_201: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_200, relu_30);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_33: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_201);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_33, primals_153, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1920, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 1920, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_203: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_38: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_77)
    mul_266: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_204: "f32[1920]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_269: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0019569471624266);  squeeze_116 = None
    mul_270: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_205: "f32[1920]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_206: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_206);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_34, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 1920, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 1920, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_208: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_39: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_79)
    mul_273: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_209: "f32[1920]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_276: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0019569471624266);  squeeze_119 = None
    mul_277: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_210: "f32[1920]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_211: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_35: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_35, primals_155, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 640, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 640, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_213: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_40: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_81)
    mul_280: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[640]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_214: "f32[640]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_283: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0019569471624266);  squeeze_122 = None
    mul_284: "f32[640]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[640]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_215: "f32[640]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_216: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_217: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_216, relu_33);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_36: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_217);  add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_36, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_218: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 1920, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 1920, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_219: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
    sub_41: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_83)
    mul_287: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_220: "f32[1920]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_290: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0019569471624266);  squeeze_125 = None
    mul_291: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_221: "f32[1920]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_222: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_222);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_37, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1920, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 1920, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_224: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_42: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_85)
    mul_294: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_225: "f32[1920]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_297: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0019569471624266);  squeeze_128 = None
    mul_298: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_226: "f32[1920]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_227: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_227);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_38, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 640, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 640, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_229: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    sub_43: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_87)
    mul_301: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[640]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_230: "f32[640]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_304: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0019569471624266);  squeeze_131 = None
    mul_305: "f32[640]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[640]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_231: "f32[640]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_232: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_233: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_232, relu_36);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_39: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_233);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_39, primals_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 1920, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 1920, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_235: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_44: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_89)
    mul_308: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_236: "f32[1920]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_311: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0019569471624266);  squeeze_134 = None
    mul_312: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_237: "f32[1920]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_238: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_40: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_238);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_40, primals_160, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_239: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 1920, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 1920, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_240: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_45: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_91)
    mul_315: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_241: "f32[1920]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_318: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0019569471624266);  squeeze_137 = None
    mul_319: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_242: "f32[1920]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_243: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_243);  add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_41, primals_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 640, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 640, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_245: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_46: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_93)
    mul_322: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[640]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_246: "f32[640]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_325: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0019569471624266);  squeeze_140 = None
    mul_326: "f32[640]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[640]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_247: "f32[640]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_248: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_249: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_248, relu_39);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_42: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_249);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_47: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_42, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 1920, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 1920, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_251: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_47: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_95)
    mul_329: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_252: "f32[1920]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_332: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0019569471624266);  squeeze_143 = None
    mul_333: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_253: "f32[1920]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_254: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_43: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_254);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_43, primals_163, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 1920, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 1920, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_256: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_48: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_97)
    mul_336: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_257: "f32[1920]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_339: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0019569471624266);  squeeze_146 = None
    mul_340: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_258: "f32[1920]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_259: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_44: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_259);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_44, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 640, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 640, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_261: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_49: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_99)
    mul_343: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[640]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_262: "f32[640]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_346: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0019569471624266);  squeeze_149 = None
    mul_347: "f32[640]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[640]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_263: "f32[640]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_264: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_265: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_264, relu_42);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_45: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_45, primals_165, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_266: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 1920, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 1920, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_267: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_50: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_101)
    mul_350: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_268: "f32[1920]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_353: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0019569471624266);  squeeze_152 = None
    mul_354: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_269: "f32[1920]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_270: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_270);  add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_46, primals_166, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_271: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1920, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 1920, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_272: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    sub_51: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
    mul_357: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_273: "f32[1920]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_360: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0019569471624266);  squeeze_155 = None
    mul_361: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_274: "f32[1920]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_275: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_47: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_275);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_52: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_47, primals_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_276: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 640, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 640, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_277: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_52: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
    sub_52: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_105)
    mul_364: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[640]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_278: "f32[640]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_367: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0019569471624266);  squeeze_158 = None
    mul_368: "f32[640]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[640]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_279: "f32[640]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_280: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_281: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_280, relu_45);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_48: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_281);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_48, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_282: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 1920, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 1920, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_283: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_53: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
    sub_53: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_107)
    mul_371: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_284: "f32[1920]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_374: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0019569471624266);  squeeze_161 = None
    mul_375: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_285: "f32[1920]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_286: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_286);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 1920, 8, 8]" = torch.ops.aten.convolution.default(relu_49, primals_169, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1920)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_287: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1920, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 1920, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_288: "f32[1, 1920, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_54: "f32[1, 1920, 1, 1]" = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
    sub_54: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_109)
    mul_378: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[1920]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_289: "f32[1920]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[1920]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_381: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0019569471624266);  squeeze_164 = None
    mul_382: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[1920]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_290: "f32[1920]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[1920, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_291: "f32[8, 1920, 8, 8]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[8, 1920, 8, 8]" = torch.ops.aten.relu.default(add_291);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[8, 640, 8, 8]" = torch.ops.aten.convolution.default(relu_50, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_292: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 640, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 640, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_293: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_55: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
    sub_55: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_111)
    mul_385: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[640]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_294: "f32[640]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_388: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0019569471624266);  squeeze_167 = None
    mul_389: "f32[640]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[640]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_295: "f32[640]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_296: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:336, code: x = x + self.shortcut(shortcut)
    add_297: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(add_296, relu_48);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    relu_51: "f32[8, 640, 8, 8]" = torch.ops.aten.relu.default(add_297);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 2560, 8, 8]" = torch.ops.aten.convolution.default(relu_51, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_298: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 2560, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 2560, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_299: "f32[1, 2560, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_56: "f32[1, 2560, 1, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
    sub_56: "f32[8, 2560, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_113)
    mul_392: "f32[8, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[2560]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[2560]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[2560]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[2560]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_300: "f32[2560]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[2560]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_395: "f32[2560]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0019569471624266);  squeeze_170 = None
    mul_396: "f32[2560]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[2560]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_301: "f32[2560]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[2560, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_302: "f32[8, 2560, 8, 8]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_52: "f32[8, 2560, 8, 8]" = torch.ops.aten.relu.default(add_302);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2560, 1, 1]" = torch.ops.aten.mean.dim(relu_52, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2560]" = torch.ops.aten.view.default(mean, [8, 2560]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone: "f32[8, 2560]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[2560, 1000]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_173, clone, permute);  primals_173 = None
    permute_1: "f32[1000, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 2560, 8, 8]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_55: "f32[8, 2560, 8, 8]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le: "b8[8, 2560, 8, 8]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_228: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_229: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_240: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_241: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_252: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_253: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_264: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_265: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_276: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_277: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_288: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_289: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_300: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_301: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_312: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_313: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_324: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_325: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_336: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_337: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_348: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_349: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_360: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_361: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_372: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_373: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_384: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_385: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_396: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_397: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_409: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_420: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_421: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_432: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_433: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_444: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_445: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
    unsqueeze_446: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_456: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_457: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
    unsqueeze_458: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_468: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_469: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
    unsqueeze_470: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_480: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_481: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
    unsqueeze_482: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_492: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_493: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
    unsqueeze_494: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_504: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_505: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
    unsqueeze_506: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_516: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_517: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
    unsqueeze_518: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_528: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_529: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
    unsqueeze_530: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_540: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_541: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
    unsqueeze_542: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_552: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_553: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
    unsqueeze_554: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_564: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_565: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
    unsqueeze_566: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_576: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_577: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
    unsqueeze_578: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_588: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_589: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
    unsqueeze_590: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_600: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_601: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
    unsqueeze_602: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_612: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_613: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
    unsqueeze_614: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_624: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_625: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
    unsqueeze_626: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_636: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_637: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
    unsqueeze_638: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_648: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_649: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
    unsqueeze_650: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_660: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_661: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
    unsqueeze_662: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_672: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_673: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
    unsqueeze_674: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_684: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_685: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
    unsqueeze_686: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_696: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_697: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
    unsqueeze_698: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_708: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_709: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
    unsqueeze_710: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_720: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_721: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
    unsqueeze_722: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_732: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_733: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
    unsqueeze_734: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_744: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_745: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
    unsqueeze_746: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_756: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_757: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
    unsqueeze_758: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_768: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_769: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
    unsqueeze_770: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_780: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_781: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
    unsqueeze_782: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_792: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_793: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
    unsqueeze_794: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_804: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_805: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
    unsqueeze_806: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_816: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_817: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
    unsqueeze_818: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_828: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_829: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
    unsqueeze_830: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_840: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_841: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
    unsqueeze_842: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_852: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_853: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
    unsqueeze_854: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_864: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_865: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
    unsqueeze_866: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_876: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_877: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
    unsqueeze_878: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_888: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_889: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
    unsqueeze_890: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_900: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_901: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
    unsqueeze_902: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_174, add);  primals_174 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_175, add_2);  primals_175 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_176, add_3);  primals_176 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_177, add_5);  primals_177 = add_5 = None
    copy__4: "f32[128]" = torch.ops.aten.copy_.default(primals_178, add_7);  primals_178 = add_7 = None
    copy__5: "f32[128]" = torch.ops.aten.copy_.default(primals_179, add_8);  primals_179 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_180, add_10);  primals_180 = add_10 = None
    copy__7: "f32[128]" = torch.ops.aten.copy_.default(primals_181, add_12);  primals_181 = add_12 = None
    copy__8: "f32[128]" = torch.ops.aten.copy_.default(primals_182, add_13);  primals_182 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_183, add_15);  primals_183 = add_15 = None
    copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_184, add_17);  primals_184 = add_17 = None
    copy__11: "f32[128]" = torch.ops.aten.copy_.default(primals_185, add_18);  primals_185 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_186, add_21);  primals_186 = add_21 = None
    copy__13: "f32[192]" = torch.ops.aten.copy_.default(primals_187, add_23);  primals_187 = add_23 = None
    copy__14: "f32[192]" = torch.ops.aten.copy_.default(primals_188, add_24);  primals_188 = add_24 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_189, add_26);  primals_189 = add_26 = None
    copy__16: "f32[192]" = torch.ops.aten.copy_.default(primals_190, add_28);  primals_190 = add_28 = None
    copy__17: "f32[192]" = torch.ops.aten.copy_.default(primals_191, add_29);  primals_191 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_192, add_31);  primals_192 = add_31 = None
    copy__19: "f32[192]" = torch.ops.aten.copy_.default(primals_193, add_33);  primals_193 = add_33 = None
    copy__20: "f32[192]" = torch.ops.aten.copy_.default(primals_194, add_34);  primals_194 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_195, add_37);  primals_195 = add_37 = None
    copy__22: "f32[192]" = torch.ops.aten.copy_.default(primals_196, add_39);  primals_196 = add_39 = None
    copy__23: "f32[192]" = torch.ops.aten.copy_.default(primals_197, add_40);  primals_197 = add_40 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_198, add_42);  primals_198 = add_42 = None
    copy__25: "f32[192]" = torch.ops.aten.copy_.default(primals_199, add_44);  primals_199 = add_44 = None
    copy__26: "f32[192]" = torch.ops.aten.copy_.default(primals_200, add_45);  primals_200 = add_45 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_48);  primals_201 = add_48 = None
    copy__28: "f32[160]" = torch.ops.aten.copy_.default(primals_202, add_50);  primals_202 = add_50 = None
    copy__29: "f32[160]" = torch.ops.aten.copy_.default(primals_203, add_51);  primals_203 = add_51 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_204, add_53);  primals_204 = add_53 = None
    copy__31: "f32[160]" = torch.ops.aten.copy_.default(primals_205, add_55);  primals_205 = add_55 = None
    copy__32: "f32[160]" = torch.ops.aten.copy_.default(primals_206, add_56);  primals_206 = add_56 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_58);  primals_207 = add_58 = None
    copy__34: "f32[640]" = torch.ops.aten.copy_.default(primals_208, add_60);  primals_208 = add_60 = None
    copy__35: "f32[640]" = torch.ops.aten.copy_.default(primals_209, add_61);  primals_209 = add_61 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_210, add_63);  primals_210 = add_63 = None
    copy__37: "f32[640]" = torch.ops.aten.copy_.default(primals_211, add_65);  primals_211 = add_65 = None
    copy__38: "f32[640]" = torch.ops.aten.copy_.default(primals_212, add_66);  primals_212 = add_66 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_69);  primals_213 = add_69 = None
    copy__40: "f32[160]" = torch.ops.aten.copy_.default(primals_214, add_71);  primals_214 = add_71 = None
    copy__41: "f32[160]" = torch.ops.aten.copy_.default(primals_215, add_72);  primals_215 = add_72 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_216, add_74);  primals_216 = add_74 = None
    copy__43: "f32[160]" = torch.ops.aten.copy_.default(primals_217, add_76);  primals_217 = add_76 = None
    copy__44: "f32[160]" = torch.ops.aten.copy_.default(primals_218, add_77);  primals_218 = add_77 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_79);  primals_219 = add_79 = None
    copy__46: "f32[640]" = torch.ops.aten.copy_.default(primals_220, add_81);  primals_220 = add_81 = None
    copy__47: "f32[640]" = torch.ops.aten.copy_.default(primals_221, add_82);  primals_221 = add_82 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_222, add_85);  primals_222 = add_85 = None
    copy__49: "f32[160]" = torch.ops.aten.copy_.default(primals_223, add_87);  primals_223 = add_87 = None
    copy__50: "f32[160]" = torch.ops.aten.copy_.default(primals_224, add_88);  primals_224 = add_88 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_90);  primals_225 = add_90 = None
    copy__52: "f32[160]" = torch.ops.aten.copy_.default(primals_226, add_92);  primals_226 = add_92 = None
    copy__53: "f32[160]" = torch.ops.aten.copy_.default(primals_227, add_93);  primals_227 = add_93 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_95);  primals_228 = add_95 = None
    copy__55: "f32[640]" = torch.ops.aten.copy_.default(primals_229, add_97);  primals_229 = add_97 = None
    copy__56: "f32[640]" = torch.ops.aten.copy_.default(primals_230, add_98);  primals_230 = add_98 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_101);  primals_231 = add_101 = None
    copy__58: "f32[160]" = torch.ops.aten.copy_.default(primals_232, add_103);  primals_232 = add_103 = None
    copy__59: "f32[160]" = torch.ops.aten.copy_.default(primals_233, add_104);  primals_233 = add_104 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_106);  primals_234 = add_106 = None
    copy__61: "f32[160]" = torch.ops.aten.copy_.default(primals_235, add_108);  primals_235 = add_108 = None
    copy__62: "f32[160]" = torch.ops.aten.copy_.default(primals_236, add_109);  primals_236 = add_109 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_111);  primals_237 = add_111 = None
    copy__64: "f32[640]" = torch.ops.aten.copy_.default(primals_238, add_113);  primals_238 = add_113 = None
    copy__65: "f32[640]" = torch.ops.aten.copy_.default(primals_239, add_114);  primals_239 = add_114 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_117);  primals_240 = add_117 = None
    copy__67: "f32[160]" = torch.ops.aten.copy_.default(primals_241, add_119);  primals_241 = add_119 = None
    copy__68: "f32[160]" = torch.ops.aten.copy_.default(primals_242, add_120);  primals_242 = add_120 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_122);  primals_243 = add_122 = None
    copy__70: "f32[160]" = torch.ops.aten.copy_.default(primals_244, add_124);  primals_244 = add_124 = None
    copy__71: "f32[160]" = torch.ops.aten.copy_.default(primals_245, add_125);  primals_245 = add_125 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_127);  primals_246 = add_127 = None
    copy__73: "f32[640]" = torch.ops.aten.copy_.default(primals_247, add_129);  primals_247 = add_129 = None
    copy__74: "f32[640]" = torch.ops.aten.copy_.default(primals_248, add_130);  primals_248 = add_130 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_133);  primals_249 = add_133 = None
    copy__76: "f32[160]" = torch.ops.aten.copy_.default(primals_250, add_135);  primals_250 = add_135 = None
    copy__77: "f32[160]" = torch.ops.aten.copy_.default(primals_251, add_136);  primals_251 = add_136 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_138);  primals_252 = add_138 = None
    copy__79: "f32[160]" = torch.ops.aten.copy_.default(primals_253, add_140);  primals_253 = add_140 = None
    copy__80: "f32[160]" = torch.ops.aten.copy_.default(primals_254, add_141);  primals_254 = add_141 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_143);  primals_255 = add_143 = None
    copy__82: "f32[640]" = torch.ops.aten.copy_.default(primals_256, add_145);  primals_256 = add_145 = None
    copy__83: "f32[640]" = torch.ops.aten.copy_.default(primals_257, add_146);  primals_257 = add_146 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_149);  primals_258 = add_149 = None
    copy__85: "f32[1920]" = torch.ops.aten.copy_.default(primals_259, add_151);  primals_259 = add_151 = None
    copy__86: "f32[1920]" = torch.ops.aten.copy_.default(primals_260, add_152);  primals_260 = add_152 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_154);  primals_261 = add_154 = None
    copy__88: "f32[1920]" = torch.ops.aten.copy_.default(primals_262, add_156);  primals_262 = add_156 = None
    copy__89: "f32[1920]" = torch.ops.aten.copy_.default(primals_263, add_157);  primals_263 = add_157 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_159);  primals_264 = add_159 = None
    copy__91: "f32[640]" = torch.ops.aten.copy_.default(primals_265, add_161);  primals_265 = add_161 = None
    copy__92: "f32[640]" = torch.ops.aten.copy_.default(primals_266, add_162);  primals_266 = add_162 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_164);  primals_267 = add_164 = None
    copy__94: "f32[640]" = torch.ops.aten.copy_.default(primals_268, add_166);  primals_268 = add_166 = None
    copy__95: "f32[640]" = torch.ops.aten.copy_.default(primals_269, add_167);  primals_269 = add_167 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_170);  primals_270 = add_170 = None
    copy__97: "f32[1920]" = torch.ops.aten.copy_.default(primals_271, add_172);  primals_271 = add_172 = None
    copy__98: "f32[1920]" = torch.ops.aten.copy_.default(primals_272, add_173);  primals_272 = add_173 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_175);  primals_273 = add_175 = None
    copy__100: "f32[1920]" = torch.ops.aten.copy_.default(primals_274, add_177);  primals_274 = add_177 = None
    copy__101: "f32[1920]" = torch.ops.aten.copy_.default(primals_275, add_178);  primals_275 = add_178 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_180);  primals_276 = add_180 = None
    copy__103: "f32[640]" = torch.ops.aten.copy_.default(primals_277, add_182);  primals_277 = add_182 = None
    copy__104: "f32[640]" = torch.ops.aten.copy_.default(primals_278, add_183);  primals_278 = add_183 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_186);  primals_279 = add_186 = None
    copy__106: "f32[1920]" = torch.ops.aten.copy_.default(primals_280, add_188);  primals_280 = add_188 = None
    copy__107: "f32[1920]" = torch.ops.aten.copy_.default(primals_281, add_189);  primals_281 = add_189 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_191);  primals_282 = add_191 = None
    copy__109: "f32[1920]" = torch.ops.aten.copy_.default(primals_283, add_193);  primals_283 = add_193 = None
    copy__110: "f32[1920]" = torch.ops.aten.copy_.default(primals_284, add_194);  primals_284 = add_194 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_196);  primals_285 = add_196 = None
    copy__112: "f32[640]" = torch.ops.aten.copy_.default(primals_286, add_198);  primals_286 = add_198 = None
    copy__113: "f32[640]" = torch.ops.aten.copy_.default(primals_287, add_199);  primals_287 = add_199 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_202);  primals_288 = add_202 = None
    copy__115: "f32[1920]" = torch.ops.aten.copy_.default(primals_289, add_204);  primals_289 = add_204 = None
    copy__116: "f32[1920]" = torch.ops.aten.copy_.default(primals_290, add_205);  primals_290 = add_205 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_207);  primals_291 = add_207 = None
    copy__118: "f32[1920]" = torch.ops.aten.copy_.default(primals_292, add_209);  primals_292 = add_209 = None
    copy__119: "f32[1920]" = torch.ops.aten.copy_.default(primals_293, add_210);  primals_293 = add_210 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_212);  primals_294 = add_212 = None
    copy__121: "f32[640]" = torch.ops.aten.copy_.default(primals_295, add_214);  primals_295 = add_214 = None
    copy__122: "f32[640]" = torch.ops.aten.copy_.default(primals_296, add_215);  primals_296 = add_215 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_218);  primals_297 = add_218 = None
    copy__124: "f32[1920]" = torch.ops.aten.copy_.default(primals_298, add_220);  primals_298 = add_220 = None
    copy__125: "f32[1920]" = torch.ops.aten.copy_.default(primals_299, add_221);  primals_299 = add_221 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_223);  primals_300 = add_223 = None
    copy__127: "f32[1920]" = torch.ops.aten.copy_.default(primals_301, add_225);  primals_301 = add_225 = None
    copy__128: "f32[1920]" = torch.ops.aten.copy_.default(primals_302, add_226);  primals_302 = add_226 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_228);  primals_303 = add_228 = None
    copy__130: "f32[640]" = torch.ops.aten.copy_.default(primals_304, add_230);  primals_304 = add_230 = None
    copy__131: "f32[640]" = torch.ops.aten.copy_.default(primals_305, add_231);  primals_305 = add_231 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_234);  primals_306 = add_234 = None
    copy__133: "f32[1920]" = torch.ops.aten.copy_.default(primals_307, add_236);  primals_307 = add_236 = None
    copy__134: "f32[1920]" = torch.ops.aten.copy_.default(primals_308, add_237);  primals_308 = add_237 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_239);  primals_309 = add_239 = None
    copy__136: "f32[1920]" = torch.ops.aten.copy_.default(primals_310, add_241);  primals_310 = add_241 = None
    copy__137: "f32[1920]" = torch.ops.aten.copy_.default(primals_311, add_242);  primals_311 = add_242 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_244);  primals_312 = add_244 = None
    copy__139: "f32[640]" = torch.ops.aten.copy_.default(primals_313, add_246);  primals_313 = add_246 = None
    copy__140: "f32[640]" = torch.ops.aten.copy_.default(primals_314, add_247);  primals_314 = add_247 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_250);  primals_315 = add_250 = None
    copy__142: "f32[1920]" = torch.ops.aten.copy_.default(primals_316, add_252);  primals_316 = add_252 = None
    copy__143: "f32[1920]" = torch.ops.aten.copy_.default(primals_317, add_253);  primals_317 = add_253 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_255);  primals_318 = add_255 = None
    copy__145: "f32[1920]" = torch.ops.aten.copy_.default(primals_319, add_257);  primals_319 = add_257 = None
    copy__146: "f32[1920]" = torch.ops.aten.copy_.default(primals_320, add_258);  primals_320 = add_258 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_260);  primals_321 = add_260 = None
    copy__148: "f32[640]" = torch.ops.aten.copy_.default(primals_322, add_262);  primals_322 = add_262 = None
    copy__149: "f32[640]" = torch.ops.aten.copy_.default(primals_323, add_263);  primals_323 = add_263 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_266);  primals_324 = add_266 = None
    copy__151: "f32[1920]" = torch.ops.aten.copy_.default(primals_325, add_268);  primals_325 = add_268 = None
    copy__152: "f32[1920]" = torch.ops.aten.copy_.default(primals_326, add_269);  primals_326 = add_269 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_271);  primals_327 = add_271 = None
    copy__154: "f32[1920]" = torch.ops.aten.copy_.default(primals_328, add_273);  primals_328 = add_273 = None
    copy__155: "f32[1920]" = torch.ops.aten.copy_.default(primals_329, add_274);  primals_329 = add_274 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_276);  primals_330 = add_276 = None
    copy__157: "f32[640]" = torch.ops.aten.copy_.default(primals_331, add_278);  primals_331 = add_278 = None
    copy__158: "f32[640]" = torch.ops.aten.copy_.default(primals_332, add_279);  primals_332 = add_279 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_282);  primals_333 = add_282 = None
    copy__160: "f32[1920]" = torch.ops.aten.copy_.default(primals_334, add_284);  primals_334 = add_284 = None
    copy__161: "f32[1920]" = torch.ops.aten.copy_.default(primals_335, add_285);  primals_335 = add_285 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_287);  primals_336 = add_287 = None
    copy__163: "f32[1920]" = torch.ops.aten.copy_.default(primals_337, add_289);  primals_337 = add_289 = None
    copy__164: "f32[1920]" = torch.ops.aten.copy_.default(primals_338, add_290);  primals_338 = add_290 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_292);  primals_339 = add_292 = None
    copy__166: "f32[640]" = torch.ops.aten.copy_.default(primals_340, add_294);  primals_340 = add_294 = None
    copy__167: "f32[640]" = torch.ops.aten.copy_.default(primals_341, add_295);  primals_341 = add_295 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_298);  primals_342 = add_298 = None
    copy__169: "f32[2560]" = torch.ops.aten.copy_.default(primals_343, add_300);  primals_343 = add_300 = None
    copy__170: "f32[2560]" = torch.ops.aten.copy_.default(primals_344, add_301);  primals_344 = add_301 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_345, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, clone, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902]
    