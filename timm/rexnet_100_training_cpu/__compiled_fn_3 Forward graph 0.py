from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[16]", primals_6: "f32[16]", primals_7: "f32[96]", primals_8: "f32[96]", primals_9: "f32[96]", primals_10: "f32[96]", primals_11: "f32[27]", primals_12: "f32[27]", primals_13: "f32[162]", primals_14: "f32[162]", primals_15: "f32[162]", primals_16: "f32[162]", primals_17: "f32[38]", primals_18: "f32[38]", primals_19: "f32[228]", primals_20: "f32[228]", primals_21: "f32[228]", primals_22: "f32[228]", primals_23: "f32[50]", primals_24: "f32[50]", primals_25: "f32[300]", primals_26: "f32[300]", primals_27: "f32[300]", primals_28: "f32[300]", primals_29: "f32[61]", primals_30: "f32[61]", primals_31: "f32[366]", primals_32: "f32[366]", primals_33: "f32[366]", primals_34: "f32[366]", primals_35: "f32[72]", primals_36: "f32[72]", primals_37: "f32[432]", primals_38: "f32[432]", primals_39: "f32[432]", primals_40: "f32[432]", primals_41: "f32[84]", primals_42: "f32[84]", primals_43: "f32[504]", primals_44: "f32[504]", primals_45: "f32[504]", primals_46: "f32[504]", primals_47: "f32[95]", primals_48: "f32[95]", primals_49: "f32[570]", primals_50: "f32[570]", primals_51: "f32[570]", primals_52: "f32[570]", primals_53: "f32[106]", primals_54: "f32[106]", primals_55: "f32[636]", primals_56: "f32[636]", primals_57: "f32[636]", primals_58: "f32[636]", primals_59: "f32[117]", primals_60: "f32[117]", primals_61: "f32[702]", primals_62: "f32[702]", primals_63: "f32[702]", primals_64: "f32[702]", primals_65: "f32[128]", primals_66: "f32[128]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[140]", primals_72: "f32[140]", primals_73: "f32[840]", primals_74: "f32[840]", primals_75: "f32[840]", primals_76: "f32[840]", primals_77: "f32[151]", primals_78: "f32[151]", primals_79: "f32[906]", primals_80: "f32[906]", primals_81: "f32[906]", primals_82: "f32[906]", primals_83: "f32[162]", primals_84: "f32[162]", primals_85: "f32[972]", primals_86: "f32[972]", primals_87: "f32[972]", primals_88: "f32[972]", primals_89: "f32[174]", primals_90: "f32[174]", primals_91: "f32[1044]", primals_92: "f32[1044]", primals_93: "f32[1044]", primals_94: "f32[1044]", primals_95: "f32[185]", primals_96: "f32[185]", primals_97: "f32[1280]", primals_98: "f32[1280]", primals_99: "f32[32, 3, 3, 3]", primals_100: "f32[32, 1, 3, 3]", primals_101: "f32[16, 32, 1, 1]", primals_102: "f32[96, 16, 1, 1]", primals_103: "f32[96, 1, 3, 3]", primals_104: "f32[27, 96, 1, 1]", primals_105: "f32[162, 27, 1, 1]", primals_106: "f32[162, 1, 3, 3]", primals_107: "f32[38, 162, 1, 1]", primals_108: "f32[228, 38, 1, 1]", primals_109: "f32[228, 1, 3, 3]", primals_110: "f32[19, 228, 1, 1]", primals_111: "f32[19]", primals_112: "f32[19]", primals_113: "f32[19]", primals_114: "f32[228, 19, 1, 1]", primals_115: "f32[228]", primals_116: "f32[50, 228, 1, 1]", primals_117: "f32[300, 50, 1, 1]", primals_118: "f32[300, 1, 3, 3]", primals_119: "f32[25, 300, 1, 1]", primals_120: "f32[25]", primals_121: "f32[25]", primals_122: "f32[25]", primals_123: "f32[300, 25, 1, 1]", primals_124: "f32[300]", primals_125: "f32[61, 300, 1, 1]", primals_126: "f32[366, 61, 1, 1]", primals_127: "f32[366, 1, 3, 3]", primals_128: "f32[30, 366, 1, 1]", primals_129: "f32[30]", primals_130: "f32[30]", primals_131: "f32[30]", primals_132: "f32[366, 30, 1, 1]", primals_133: "f32[366]", primals_134: "f32[72, 366, 1, 1]", primals_135: "f32[432, 72, 1, 1]", primals_136: "f32[432, 1, 3, 3]", primals_137: "f32[36, 432, 1, 1]", primals_138: "f32[36]", primals_139: "f32[36]", primals_140: "f32[36]", primals_141: "f32[432, 36, 1, 1]", primals_142: "f32[432]", primals_143: "f32[84, 432, 1, 1]", primals_144: "f32[504, 84, 1, 1]", primals_145: "f32[504, 1, 3, 3]", primals_146: "f32[42, 504, 1, 1]", primals_147: "f32[42]", primals_148: "f32[42]", primals_149: "f32[42]", primals_150: "f32[504, 42, 1, 1]", primals_151: "f32[504]", primals_152: "f32[95, 504, 1, 1]", primals_153: "f32[570, 95, 1, 1]", primals_154: "f32[570, 1, 3, 3]", primals_155: "f32[47, 570, 1, 1]", primals_156: "f32[47]", primals_157: "f32[47]", primals_158: "f32[47]", primals_159: "f32[570, 47, 1, 1]", primals_160: "f32[570]", primals_161: "f32[106, 570, 1, 1]", primals_162: "f32[636, 106, 1, 1]", primals_163: "f32[636, 1, 3, 3]", primals_164: "f32[53, 636, 1, 1]", primals_165: "f32[53]", primals_166: "f32[53]", primals_167: "f32[53]", primals_168: "f32[636, 53, 1, 1]", primals_169: "f32[636]", primals_170: "f32[117, 636, 1, 1]", primals_171: "f32[702, 117, 1, 1]", primals_172: "f32[702, 1, 3, 3]", primals_173: "f32[58, 702, 1, 1]", primals_174: "f32[58]", primals_175: "f32[58]", primals_176: "f32[58]", primals_177: "f32[702, 58, 1, 1]", primals_178: "f32[702]", primals_179: "f32[128, 702, 1, 1]", primals_180: "f32[768, 128, 1, 1]", primals_181: "f32[768, 1, 3, 3]", primals_182: "f32[64, 768, 1, 1]", primals_183: "f32[64]", primals_184: "f32[64]", primals_185: "f32[64]", primals_186: "f32[768, 64, 1, 1]", primals_187: "f32[768]", primals_188: "f32[140, 768, 1, 1]", primals_189: "f32[840, 140, 1, 1]", primals_190: "f32[840, 1, 3, 3]", primals_191: "f32[70, 840, 1, 1]", primals_192: "f32[70]", primals_193: "f32[70]", primals_194: "f32[70]", primals_195: "f32[840, 70, 1, 1]", primals_196: "f32[840]", primals_197: "f32[151, 840, 1, 1]", primals_198: "f32[906, 151, 1, 1]", primals_199: "f32[906, 1, 3, 3]", primals_200: "f32[75, 906, 1, 1]", primals_201: "f32[75]", primals_202: "f32[75]", primals_203: "f32[75]", primals_204: "f32[906, 75, 1, 1]", primals_205: "f32[906]", primals_206: "f32[162, 906, 1, 1]", primals_207: "f32[972, 162, 1, 1]", primals_208: "f32[972, 1, 3, 3]", primals_209: "f32[81, 972, 1, 1]", primals_210: "f32[81]", primals_211: "f32[81]", primals_212: "f32[81]", primals_213: "f32[972, 81, 1, 1]", primals_214: "f32[972]", primals_215: "f32[174, 972, 1, 1]", primals_216: "f32[1044, 174, 1, 1]", primals_217: "f32[1044, 1, 3, 3]", primals_218: "f32[87, 1044, 1, 1]", primals_219: "f32[87]", primals_220: "f32[87]", primals_221: "f32[87]", primals_222: "f32[1044, 87, 1, 1]", primals_223: "f32[1044]", primals_224: "f32[185, 1044, 1, 1]", primals_225: "f32[1280, 185, 1, 1]", primals_226: "f32[1000, 1280]", primals_227: "f32[1000]", primals_228: "i64[]", primals_229: "f32[32]", primals_230: "f32[32]", primals_231: "i64[]", primals_232: "f32[32]", primals_233: "f32[32]", primals_234: "i64[]", primals_235: "f32[16]", primals_236: "f32[16]", primals_237: "i64[]", primals_238: "f32[96]", primals_239: "f32[96]", primals_240: "i64[]", primals_241: "f32[96]", primals_242: "f32[96]", primals_243: "i64[]", primals_244: "f32[27]", primals_245: "f32[27]", primals_246: "i64[]", primals_247: "f32[162]", primals_248: "f32[162]", primals_249: "i64[]", primals_250: "f32[162]", primals_251: "f32[162]", primals_252: "i64[]", primals_253: "f32[38]", primals_254: "f32[38]", primals_255: "i64[]", primals_256: "f32[228]", primals_257: "f32[228]", primals_258: "i64[]", primals_259: "f32[228]", primals_260: "f32[228]", primals_261: "i64[]", primals_262: "f32[50]", primals_263: "f32[50]", primals_264: "i64[]", primals_265: "f32[300]", primals_266: "f32[300]", primals_267: "i64[]", primals_268: "f32[300]", primals_269: "f32[300]", primals_270: "i64[]", primals_271: "f32[61]", primals_272: "f32[61]", primals_273: "i64[]", primals_274: "f32[366]", primals_275: "f32[366]", primals_276: "i64[]", primals_277: "f32[366]", primals_278: "f32[366]", primals_279: "i64[]", primals_280: "f32[72]", primals_281: "f32[72]", primals_282: "i64[]", primals_283: "f32[432]", primals_284: "f32[432]", primals_285: "i64[]", primals_286: "f32[432]", primals_287: "f32[432]", primals_288: "i64[]", primals_289: "f32[84]", primals_290: "f32[84]", primals_291: "i64[]", primals_292: "f32[504]", primals_293: "f32[504]", primals_294: "i64[]", primals_295: "f32[504]", primals_296: "f32[504]", primals_297: "i64[]", primals_298: "f32[95]", primals_299: "f32[95]", primals_300: "i64[]", primals_301: "f32[570]", primals_302: "f32[570]", primals_303: "i64[]", primals_304: "f32[570]", primals_305: "f32[570]", primals_306: "i64[]", primals_307: "f32[106]", primals_308: "f32[106]", primals_309: "i64[]", primals_310: "f32[636]", primals_311: "f32[636]", primals_312: "i64[]", primals_313: "f32[636]", primals_314: "f32[636]", primals_315: "i64[]", primals_316: "f32[117]", primals_317: "f32[117]", primals_318: "i64[]", primals_319: "f32[702]", primals_320: "f32[702]", primals_321: "i64[]", primals_322: "f32[702]", primals_323: "f32[702]", primals_324: "i64[]", primals_325: "f32[128]", primals_326: "f32[128]", primals_327: "i64[]", primals_328: "f32[768]", primals_329: "f32[768]", primals_330: "i64[]", primals_331: "f32[768]", primals_332: "f32[768]", primals_333: "i64[]", primals_334: "f32[140]", primals_335: "f32[140]", primals_336: "i64[]", primals_337: "f32[840]", primals_338: "f32[840]", primals_339: "i64[]", primals_340: "f32[840]", primals_341: "f32[840]", primals_342: "i64[]", primals_343: "f32[151]", primals_344: "f32[151]", primals_345: "i64[]", primals_346: "f32[906]", primals_347: "f32[906]", primals_348: "i64[]", primals_349: "f32[906]", primals_350: "f32[906]", primals_351: "i64[]", primals_352: "f32[162]", primals_353: "f32[162]", primals_354: "i64[]", primals_355: "f32[972]", primals_356: "f32[972]", primals_357: "i64[]", primals_358: "f32[972]", primals_359: "f32[972]", primals_360: "i64[]", primals_361: "f32[174]", primals_362: "f32[174]", primals_363: "i64[]", primals_364: "f32[1044]", primals_365: "f32[1044]", primals_366: "i64[]", primals_367: "f32[1044]", primals_368: "f32[1044]", primals_369: "i64[]", primals_370: "f32[185]", primals_371: "f32[185]", primals_372: "i64[]", primals_373: "f32[1280]", primals_374: "f32[1280]", primals_375: "f32[19]", primals_376: "f32[19]", primals_377: "i64[]", primals_378: "f32[25]", primals_379: "f32[25]", primals_380: "i64[]", primals_381: "f32[30]", primals_382: "f32[30]", primals_383: "i64[]", primals_384: "f32[36]", primals_385: "f32[36]", primals_386: "i64[]", primals_387: "f32[42]", primals_388: "f32[42]", primals_389: "i64[]", primals_390: "f32[47]", primals_391: "f32[47]", primals_392: "i64[]", primals_393: "f32[53]", primals_394: "f32[53]", primals_395: "i64[]", primals_396: "f32[58]", primals_397: "f32[58]", primals_398: "i64[]", primals_399: "f32[64]", primals_400: "f32[64]", primals_401: "i64[]", primals_402: "f32[70]", primals_403: "f32[70]", primals_404: "i64[]", primals_405: "f32[75]", primals_406: "f32[75]", primals_407: "i64[]", primals_408: "f32[81]", primals_409: "f32[81]", primals_410: "i64[]", primals_411: "f32[87]", primals_412: "f32[87]", primals_413: "i64[]", primals_414: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_414, primals_99, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    
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
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 32, 112, 112]" = torch.ops.aten.clone.default(add_4)
    sigmoid: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(add_4)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, sigmoid);  add_4 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(mul_7, primals_100, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min: "f32[8, 32, 112, 112]" = torch.ops.aten.clamp_min.default(add_9, 0.0)
    clamp_max: "f32[8, 32, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6.0);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(clamp_max, primals_101, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_15: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_16: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_12: "f32[16]" = torch.ops.aten.add.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(mul_18, 0.1);  mul_18 = None
    mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_13: "f32[16]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_21: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_15, unsqueeze_9);  mul_15 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_11);  mul_21 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(add_14, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 96, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 96, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_22: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_23: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_24: "f32[96]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_17: "f32[96]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    squeeze_11: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_25: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_26: "f32[96]" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
    mul_27: "f32[96]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_18: "f32[96]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
    unsqueeze_12: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_28: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
    unsqueeze_14: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 96, 112, 112]" = torch.ops.aten.clone.default(add_19)
    sigmoid_1: "f32[8, 96, 112, 112]" = torch.ops.aten.sigmoid.default(add_19)
    mul_29: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_1);  add_19 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 96, 56, 56]" = torch.ops.aten.convolution.default(mul_29, primals_103, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 96, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 96, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_30: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_31: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_32: "f32[96]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_22: "f32[96]" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
    squeeze_14: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_33: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_34: "f32[96]" = torch.ops.aten.mul.Tensor(mul_33, 0.1);  mul_33 = None
    mul_35: "f32[96]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_23: "f32[96]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
    unsqueeze_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_36: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_30, unsqueeze_17);  mul_30 = unsqueeze_17 = None
    unsqueeze_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_36, unsqueeze_19);  mul_36 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_1: "f32[8, 96, 56, 56]" = torch.ops.aten.clamp_min.default(add_24, 0.0)
    clamp_max_1: "f32[8, 96, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6.0);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 27, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_1, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 27, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 27, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 27, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 27, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 27, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_37: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[27]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[27]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_38: "f32[27]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_39: "f32[27]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_27: "f32[27]" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
    squeeze_17: "f32[27]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_40: "f32[27]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_41: "f32[27]" = torch.ops.aten.mul.Tensor(mul_40, 0.1);  mul_40 = None
    mul_42: "f32[27]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_28: "f32[27]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    unsqueeze_20: "f32[27, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_43: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_21);  mul_37 = unsqueeze_21 = None
    unsqueeze_22: "f32[27, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 27, 56, 56]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_23);  mul_43 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 162, 56, 56]" = torch.ops.aten.convolution.default(add_29, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 162, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 162, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 162, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 162, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_44: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[162]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[162]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_45: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_46: "f32[162]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_32: "f32[162]" = torch.ops.aten.add.Tensor(mul_45, mul_46);  mul_45 = mul_46 = None
    squeeze_20: "f32[162]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_47: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_48: "f32[162]" = torch.ops.aten.mul.Tensor(mul_47, 0.1);  mul_47 = None
    mul_49: "f32[162]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_33: "f32[162]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
    unsqueeze_24: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_50: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_25);  mul_44 = unsqueeze_25 = None
    unsqueeze_26: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 162, 56, 56]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_27);  mul_50 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 162, 56, 56]" = torch.ops.aten.clone.default(add_34)
    sigmoid_2: "f32[8, 162, 56, 56]" = torch.ops.aten.sigmoid.default(add_34)
    mul_51: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(add_34, sigmoid_2);  add_34 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 162, 56, 56]" = torch.ops.aten.convolution.default(mul_51, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 162)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 162, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 162, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 162, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 162, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_52: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[162]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[162]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_53: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_54: "f32[162]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_37: "f32[162]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    squeeze_23: "f32[162]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_55: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_56: "f32[162]" = torch.ops.aten.mul.Tensor(mul_55, 0.1);  mul_55 = None
    mul_57: "f32[162]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_38: "f32[162]" = torch.ops.aten.add.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
    unsqueeze_28: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_58: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_29);  mul_52 = unsqueeze_29 = None
    unsqueeze_30: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 162, 56, 56]" = torch.ops.aten.add.Tensor(mul_58, unsqueeze_31);  mul_58 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_2: "f32[8, 162, 56, 56]" = torch.ops.aten.clamp_min.default(add_39, 0.0)
    clamp_max_2: "f32[8, 162, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6.0);  clamp_min_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 38, 56, 56]" = torch.ops.aten.convolution.default(clamp_max_2, primals_107, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 38, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 38, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 38, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 38, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 38, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_59: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[38]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[38]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_60: "f32[38]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_61: "f32[38]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_42: "f32[38]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    squeeze_26: "f32[38]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_62: "f32[38]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_63: "f32[38]" = torch.ops.aten.mul.Tensor(mul_62, 0.1);  mul_62 = None
    mul_64: "f32[38]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_43: "f32[38]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    unsqueeze_32: "f32[38, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_65: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(mul_59, unsqueeze_33);  mul_59 = unsqueeze_33 = None
    unsqueeze_34: "f32[38, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 38, 56, 56]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_35);  mul_65 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_1: "f32[8, 38, 56, 56]" = torch.ops.aten.slice.Tensor(add_44, 0, 0, 9223372036854775807);  add_44 = None
    slice_2: "f32[8, 27, 56, 56]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 27)
    add_45: "f32[8, 27, 56, 56]" = torch.ops.aten.add.Tensor(slice_2, add_29);  slice_2 = None
    slice_4: "f32[8, 11, 56, 56]" = torch.ops.aten.slice.Tensor(slice_1, 1, 27, 9223372036854775807);  slice_1 = None
    cat: "f32[8, 38, 56, 56]" = torch.ops.aten.cat.default([add_45, slice_4], 1);  add_45 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 228, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_108, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 228, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 228, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 228, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 228, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 228, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_66: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[228]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[228]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_67: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_68: "f32[228]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_48: "f32[228]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    squeeze_29: "f32[228]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_69: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_70: "f32[228]" = torch.ops.aten.mul.Tensor(mul_69, 0.1);  mul_69 = None
    mul_71: "f32[228]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_49: "f32[228]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
    unsqueeze_36: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_72: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_37);  mul_66 = unsqueeze_37 = None
    unsqueeze_38: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 228, 56, 56]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_39);  mul_72 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 228, 56, 56]" = torch.ops.aten.clone.default(add_50)
    sigmoid_3: "f32[8, 228, 56, 56]" = torch.ops.aten.sigmoid.default(add_50)
    mul_73: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(add_50, sigmoid_3);  add_50 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 228, 28, 28]" = torch.ops.aten.convolution.default(mul_73, primals_109, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 228, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 228, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 228, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 228, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 228, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_74: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[228]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[228]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_75: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_76: "f32[228]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_53: "f32[228]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    squeeze_32: "f32[228]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_77: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_78: "f32[228]" = torch.ops.aten.mul.Tensor(mul_77, 0.1);  mul_77 = None
    mul_79: "f32[228]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_54: "f32[228]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    unsqueeze_40: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_80: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(mul_74, unsqueeze_41);  mul_74 = unsqueeze_41 = None
    unsqueeze_42: "f32[228, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 228, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_43);  mul_80 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 228, 1, 1]" = torch.ops.aten.mean.dim(add_55, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_11: "f32[8, 19, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_110, primals_111, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_377, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 19, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 19, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 19, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 19, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 19, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_81: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    squeeze_33: "f32[19]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    mul_82: "f32[19]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
    mul_83: "f32[19]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_58: "f32[19]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    squeeze_35: "f32[19]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_84: "f32[19]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.1428571428571428);  squeeze_35 = None
    mul_85: "f32[19]" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
    mul_86: "f32[19]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_59: "f32[19]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    unsqueeze_44: "f32[19, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
    unsqueeze_45: "f32[19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_87: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_45);  mul_81 = unsqueeze_45 = None
    unsqueeze_46: "f32[19, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
    unsqueeze_47: "f32[19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 19, 1, 1]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_47);  mul_87 = unsqueeze_47 = None
    relu: "f32[8, 19, 1, 1]" = torch.ops.aten.relu.default(add_60);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_12: "f32[8, 228, 1, 1]" = torch.ops.aten.convolution.default(relu, primals_114, primals_115, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[8, 228, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_88: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_3: "f32[8, 228, 28, 28]" = torch.ops.aten.clamp_min.default(mul_88, 0.0);  mul_88 = None
    clamp_max_3: "f32[8, 228, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6.0);  clamp_min_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 50, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_3, primals_116, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 50, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 50, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_62: "f32[1, 50, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 50, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[8, 50, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_25)
    mul_89: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[50]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[50]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_90: "f32[50]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_91: "f32[50]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_63: "f32[50]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    squeeze_38: "f32[50]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_92: "f32[50]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_93: "f32[50]" = torch.ops.aten.mul.Tensor(mul_92, 0.1);  mul_92 = None
    mul_94: "f32[50]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_64: "f32[50]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    unsqueeze_48: "f32[50, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_49: "f32[50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_95: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_49);  mul_89 = unsqueeze_49 = None
    unsqueeze_50: "f32[50, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_51: "f32[50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_65: "f32[8, 50, 28, 28]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_51);  mul_95 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 300, 28, 28]" = torch.ops.aten.convolution.default(add_65, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_66: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 300, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 300, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_67: "f32[1, 300, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 300, 1, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_13: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_27)
    mul_96: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[300]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[300]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_97: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_98: "f32[300]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_68: "f32[300]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    squeeze_41: "f32[300]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_99: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_100: "f32[300]" = torch.ops.aten.mul.Tensor(mul_99, 0.1);  mul_99 = None
    mul_101: "f32[300]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_69: "f32[300]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
    unsqueeze_52: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_53: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_102: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_53);  mul_96 = unsqueeze_53 = None
    unsqueeze_54: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_55: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_70: "f32[8, 300, 28, 28]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_55);  mul_102 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 300, 28, 28]" = torch.ops.aten.clone.default(add_70)
    sigmoid_5: "f32[8, 300, 28, 28]" = torch.ops.aten.sigmoid.default(add_70)
    mul_103: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(add_70, sigmoid_5);  add_70 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 300, 28, 28]" = torch.ops.aten.convolution.default(mul_103, primals_118, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 300)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_71: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 300, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 300, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_72: "f32[1, 300, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 300, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_14: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_29)
    mul_104: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[300]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[300]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_105: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_106: "f32[300]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_73: "f32[300]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
    squeeze_44: "f32[300]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_107: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_108: "f32[300]" = torch.ops.aten.mul.Tensor(mul_107, 0.1);  mul_107 = None
    mul_109: "f32[300]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_74: "f32[300]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
    unsqueeze_56: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_57: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_110: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(mul_104, unsqueeze_57);  mul_104 = unsqueeze_57 = None
    unsqueeze_58: "f32[300, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_59: "f32[300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_75: "f32[8, 300, 28, 28]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_59);  mul_110 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 300, 1, 1]" = torch.ops.aten.mean.dim(add_75, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_16: "f32[8, 25, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_119, primals_120, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_76: "i64[]" = torch.ops.aten.add.Tensor(primals_380, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 25, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 25, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_77: "f32[1, 25, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 25, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_15: "f32[8, 25, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_31)
    mul_111: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    squeeze_45: "f32[25]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    mul_112: "f32[25]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1);  squeeze_45 = None
    mul_113: "f32[25]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_78: "f32[25]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    squeeze_47: "f32[25]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_114: "f32[25]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.1428571428571428);  squeeze_47 = None
    mul_115: "f32[25]" = torch.ops.aten.mul.Tensor(mul_114, 0.1);  mul_114 = None
    mul_116: "f32[25]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_79: "f32[25]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    unsqueeze_60: "f32[25, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_61: "f32[25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_117: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_61);  mul_111 = unsqueeze_61 = None
    unsqueeze_62: "f32[25, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_63: "f32[25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_80: "f32[8, 25, 1, 1]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_63);  mul_117 = unsqueeze_63 = None
    relu_1: "f32[8, 25, 1, 1]" = torch.ops.aten.relu.default(add_80);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_17: "f32[8, 300, 1, 1]" = torch.ops.aten.convolution.default(relu_1, primals_123, primals_124, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[8, 300, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_118: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_4: "f32[8, 300, 28, 28]" = torch.ops.aten.clamp_min.default(mul_118, 0.0);  mul_118 = None
    clamp_max_4: "f32[8, 300, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6.0);  clamp_min_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 61, 28, 28]" = torch.ops.aten.convolution.default(clamp_max_4, primals_125, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 61, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 61, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_82: "f32[1, 61, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 61, 1, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_16: "f32[8, 61, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_33)
    mul_119: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[61]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[61]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_120: "f32[61]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_121: "f32[61]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_83: "f32[61]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_50: "f32[61]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_122: "f32[61]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_123: "f32[61]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[61]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_84: "f32[61]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_64: "f32[61, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_65: "f32[61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_125: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_65);  mul_119 = unsqueeze_65 = None
    unsqueeze_66: "f32[61, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_67: "f32[61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_85: "f32[8, 61, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_67);  mul_125 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_5: "f32[8, 61, 28, 28]" = torch.ops.aten.slice.Tensor(add_85, 0, 0, 9223372036854775807);  add_85 = None
    slice_6: "f32[8, 50, 28, 28]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 50)
    add_86: "f32[8, 50, 28, 28]" = torch.ops.aten.add.Tensor(slice_6, add_65);  slice_6 = None
    slice_8: "f32[8, 11, 28, 28]" = torch.ops.aten.slice.Tensor(slice_5, 1, 50, 9223372036854775807);  slice_5 = None
    cat_1: "f32[8, 61, 28, 28]" = torch.ops.aten.cat.default([add_86, slice_8], 1);  add_86 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 366, 28, 28]" = torch.ops.aten.convolution.default(cat_1, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_87: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 366, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 366, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_88: "f32[1, 366, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 366, 1, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_17: "f32[8, 366, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_35)
    mul_126: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[366]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[366]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_127: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_128: "f32[366]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_89: "f32[366]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_53: "f32[366]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_129: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_130: "f32[366]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[366]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_90: "f32[366]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_68: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_69: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_132: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_69);  mul_126 = unsqueeze_69 = None
    unsqueeze_70: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_71: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_91: "f32[8, 366, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_71);  mul_132 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_5: "f32[8, 366, 28, 28]" = torch.ops.aten.clone.default(add_91)
    sigmoid_7: "f32[8, 366, 28, 28]" = torch.ops.aten.sigmoid.default(add_91)
    mul_133: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(add_91, sigmoid_7);  add_91 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 366, 14, 14]" = torch.ops.aten.convolution.default(mul_133, primals_127, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 366)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 366, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 366, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_93: "f32[1, 366, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 366, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_18: "f32[8, 366, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_37)
    mul_134: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[366]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[366]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_135: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_136: "f32[366]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_94: "f32[366]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    squeeze_56: "f32[366]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_137: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_138: "f32[366]" = torch.ops.aten.mul.Tensor(mul_137, 0.1);  mul_137 = None
    mul_139: "f32[366]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_95: "f32[366]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
    unsqueeze_72: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_73: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_140: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_73);  mul_134 = unsqueeze_73 = None
    unsqueeze_74: "f32[366, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_75: "f32[366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_96: "f32[8, 366, 14, 14]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_75);  mul_140 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 366, 1, 1]" = torch.ops.aten.mean.dim(add_96, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_21: "f32[8, 30, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_128, primals_129, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_97: "i64[]" = torch.ops.aten.add.Tensor(primals_383, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 30, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 30, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_98: "f32[1, 30, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 30, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_19: "f32[8, 30, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_39)
    mul_141: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    squeeze_57: "f32[30]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    mul_142: "f32[30]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
    mul_143: "f32[30]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_99: "f32[30]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    squeeze_59: "f32[30]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_144: "f32[30]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.1428571428571428);  squeeze_59 = None
    mul_145: "f32[30]" = torch.ops.aten.mul.Tensor(mul_144, 0.1);  mul_144 = None
    mul_146: "f32[30]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_100: "f32[30]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    unsqueeze_76: "f32[30, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
    unsqueeze_77: "f32[30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_147: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_77);  mul_141 = unsqueeze_77 = None
    unsqueeze_78: "f32[30, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1);  primals_131 = None
    unsqueeze_79: "f32[30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_101: "f32[8, 30, 1, 1]" = torch.ops.aten.add.Tensor(mul_147, unsqueeze_79);  mul_147 = unsqueeze_79 = None
    relu_2: "f32[8, 30, 1, 1]" = torch.ops.aten.relu.default(add_101);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_22: "f32[8, 366, 1, 1]" = torch.ops.aten.convolution.default(relu_2, primals_132, primals_133, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[8, 366, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_148: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(add_96, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_5: "f32[8, 366, 14, 14]" = torch.ops.aten.clamp_min.default(mul_148, 0.0);  mul_148 = None
    clamp_max_5: "f32[8, 366, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6.0);  clamp_min_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_5, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_102: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 72, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 72, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_103: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_20: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_41)
    mul_149: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_150: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_151: "f32[72]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_104: "f32[72]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
    squeeze_62: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_152: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_153: "f32[72]" = torch.ops.aten.mul.Tensor(mul_152, 0.1);  mul_152 = None
    mul_154: "f32[72]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_105: "f32[72]" = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
    unsqueeze_80: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_81: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_155: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_149, unsqueeze_81);  mul_149 = unsqueeze_81 = None
    unsqueeze_82: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_83: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_106: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_83);  mul_155 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 432, 14, 14]" = torch.ops.aten.convolution.default(add_106, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 432, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 432, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_108: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_21: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_43)
    mul_156: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_157: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_158: "f32[432]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_109: "f32[432]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    squeeze_65: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_159: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0006381620931717);  squeeze_65 = None
    mul_160: "f32[432]" = torch.ops.aten.mul.Tensor(mul_159, 0.1);  mul_159 = None
    mul_161: "f32[432]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_110: "f32[432]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
    unsqueeze_84: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_85: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_162: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_85);  mul_156 = unsqueeze_85 = None
    unsqueeze_86: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_87: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_111: "f32[8, 432, 14, 14]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_87);  mul_162 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 432, 14, 14]" = torch.ops.aten.clone.default(add_111)
    sigmoid_9: "f32[8, 432, 14, 14]" = torch.ops.aten.sigmoid.default(add_111)
    mul_163: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(add_111, sigmoid_9);  add_111 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 432, 14, 14]" = torch.ops.aten.convolution.default(mul_163, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_112: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 432, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 432, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_113: "f32[1, 432, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 432, 1, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_22: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_45)
    mul_164: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[432]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_165: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_166: "f32[432]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_114: "f32[432]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    squeeze_68: "f32[432]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_167: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_168: "f32[432]" = torch.ops.aten.mul.Tensor(mul_167, 0.1);  mul_167 = None
    mul_169: "f32[432]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_115: "f32[432]" = torch.ops.aten.add.Tensor(mul_168, mul_169);  mul_168 = mul_169 = None
    unsqueeze_88: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_89: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_170: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_89);  mul_164 = unsqueeze_89 = None
    unsqueeze_90: "f32[432, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_91: "f32[432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_116: "f32[8, 432, 14, 14]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_91);  mul_170 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 432, 1, 1]" = torch.ops.aten.mean.dim(add_116, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_26: "f32[8, 36, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_386, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 36, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 36, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_118: "f32[1, 36, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 36, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[8, 36, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_47)
    mul_171: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    squeeze_69: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    mul_172: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1);  squeeze_69 = None
    mul_173: "f32[36]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_119: "f32[36]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_71: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_174: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.1428571428571428);  squeeze_71 = None
    mul_175: "f32[36]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[36]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_120: "f32[36]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    unsqueeze_92: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1)
    unsqueeze_93: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_177: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_93);  mul_171 = unsqueeze_93 = None
    unsqueeze_94: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1);  primals_140 = None
    unsqueeze_95: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_121: "f32[8, 36, 1, 1]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_95);  mul_177 = unsqueeze_95 = None
    relu_3: "f32[8, 36, 1, 1]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_27: "f32[8, 432, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_141, primals_142, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[8, 432, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_178: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_6: "f32[8, 432, 14, 14]" = torch.ops.aten.clamp_min.default(mul_178, 0.0);  mul_178 = None
    clamp_max_6: "f32[8, 432, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6.0);  clamp_min_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 84, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_6, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 84, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 84, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_123: "f32[1, 84, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 84, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_24: "f32[8, 84, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_49)
    mul_179: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[84]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[84]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_180: "f32[84]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_181: "f32[84]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_124: "f32[84]" = torch.ops.aten.add.Tensor(mul_180, mul_181);  mul_180 = mul_181 = None
    squeeze_74: "f32[84]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_182: "f32[84]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0006381620931717);  squeeze_74 = None
    mul_183: "f32[84]" = torch.ops.aten.mul.Tensor(mul_182, 0.1);  mul_182 = None
    mul_184: "f32[84]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_125: "f32[84]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    unsqueeze_96: "f32[84, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_97: "f32[84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_185: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_97);  mul_179 = unsqueeze_97 = None
    unsqueeze_98: "f32[84, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_99: "f32[84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_126: "f32[8, 84, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_99);  mul_185 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_9: "f32[8, 84, 14, 14]" = torch.ops.aten.slice.Tensor(add_126, 0, 0, 9223372036854775807);  add_126 = None
    slice_10: "f32[8, 72, 14, 14]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 72)
    add_127: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(slice_10, add_106);  slice_10 = None
    slice_12: "f32[8, 12, 14, 14]" = torch.ops.aten.slice.Tensor(slice_9, 1, 72, 9223372036854775807);  slice_9 = None
    cat_2: "f32[8, 84, 14, 14]" = torch.ops.aten.cat.default([add_127, slice_12], 1);  add_127 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 504, 14, 14]" = torch.ops.aten.convolution.default(cat_2, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 504, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 504, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_129: "f32[1, 504, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 504, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_51)
    mul_186: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[504]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[504]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_187: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_188: "f32[504]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_130: "f32[504]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
    squeeze_77: "f32[504]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_189: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_190: "f32[504]" = torch.ops.aten.mul.Tensor(mul_189, 0.1);  mul_189 = None
    mul_191: "f32[504]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_131: "f32[504]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    unsqueeze_100: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_101: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_192: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_101);  mul_186 = unsqueeze_101 = None
    unsqueeze_102: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_103: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_132: "f32[8, 504, 14, 14]" = torch.ops.aten.add.Tensor(mul_192, unsqueeze_103);  mul_192 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 504, 14, 14]" = torch.ops.aten.clone.default(add_132)
    sigmoid_11: "f32[8, 504, 14, 14]" = torch.ops.aten.sigmoid.default(add_132)
    mul_193: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(add_132, sigmoid_11);  add_132 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 504, 14, 14]" = torch.ops.aten.convolution.default(mul_193, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 504)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 504, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 504, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_134: "f32[1, 504, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 504, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_26: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_53)
    mul_194: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[504]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[504]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_195: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_196: "f32[504]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_135: "f32[504]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    squeeze_80: "f32[504]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_197: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_198: "f32[504]" = torch.ops.aten.mul.Tensor(mul_197, 0.1);  mul_197 = None
    mul_199: "f32[504]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_136: "f32[504]" = torch.ops.aten.add.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
    unsqueeze_104: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_105: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_200: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(mul_194, unsqueeze_105);  mul_194 = unsqueeze_105 = None
    unsqueeze_106: "f32[504, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_107: "f32[504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_137: "f32[8, 504, 14, 14]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_107);  mul_200 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 504, 1, 1]" = torch.ops.aten.mean.dim(add_137, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_31: "f32[8, 42, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_138: "i64[]" = torch.ops.aten.add.Tensor(primals_389, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 42, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 42, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_139: "f32[1, 42, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 42, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_27: "f32[8, 42, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_55)
    mul_201: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
    squeeze_81: "f32[42]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    mul_202: "f32[42]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1);  squeeze_81 = None
    mul_203: "f32[42]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_140: "f32[42]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    squeeze_83: "f32[42]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_204: "f32[42]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.1428571428571428);  squeeze_83 = None
    mul_205: "f32[42]" = torch.ops.aten.mul.Tensor(mul_204, 0.1);  mul_204 = None
    mul_206: "f32[42]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_141: "f32[42]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    unsqueeze_108: "f32[42, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1)
    unsqueeze_109: "f32[42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_207: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_109);  mul_201 = unsqueeze_109 = None
    unsqueeze_110: "f32[42, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1);  primals_149 = None
    unsqueeze_111: "f32[42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_142: "f32[8, 42, 1, 1]" = torch.ops.aten.add.Tensor(mul_207, unsqueeze_111);  mul_207 = unsqueeze_111 = None
    relu_4: "f32[8, 42, 1, 1]" = torch.ops.aten.relu.default(add_142);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_32: "f32[8, 504, 1, 1]" = torch.ops.aten.convolution.default(relu_4, primals_150, primals_151, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 504, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_208: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_7: "f32[8, 504, 14, 14]" = torch.ops.aten.clamp_min.default(mul_208, 0.0);  mul_208 = None
    clamp_max_7: "f32[8, 504, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6.0);  clamp_min_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 95, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_7, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 95, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 95, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_144: "f32[1, 95, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 95, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_28: "f32[8, 95, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_57)
    mul_209: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[95]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[95]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_210: "f32[95]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_211: "f32[95]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_145: "f32[95]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    squeeze_86: "f32[95]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_212: "f32[95]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_213: "f32[95]" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
    mul_214: "f32[95]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_146: "f32[95]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    unsqueeze_112: "f32[95, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_113: "f32[95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_215: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_113);  mul_209 = unsqueeze_113 = None
    unsqueeze_114: "f32[95, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_115: "f32[95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_147: "f32[8, 95, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_115);  mul_215 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_13: "f32[8, 95, 14, 14]" = torch.ops.aten.slice.Tensor(add_147, 0, 0, 9223372036854775807);  add_147 = None
    slice_14: "f32[8, 84, 14, 14]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 84)
    add_148: "f32[8, 84, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, cat_2);  slice_14 = None
    slice_16: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(slice_13, 1, 84, 9223372036854775807);  slice_13 = None
    cat_3: "f32[8, 95, 14, 14]" = torch.ops.aten.cat.default([add_148, slice_16], 1);  add_148 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 570, 14, 14]" = torch.ops.aten.convolution.default(cat_3, primals_153, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_149: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 570, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 570, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_150: "f32[1, 570, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 570, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_29: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_59)
    mul_216: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[570]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[570]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_217: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_218: "f32[570]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_151: "f32[570]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    squeeze_89: "f32[570]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_219: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_220: "f32[570]" = torch.ops.aten.mul.Tensor(mul_219, 0.1);  mul_219 = None
    mul_221: "f32[570]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_152: "f32[570]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    unsqueeze_116: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_117: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_222: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_117);  mul_216 = unsqueeze_117 = None
    unsqueeze_118: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_119: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_153: "f32[8, 570, 14, 14]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_119);  mul_222 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_8: "f32[8, 570, 14, 14]" = torch.ops.aten.clone.default(add_153)
    sigmoid_13: "f32[8, 570, 14, 14]" = torch.ops.aten.sigmoid.default(add_153)
    mul_223: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(add_153, sigmoid_13);  add_153 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 570, 14, 14]" = torch.ops.aten.convolution.default(mul_223, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 570)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 570, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 570, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_155: "f32[1, 570, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 570, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_30: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_61)
    mul_224: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[570]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[570]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_225: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_226: "f32[570]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_156: "f32[570]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_92: "f32[570]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_227: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_228: "f32[570]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[570]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_157: "f32[570]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_120: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_121: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_230: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_121);  mul_224 = unsqueeze_121 = None
    unsqueeze_122: "f32[570, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_123: "f32[570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_158: "f32[8, 570, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_123);  mul_230 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 570, 1, 1]" = torch.ops.aten.mean.dim(add_158, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_36: "f32[8, 47, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_155, primals_156, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_392, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 47, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 47, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_160: "f32[1, 47, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 47, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_31: "f32[8, 47, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_63)
    mul_231: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
    squeeze_93: "f32[47]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    mul_232: "f32[47]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
    mul_233: "f32[47]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_161: "f32[47]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_95: "f32[47]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_234: "f32[47]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.1428571428571428);  squeeze_95 = None
    mul_235: "f32[47]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[47]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_162: "f32[47]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_124: "f32[47, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1)
    unsqueeze_125: "f32[47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_237: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_125);  mul_231 = unsqueeze_125 = None
    unsqueeze_126: "f32[47, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1);  primals_158 = None
    unsqueeze_127: "f32[47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_163: "f32[8, 47, 1, 1]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_127);  mul_237 = unsqueeze_127 = None
    relu_5: "f32[8, 47, 1, 1]" = torch.ops.aten.relu.default(add_163);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_37: "f32[8, 570, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_159, primals_160, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_14: "f32[8, 570, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_238: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(add_158, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_8: "f32[8, 570, 14, 14]" = torch.ops.aten.clamp_min.default(mul_238, 0.0);  mul_238 = None
    clamp_max_8: "f32[8, 570, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6.0);  clamp_min_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 106, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_8, primals_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_164: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 106, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 106, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_165: "f32[1, 106, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 106, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_32: "f32[8, 106, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_65)
    mul_239: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[106]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[106]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_240: "f32[106]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_241: "f32[106]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_166: "f32[106]" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
    squeeze_98: "f32[106]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_242: "f32[106]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_243: "f32[106]" = torch.ops.aten.mul.Tensor(mul_242, 0.1);  mul_242 = None
    mul_244: "f32[106]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_167: "f32[106]" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    unsqueeze_128: "f32[106, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_129: "f32[106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_245: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_129);  mul_239 = unsqueeze_129 = None
    unsqueeze_130: "f32[106, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_131: "f32[106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_168: "f32[8, 106, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_131);  mul_245 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_17: "f32[8, 106, 14, 14]" = torch.ops.aten.slice.Tensor(add_168, 0, 0, 9223372036854775807);  add_168 = None
    slice_18: "f32[8, 95, 14, 14]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 95)
    add_169: "f32[8, 95, 14, 14]" = torch.ops.aten.add.Tensor(slice_18, cat_3);  slice_18 = None
    slice_20: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(slice_17, 1, 95, 9223372036854775807);  slice_17 = None
    cat_4: "f32[8, 106, 14, 14]" = torch.ops.aten.cat.default([add_169, slice_20], 1);  add_169 = slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 636, 14, 14]" = torch.ops.aten.convolution.default(cat_4, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 636, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 636, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_171: "f32[1, 636, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 636, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_33: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_67)
    mul_246: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[636]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[636]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_247: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_248: "f32[636]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_172: "f32[636]" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
    squeeze_101: "f32[636]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_249: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_250: "f32[636]" = torch.ops.aten.mul.Tensor(mul_249, 0.1);  mul_249 = None
    mul_251: "f32[636]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_173: "f32[636]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    unsqueeze_132: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_133: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_252: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_133);  mul_246 = unsqueeze_133 = None
    unsqueeze_134: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_135: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_174: "f32[8, 636, 14, 14]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_135);  mul_252 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 636, 14, 14]" = torch.ops.aten.clone.default(add_174)
    sigmoid_15: "f32[8, 636, 14, 14]" = torch.ops.aten.sigmoid.default(add_174)
    mul_253: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(add_174, sigmoid_15);  add_174 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 636, 14, 14]" = torch.ops.aten.convolution.default(mul_253, primals_163, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 636)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 636, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 636, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_176: "f32[1, 636, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 636, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_34: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_69)
    mul_254: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[636]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[636]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_255: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_256: "f32[636]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_177: "f32[636]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    squeeze_104: "f32[636]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_257: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_258: "f32[636]" = torch.ops.aten.mul.Tensor(mul_257, 0.1);  mul_257 = None
    mul_259: "f32[636]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_178: "f32[636]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    unsqueeze_136: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_137: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_260: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_137);  mul_254 = unsqueeze_137 = None
    unsqueeze_138: "f32[636, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_139: "f32[636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_179: "f32[8, 636, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_139);  mul_260 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 636, 1, 1]" = torch.ops.aten.mean.dim(add_179, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_41: "f32[8, 53, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_164, primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 53, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 53, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_181: "f32[1, 53, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 53, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_35: "f32[8, 53, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_71)
    mul_261: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
    squeeze_105: "f32[53]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    mul_262: "f32[53]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1);  squeeze_105 = None
    mul_263: "f32[53]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_182: "f32[53]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    squeeze_107: "f32[53]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_264: "f32[53]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.1428571428571428);  squeeze_107 = None
    mul_265: "f32[53]" = torch.ops.aten.mul.Tensor(mul_264, 0.1);  mul_264 = None
    mul_266: "f32[53]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_183: "f32[53]" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    unsqueeze_140: "f32[53, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1)
    unsqueeze_141: "f32[53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_267: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_141);  mul_261 = unsqueeze_141 = None
    unsqueeze_142: "f32[53, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1);  primals_167 = None
    unsqueeze_143: "f32[53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_184: "f32[8, 53, 1, 1]" = torch.ops.aten.add.Tensor(mul_267, unsqueeze_143);  mul_267 = unsqueeze_143 = None
    relu_6: "f32[8, 53, 1, 1]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_42: "f32[8, 636, 1, 1]" = torch.ops.aten.convolution.default(relu_6, primals_168, primals_169, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_16: "f32[8, 636, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_268: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(add_179, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_9: "f32[8, 636, 14, 14]" = torch.ops.aten.clamp_min.default(mul_268, 0.0);  mul_268 = None
    clamp_max_9: "f32[8, 636, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6.0);  clamp_min_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[8, 117, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_9, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 117, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 117, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_186: "f32[1, 117, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 117, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_36: "f32[8, 117, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_73)
    mul_269: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[117]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[117]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_270: "f32[117]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_271: "f32[117]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_187: "f32[117]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    squeeze_110: "f32[117]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_272: "f32[117]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_273: "f32[117]" = torch.ops.aten.mul.Tensor(mul_272, 0.1);  mul_272 = None
    mul_274: "f32[117]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_188: "f32[117]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
    unsqueeze_144: "f32[117, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_145: "f32[117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_275: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(mul_269, unsqueeze_145);  mul_269 = unsqueeze_145 = None
    unsqueeze_146: "f32[117, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_147: "f32[117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_189: "f32[8, 117, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_147);  mul_275 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_21: "f32[8, 117, 14, 14]" = torch.ops.aten.slice.Tensor(add_189, 0, 0, 9223372036854775807);  add_189 = None
    slice_22: "f32[8, 106, 14, 14]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 106)
    add_190: "f32[8, 106, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, cat_4);  slice_22 = None
    slice_24: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(slice_21, 1, 106, 9223372036854775807);  slice_21 = None
    cat_5: "f32[8, 117, 14, 14]" = torch.ops.aten.cat.default([add_190, slice_24], 1);  add_190 = slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 702, 14, 14]" = torch.ops.aten.convolution.default(cat_5, primals_171, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 702, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 702, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_192: "f32[1, 702, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 702, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_37: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_75)
    mul_276: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[702]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[702]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_277: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_278: "f32[702]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_193: "f32[702]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    squeeze_113: "f32[702]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_279: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_280: "f32[702]" = torch.ops.aten.mul.Tensor(mul_279, 0.1);  mul_279 = None
    mul_281: "f32[702]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_194: "f32[702]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
    unsqueeze_148: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_149: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_282: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_149);  mul_276 = unsqueeze_149 = None
    unsqueeze_150: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_151: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_195: "f32[8, 702, 14, 14]" = torch.ops.aten.add.Tensor(mul_282, unsqueeze_151);  mul_282 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 702, 14, 14]" = torch.ops.aten.clone.default(add_195)
    sigmoid_17: "f32[8, 702, 14, 14]" = torch.ops.aten.sigmoid.default(add_195)
    mul_283: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(add_195, sigmoid_17);  add_195 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 702, 14, 14]" = torch.ops.aten.convolution.default(mul_283, primals_172, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 702)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 702, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 702, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_197: "f32[1, 702, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 702, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_38: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_77)
    mul_284: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[702]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[702]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_285: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_286: "f32[702]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_198: "f32[702]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    squeeze_116: "f32[702]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_287: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_288: "f32[702]" = torch.ops.aten.mul.Tensor(mul_287, 0.1);  mul_287 = None
    mul_289: "f32[702]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_199: "f32[702]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_152: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_153: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_290: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_153);  mul_284 = unsqueeze_153 = None
    unsqueeze_154: "f32[702, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_155: "f32[702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_200: "f32[8, 702, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_155);  mul_290 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 702, 1, 1]" = torch.ops.aten.mean.dim(add_200, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_46: "f32[8, 58, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_398, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 58, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 58, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_202: "f32[1, 58, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 58, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_39: "f32[8, 58, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_79)
    mul_291: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
    squeeze_117: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    mul_292: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1);  squeeze_117 = None
    mul_293: "f32[58]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_203: "f32[58]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    squeeze_119: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_294: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.1428571428571428);  squeeze_119 = None
    mul_295: "f32[58]" = torch.ops.aten.mul.Tensor(mul_294, 0.1);  mul_294 = None
    mul_296: "f32[58]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_204: "f32[58]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    unsqueeze_156: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1)
    unsqueeze_157: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_297: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(mul_291, unsqueeze_157);  mul_291 = unsqueeze_157 = None
    unsqueeze_158: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1);  primals_176 = None
    unsqueeze_159: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_205: "f32[8, 58, 1, 1]" = torch.ops.aten.add.Tensor(mul_297, unsqueeze_159);  mul_297 = unsqueeze_159 = None
    relu_7: "f32[8, 58, 1, 1]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_47: "f32[8, 702, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_177, primals_178, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_18: "f32[8, 702, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_298: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(add_200, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_10: "f32[8, 702, 14, 14]" = torch.ops.aten.clamp_min.default(mul_298, 0.0);  mul_298 = None
    clamp_max_10: "f32[8, 702, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6.0);  clamp_min_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(clamp_max_10, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_207: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_40: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_81)
    mul_299: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_300: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_301: "f32[128]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_208: "f32[128]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    squeeze_122: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_302: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_303: "f32[128]" = torch.ops.aten.mul.Tensor(mul_302, 0.1);  mul_302 = None
    mul_304: "f32[128]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_209: "f32[128]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    unsqueeze_160: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_161: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_305: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_161);  mul_299 = unsqueeze_161 = None
    unsqueeze_162: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_163: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_210: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_163);  mul_305 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_25: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(add_210, 0, 0, 9223372036854775807);  add_210 = None
    slice_26: "f32[8, 117, 14, 14]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 117)
    add_211: "f32[8, 117, 14, 14]" = torch.ops.aten.add.Tensor(slice_26, cat_5);  slice_26 = None
    slice_28: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(slice_25, 1, 117, 9223372036854775807);  slice_25 = None
    cat_6: "f32[8, 128, 14, 14]" = torch.ops.aten.cat.default([add_211, slice_28], 1);  add_211 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(cat_6, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 768, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 768, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_213: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_41: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_83)
    mul_306: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_307: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_308: "f32[768]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_214: "f32[768]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    squeeze_125: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_309: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_310: "f32[768]" = torch.ops.aten.mul.Tensor(mul_309, 0.1);  mul_309 = None
    mul_311: "f32[768]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_215: "f32[768]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    unsqueeze_164: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_165: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_312: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_165);  mul_306 = unsqueeze_165 = None
    unsqueeze_166: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_167: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_216: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_312, unsqueeze_167);  mul_312 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_11: "f32[8, 768, 14, 14]" = torch.ops.aten.clone.default(add_216)
    sigmoid_19: "f32[8, 768, 14, 14]" = torch.ops.aten.sigmoid.default(add_216)
    mul_313: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, sigmoid_19);  add_216 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 768, 7, 7]" = torch.ops.aten.convolution.default(mul_313, primals_181, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 768)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 768, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 768, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_218: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_42: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_85)
    mul_314: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_315: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_316: "f32[768]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_219: "f32[768]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
    squeeze_128: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_317: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0025575447570332);  squeeze_128 = None
    mul_318: "f32[768]" = torch.ops.aten.mul.Tensor(mul_317, 0.1);  mul_317 = None
    mul_319: "f32[768]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_220: "f32[768]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    unsqueeze_168: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_169: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_320: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_169);  mul_314 = unsqueeze_169 = None
    unsqueeze_170: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_171: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_221: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_171);  mul_320 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_221, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_51: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 64, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 64, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_223: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_43: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_87)
    mul_321: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
    squeeze_129: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    mul_322: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1);  squeeze_129 = None
    mul_323: "f32[64]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_224: "f32[64]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    squeeze_131: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_324: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.1428571428571428);  squeeze_131 = None
    mul_325: "f32[64]" = torch.ops.aten.mul.Tensor(mul_324, 0.1);  mul_324 = None
    mul_326: "f32[64]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_225: "f32[64]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    unsqueeze_172: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
    unsqueeze_173: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_327: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_173);  mul_321 = unsqueeze_173 = None
    unsqueeze_174: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1);  primals_185 = None
    unsqueeze_175: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_226: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_175);  mul_327 = unsqueeze_175 = None
    relu_8: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_52: "f32[8, 768, 1, 1]" = torch.ops.aten.convolution.default(relu_8, primals_186, primals_187, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_20: "f32[8, 768, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_328: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_221, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_11: "f32[8, 768, 7, 7]" = torch.ops.aten.clamp_min.default(mul_328, 0.0);  mul_328 = None
    clamp_max_11: "f32[8, 768, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6.0);  clamp_min_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[8, 140, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_11, primals_188, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 140, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 140, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_228: "f32[1, 140, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 140, 1, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_44: "f32[8, 140, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_89)
    mul_329: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[140]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[140]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_330: "f32[140]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_331: "f32[140]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_229: "f32[140]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_134: "f32[140]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_332: "f32[140]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0025575447570332);  squeeze_134 = None
    mul_333: "f32[140]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[140]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_230: "f32[140]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_176: "f32[140, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_177: "f32[140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_335: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_177);  mul_329 = unsqueeze_177 = None
    unsqueeze_178: "f32[140, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_179: "f32[140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_231: "f32[8, 140, 7, 7]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_179);  mul_335 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 840, 7, 7]" = torch.ops.aten.convolution.default(add_231, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_232: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 840, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 840, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_233: "f32[1, 840, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 840, 1, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
    sub_45: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_91)
    mul_336: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[840]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[840]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_337: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_338: "f32[840]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_234: "f32[840]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_137: "f32[840]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_339: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0025575447570332);  squeeze_137 = None
    mul_340: "f32[840]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[840]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_235: "f32[840]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_180: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_181: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_342: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_181);  mul_336 = unsqueeze_181 = None
    unsqueeze_182: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_183: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_236: "f32[8, 840, 7, 7]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_183);  mul_342 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 840, 7, 7]" = torch.ops.aten.clone.default(add_236)
    sigmoid_21: "f32[8, 840, 7, 7]" = torch.ops.aten.sigmoid.default(add_236)
    mul_343: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(add_236, sigmoid_21);  add_236 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[8, 840, 7, 7]" = torch.ops.aten.convolution.default(mul_343, primals_190, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 840)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_237: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 840, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 840, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_238: "f32[1, 840, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 840, 1, 1]" = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
    sub_46: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_93)
    mul_344: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[840]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[840]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_345: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_346: "f32[840]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_239: "f32[840]" = torch.ops.aten.add.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    squeeze_140: "f32[840]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_347: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0025575447570332);  squeeze_140 = None
    mul_348: "f32[840]" = torch.ops.aten.mul.Tensor(mul_347, 0.1);  mul_347 = None
    mul_349: "f32[840]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_240: "f32[840]" = torch.ops.aten.add.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
    unsqueeze_184: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_185: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_350: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(mul_344, unsqueeze_185);  mul_344 = unsqueeze_185 = None
    unsqueeze_186: "f32[840, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_187: "f32[840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_241: "f32[8, 840, 7, 7]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_187);  mul_350 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 840, 1, 1]" = torch.ops.aten.mean.dim(add_241, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_56: "f32[8, 70, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_191, primals_192, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_242: "i64[]" = torch.ops.aten.add.Tensor(primals_404, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 70, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 70, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_243: "f32[1, 70, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 70, 1, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
    sub_47: "f32[8, 70, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_95)
    mul_351: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
    squeeze_141: "f32[70]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    mul_352: "f32[70]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1);  squeeze_141 = None
    mul_353: "f32[70]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_244: "f32[70]" = torch.ops.aten.add.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
    squeeze_143: "f32[70]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_354: "f32[70]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.1428571428571428);  squeeze_143 = None
    mul_355: "f32[70]" = torch.ops.aten.mul.Tensor(mul_354, 0.1);  mul_354 = None
    mul_356: "f32[70]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_245: "f32[70]" = torch.ops.aten.add.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
    unsqueeze_188: "f32[70, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1)
    unsqueeze_189: "f32[70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_357: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(mul_351, unsqueeze_189);  mul_351 = unsqueeze_189 = None
    unsqueeze_190: "f32[70, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1);  primals_194 = None
    unsqueeze_191: "f32[70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_246: "f32[8, 70, 1, 1]" = torch.ops.aten.add.Tensor(mul_357, unsqueeze_191);  mul_357 = unsqueeze_191 = None
    relu_9: "f32[8, 70, 1, 1]" = torch.ops.aten.relu.default(add_246);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_57: "f32[8, 840, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_195, primals_196, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_22: "f32[8, 840, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_358: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(add_241, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_12: "f32[8, 840, 7, 7]" = torch.ops.aten.clamp_min.default(mul_358, 0.0);  mul_358 = None
    clamp_max_12: "f32[8, 840, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6.0);  clamp_min_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[8, 151, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_12, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_247: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 151, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 151, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_248: "f32[1, 151, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 151, 1, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_48: "f32[8, 151, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_97)
    mul_359: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[151]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[151]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_360: "f32[151]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_361: "f32[151]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_249: "f32[151]" = torch.ops.aten.add.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
    squeeze_146: "f32[151]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_362: "f32[151]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0025575447570332);  squeeze_146 = None
    mul_363: "f32[151]" = torch.ops.aten.mul.Tensor(mul_362, 0.1);  mul_362 = None
    mul_364: "f32[151]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_250: "f32[151]" = torch.ops.aten.add.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
    unsqueeze_192: "f32[151, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_193: "f32[151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_365: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(mul_359, unsqueeze_193);  mul_359 = unsqueeze_193 = None
    unsqueeze_194: "f32[151, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_195: "f32[151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_251: "f32[8, 151, 7, 7]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_195);  mul_365 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_29: "f32[8, 151, 7, 7]" = torch.ops.aten.slice.Tensor(add_251, 0, 0, 9223372036854775807);  add_251 = None
    slice_30: "f32[8, 140, 7, 7]" = torch.ops.aten.slice.Tensor(slice_29, 1, 0, 140)
    add_252: "f32[8, 140, 7, 7]" = torch.ops.aten.add.Tensor(slice_30, add_231);  slice_30 = None
    slice_32: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(slice_29, 1, 140, 9223372036854775807);  slice_29 = None
    cat_7: "f32[8, 151, 7, 7]" = torch.ops.aten.cat.default([add_252, slice_32], 1);  add_252 = slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 906, 7, 7]" = torch.ops.aten.convolution.default(cat_7, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_253: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 906, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 906, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_254: "f32[1, 906, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 906, 1, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
    sub_49: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_99)
    mul_366: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[906]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[906]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_367: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_368: "f32[906]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_255: "f32[906]" = torch.ops.aten.add.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    squeeze_149: "f32[906]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_369: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0025575447570332);  squeeze_149 = None
    mul_370: "f32[906]" = torch.ops.aten.mul.Tensor(mul_369, 0.1);  mul_369 = None
    mul_371: "f32[906]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_256: "f32[906]" = torch.ops.aten.add.Tensor(mul_370, mul_371);  mul_370 = mul_371 = None
    unsqueeze_196: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_197: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_372: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_197);  mul_366 = unsqueeze_197 = None
    unsqueeze_198: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_199: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_257: "f32[8, 906, 7, 7]" = torch.ops.aten.add.Tensor(mul_372, unsqueeze_199);  mul_372 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 906, 7, 7]" = torch.ops.aten.clone.default(add_257)
    sigmoid_23: "f32[8, 906, 7, 7]" = torch.ops.aten.sigmoid.default(add_257)
    mul_373: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(add_257, sigmoid_23);  add_257 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 906, 7, 7]" = torch.ops.aten.convolution.default(mul_373, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 906)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_258: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 906, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 906, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_259: "f32[1, 906, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 906, 1, 1]" = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
    sub_50: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_101)
    mul_374: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[906]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[906]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_375: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_376: "f32[906]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_260: "f32[906]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    squeeze_152: "f32[906]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_377: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0025575447570332);  squeeze_152 = None
    mul_378: "f32[906]" = torch.ops.aten.mul.Tensor(mul_377, 0.1);  mul_377 = None
    mul_379: "f32[906]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_261: "f32[906]" = torch.ops.aten.add.Tensor(mul_378, mul_379);  mul_378 = mul_379 = None
    unsqueeze_200: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_201: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_380: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(mul_374, unsqueeze_201);  mul_374 = unsqueeze_201 = None
    unsqueeze_202: "f32[906, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_203: "f32[906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_262: "f32[8, 906, 7, 7]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_203);  mul_380 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 906, 1, 1]" = torch.ops.aten.mean.dim(add_262, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_61: "f32[8, 75, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_200, primals_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_263: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 75, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 75, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_264: "f32[1, 75, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 75, 1, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
    sub_51: "f32[8, 75, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_103)
    mul_381: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
    squeeze_153: "f32[75]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    mul_382: "f32[75]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1);  squeeze_153 = None
    mul_383: "f32[75]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_265: "f32[75]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    squeeze_155: "f32[75]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_384: "f32[75]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.1428571428571428);  squeeze_155 = None
    mul_385: "f32[75]" = torch.ops.aten.mul.Tensor(mul_384, 0.1);  mul_384 = None
    mul_386: "f32[75]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_266: "f32[75]" = torch.ops.aten.add.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    unsqueeze_204: "f32[75, 1]" = torch.ops.aten.unsqueeze.default(primals_202, -1)
    unsqueeze_205: "f32[75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_387: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(mul_381, unsqueeze_205);  mul_381 = unsqueeze_205 = None
    unsqueeze_206: "f32[75, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1);  primals_203 = None
    unsqueeze_207: "f32[75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_267: "f32[8, 75, 1, 1]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_207);  mul_387 = unsqueeze_207 = None
    relu_10: "f32[8, 75, 1, 1]" = torch.ops.aten.relu.default(add_267);  add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_62: "f32[8, 906, 1, 1]" = torch.ops.aten.convolution.default(relu_10, primals_204, primals_205, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_24: "f32[8, 906, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_388: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(add_262, sigmoid_24);  sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_13: "f32[8, 906, 7, 7]" = torch.ops.aten.clamp_min.default(mul_388, 0.0);  mul_388 = None
    clamp_max_13: "f32[8, 906, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6.0);  clamp_min_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[8, 162, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_13, primals_206, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_268: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 162, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 162, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_269: "f32[1, 162, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_52: "f32[1, 162, 1, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
    sub_52: "f32[8, 162, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_105)
    mul_389: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[162]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[162]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_390: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_391: "f32[162]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_270: "f32[162]" = torch.ops.aten.add.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    squeeze_158: "f32[162]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_392: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0025575447570332);  squeeze_158 = None
    mul_393: "f32[162]" = torch.ops.aten.mul.Tensor(mul_392, 0.1);  mul_392 = None
    mul_394: "f32[162]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_271: "f32[162]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_208: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_209: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_395: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(mul_389, unsqueeze_209);  mul_389 = unsqueeze_209 = None
    unsqueeze_210: "f32[162, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_211: "f32[162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_272: "f32[8, 162, 7, 7]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_211);  mul_395 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_33: "f32[8, 162, 7, 7]" = torch.ops.aten.slice.Tensor(add_272, 0, 0, 9223372036854775807);  add_272 = None
    slice_34: "f32[8, 151, 7, 7]" = torch.ops.aten.slice.Tensor(slice_33, 1, 0, 151)
    add_273: "f32[8, 151, 7, 7]" = torch.ops.aten.add.Tensor(slice_34, cat_7);  slice_34 = None
    slice_36: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(slice_33, 1, 151, 9223372036854775807);  slice_33 = None
    cat_8: "f32[8, 162, 7, 7]" = torch.ops.aten.cat.default([add_273, slice_36], 1);  add_273 = slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 972, 7, 7]" = torch.ops.aten.convolution.default(cat_8, primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_274: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 972, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 972, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_275: "f32[1, 972, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_53: "f32[1, 972, 1, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
    sub_53: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_107)
    mul_396: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[972]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[972]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_397: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_398: "f32[972]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_276: "f32[972]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    squeeze_161: "f32[972]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_399: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0025575447570332);  squeeze_161 = None
    mul_400: "f32[972]" = torch.ops.aten.mul.Tensor(mul_399, 0.1);  mul_399 = None
    mul_401: "f32[972]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_277: "f32[972]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    unsqueeze_212: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_213: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_402: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_213);  mul_396 = unsqueeze_213 = None
    unsqueeze_214: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_215: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_278: "f32[8, 972, 7, 7]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_215);  mul_402 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_14: "f32[8, 972, 7, 7]" = torch.ops.aten.clone.default(add_278)
    sigmoid_25: "f32[8, 972, 7, 7]" = torch.ops.aten.sigmoid.default(add_278)
    mul_403: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(add_278, sigmoid_25);  add_278 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 972, 7, 7]" = torch.ops.aten.convolution.default(mul_403, primals_208, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 972)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_279: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 972, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 972, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_280: "f32[1, 972, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_54: "f32[1, 972, 1, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
    sub_54: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_109)
    mul_404: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[972]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[972]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_405: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_406: "f32[972]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_281: "f32[972]" = torch.ops.aten.add.Tensor(mul_405, mul_406);  mul_405 = mul_406 = None
    squeeze_164: "f32[972]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_407: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0025575447570332);  squeeze_164 = None
    mul_408: "f32[972]" = torch.ops.aten.mul.Tensor(mul_407, 0.1);  mul_407 = None
    mul_409: "f32[972]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_282: "f32[972]" = torch.ops.aten.add.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    unsqueeze_216: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_217: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_410: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(mul_404, unsqueeze_217);  mul_404 = unsqueeze_217 = None
    unsqueeze_218: "f32[972, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_219: "f32[972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_283: "f32[8, 972, 7, 7]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_219);  mul_410 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 972, 1, 1]" = torch.ops.aten.mean.dim(add_283, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_66: "f32[8, 81, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_209, primals_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_284: "i64[]" = torch.ops.aten.add.Tensor(primals_410, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 81, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 81, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_285: "f32[1, 81, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_55: "f32[1, 81, 1, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_55: "f32[8, 81, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_111)
    mul_411: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
    squeeze_165: "f32[81]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    mul_412: "f32[81]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1);  squeeze_165 = None
    mul_413: "f32[81]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_286: "f32[81]" = torch.ops.aten.add.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
    squeeze_167: "f32[81]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_414: "f32[81]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.1428571428571428);  squeeze_167 = None
    mul_415: "f32[81]" = torch.ops.aten.mul.Tensor(mul_414, 0.1);  mul_414 = None
    mul_416: "f32[81]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_287: "f32[81]" = torch.ops.aten.add.Tensor(mul_415, mul_416);  mul_415 = mul_416 = None
    unsqueeze_220: "f32[81, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1)
    unsqueeze_221: "f32[81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_417: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(mul_411, unsqueeze_221);  mul_411 = unsqueeze_221 = None
    unsqueeze_222: "f32[81, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1);  primals_212 = None
    unsqueeze_223: "f32[81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_288: "f32[8, 81, 1, 1]" = torch.ops.aten.add.Tensor(mul_417, unsqueeze_223);  mul_417 = unsqueeze_223 = None
    relu_11: "f32[8, 81, 1, 1]" = torch.ops.aten.relu.default(add_288);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_67: "f32[8, 972, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_213, primals_214, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_26: "f32[8, 972, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_418: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(add_283, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_14: "f32[8, 972, 7, 7]" = torch.ops.aten.clamp_min.default(mul_418, 0.0);  mul_418 = None
    clamp_max_14: "f32[8, 972, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6.0);  clamp_min_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_68: "f32[8, 174, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_14, primals_215, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_289: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 174, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 174, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_290: "f32[1, 174, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_56: "f32[1, 174, 1, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
    sub_56: "f32[8, 174, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_113)
    mul_419: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[174]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[174]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_420: "f32[174]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_421: "f32[174]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_291: "f32[174]" = torch.ops.aten.add.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    squeeze_170: "f32[174]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_422: "f32[174]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0025575447570332);  squeeze_170 = None
    mul_423: "f32[174]" = torch.ops.aten.mul.Tensor(mul_422, 0.1);  mul_422 = None
    mul_424: "f32[174]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_292: "f32[174]" = torch.ops.aten.add.Tensor(mul_423, mul_424);  mul_423 = mul_424 = None
    unsqueeze_224: "f32[174, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_225: "f32[174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_425: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_225);  mul_419 = unsqueeze_225 = None
    unsqueeze_226: "f32[174, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_227: "f32[174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_293: "f32[8, 174, 7, 7]" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_227);  mul_425 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_37: "f32[8, 174, 7, 7]" = torch.ops.aten.slice.Tensor(add_293, 0, 0, 9223372036854775807);  add_293 = None
    slice_38: "f32[8, 162, 7, 7]" = torch.ops.aten.slice.Tensor(slice_37, 1, 0, 162)
    add_294: "f32[8, 162, 7, 7]" = torch.ops.aten.add.Tensor(slice_38, cat_8);  slice_38 = None
    slice_40: "f32[8, 12, 7, 7]" = torch.ops.aten.slice.Tensor(slice_37, 1, 162, 9223372036854775807);  slice_37 = None
    cat_9: "f32[8, 174, 7, 7]" = torch.ops.aten.cat.default([add_294, slice_40], 1);  add_294 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_69: "f32[8, 1044, 7, 7]" = torch.ops.aten.convolution.default(cat_9, primals_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_295: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 1044, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 1044, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_296: "f32[1, 1044, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_57: "f32[1, 1044, 1, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
    sub_57: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_115)
    mul_426: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[1044]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_172: "f32[1044]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_427: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_428: "f32[1044]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_297: "f32[1044]" = torch.ops.aten.add.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
    squeeze_173: "f32[1044]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_429: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0025575447570332);  squeeze_173 = None
    mul_430: "f32[1044]" = torch.ops.aten.mul.Tensor(mul_429, 0.1);  mul_429 = None
    mul_431: "f32[1044]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_298: "f32[1044]" = torch.ops.aten.add.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
    unsqueeze_228: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_229: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_432: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_229);  mul_426 = unsqueeze_229 = None
    unsqueeze_230: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_231: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_299: "f32[8, 1044, 7, 7]" = torch.ops.aten.add.Tensor(mul_432, unsqueeze_231);  mul_432 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 1044, 7, 7]" = torch.ops.aten.clone.default(add_299)
    sigmoid_27: "f32[8, 1044, 7, 7]" = torch.ops.aten.sigmoid.default(add_299)
    mul_433: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(add_299, sigmoid_27);  add_299 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_70: "f32[8, 1044, 7, 7]" = torch.ops.aten.convolution.default(mul_433, primals_217, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1044)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_300: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 1044, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 1044, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_301: "f32[1, 1044, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_58: "f32[1, 1044, 1, 1]" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
    sub_58: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_117)
    mul_434: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[1044]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_175: "f32[1044]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_435: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_436: "f32[1044]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_302: "f32[1044]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_176: "f32[1044]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_437: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0025575447570332);  squeeze_176 = None
    mul_438: "f32[1044]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[1044]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_303: "f32[1044]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_232: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_233: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_440: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_233);  mul_434 = unsqueeze_233 = None
    unsqueeze_234: "f32[1044, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_235: "f32[1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_304: "f32[8, 1044, 7, 7]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_235);  mul_440 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 1044, 1, 1]" = torch.ops.aten.mean.dim(add_304, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_71: "f32[8, 87, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_218, primals_219, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    add_305: "i64[]" = torch.ops.aten.add.Tensor(primals_413, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 87, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 87, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_306: "f32[1, 87, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_59: "f32[1, 87, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
    sub_59: "f32[8, 87, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_119)
    mul_441: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
    squeeze_177: "f32[87]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    mul_442: "f32[87]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1);  squeeze_177 = None
    mul_443: "f32[87]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_307: "f32[87]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_179: "f32[87]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_444: "f32[87]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.1428571428571428);  squeeze_179 = None
    mul_445: "f32[87]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[87]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_308: "f32[87]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_236: "f32[87, 1]" = torch.ops.aten.unsqueeze.default(primals_220, -1)
    unsqueeze_237: "f32[87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_447: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_237);  mul_441 = unsqueeze_237 = None
    unsqueeze_238: "f32[87, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1);  primals_221 = None
    unsqueeze_239: "f32[87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_309: "f32[8, 87, 1, 1]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_239);  mul_447 = unsqueeze_239 = None
    relu_12: "f32[8, 87, 1, 1]" = torch.ops.aten.relu.default(add_309);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_72: "f32[8, 1044, 1, 1]" = torch.ops.aten.convolution.default(relu_12, primals_222, primals_223, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_28: "f32[8, 1044, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_448: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(add_304, sigmoid_28);  sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    clamp_min_15: "f32[8, 1044, 7, 7]" = torch.ops.aten.clamp_min.default(mul_448, 0.0);  mul_448 = None
    clamp_max_15: "f32[8, 1044, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6.0);  clamp_min_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_73: "f32[8, 185, 7, 7]" = torch.ops.aten.convolution.default(clamp_max_15, primals_224, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_310: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 185, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 185, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_311: "f32[1, 185, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_60: "f32[1, 185, 1, 1]" = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
    sub_60: "f32[8, 185, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_121)
    mul_449: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[185]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_181: "f32[185]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_450: "f32[185]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_451: "f32[185]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_312: "f32[185]" = torch.ops.aten.add.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    squeeze_182: "f32[185]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_452: "f32[185]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0025575447570332);  squeeze_182 = None
    mul_453: "f32[185]" = torch.ops.aten.mul.Tensor(mul_452, 0.1);  mul_452 = None
    mul_454: "f32[185]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_313: "f32[185]" = torch.ops.aten.add.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    unsqueeze_240: "f32[185, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_241: "f32[185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_455: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(mul_449, unsqueeze_241);  mul_449 = unsqueeze_241 = None
    unsqueeze_242: "f32[185, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_243: "f32[185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_314: "f32[8, 185, 7, 7]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_243);  mul_455 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_41: "f32[8, 185, 7, 7]" = torch.ops.aten.slice.Tensor(add_314, 0, 0, 9223372036854775807);  add_314 = None
    slice_42: "f32[8, 174, 7, 7]" = torch.ops.aten.slice.Tensor(slice_41, 1, 0, 174)
    add_315: "f32[8, 174, 7, 7]" = torch.ops.aten.add.Tensor(slice_42, cat_9);  slice_42 = None
    slice_44: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(slice_41, 1, 174, 9223372036854775807);  slice_41 = None
    cat_10: "f32[8, 185, 7, 7]" = torch.ops.aten.cat.default([add_315, slice_44], 1);  add_315 = slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_74: "f32[8, 1280, 7, 7]" = torch.ops.aten.convolution.default(cat_10, primals_225, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_316: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 1280, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 1280, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_317: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_61: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
    sub_61: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_123)
    mul_456: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_184: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_457: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_458: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_318: "f32[1280]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    squeeze_185: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_459: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0025575447570332);  squeeze_185 = None
    mul_460: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_459, 0.1);  mul_459 = None
    mul_461: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_319: "f32[1280]" = torch.ops.aten.add.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_244: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_245: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_462: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_245);  mul_456 = unsqueeze_245 = None
    unsqueeze_246: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_247: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_320: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_247);  mul_462 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 1280, 7, 7]" = torch.ops.aten.clone.default(add_320)
    sigmoid_29: "f32[8, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(add_320)
    mul_463: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_320, sigmoid_29);  add_320 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_13: "f32[8, 1280, 1, 1]" = torch.ops.aten.mean.dim(mul_463, [-1, -2], True);  mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1280]" = torch.ops.aten.view.default(mean_13, [8, 1280]);  mean_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:131, code: x = self.drop(x)
    clone_17: "f32[8, 1280]" = torch.ops.aten.clone.default(view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_227, clone_17, permute);  primals_227 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_30: "f32[8, 1280, 7, 7]" = torch.ops.aten.sigmoid.default(clone_16)
    full_default: "f32[8, 1280, 7, 7]" = torch.ops.aten.full.default([8, 1280, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_62: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_30);  full_default = None
    mul_464: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(clone_16, sub_62);  clone_16 = sub_62 = None
    add_321: "f32[8, 1280, 7, 7]" = torch.ops.aten.add.Scalar(mul_464, 1);  mul_464 = None
    mul_465: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_30, add_321);  sigmoid_30 = add_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_248: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_249: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
    unsqueeze_250: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_260: "f32[1, 185]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_261: "f32[1, 185, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_284: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_285: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_31: "f32[8, 1044, 7, 7]" = torch.ops.aten.sigmoid.default(clone_15)
    full_default_7: "f32[8, 1044, 7, 7]" = torch.ops.aten.full.default([8, 1044, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_80: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_7, sigmoid_31);  full_default_7 = None
    mul_507: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(clone_15, sub_80);  clone_15 = sub_80 = None
    add_324: "f32[8, 1044, 7, 7]" = torch.ops.aten.add.Scalar(mul_507, 1);  mul_507 = None
    mul_508: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_31, add_324);  sigmoid_31 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_296: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_297: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_308: "f32[1, 174]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_309: "f32[1, 174, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_332: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_333: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_32: "f32[8, 972, 7, 7]" = torch.ops.aten.sigmoid.default(clone_14)
    full_default_14: "f32[8, 972, 7, 7]" = torch.ops.aten.full.default([8, 972, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_98: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_32);  full_default_14 = None
    mul_550: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(clone_14, sub_98);  clone_14 = sub_98 = None
    add_328: "f32[8, 972, 7, 7]" = torch.ops.aten.add.Scalar(mul_550, 1);  mul_550 = None
    mul_551: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_32, add_328);  sigmoid_32 = add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_344: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_345: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_356: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_357: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_380: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_381: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 906, 7, 7]" = torch.ops.aten.sigmoid.default(clone_13)
    full_default_21: "f32[8, 906, 7, 7]" = torch.ops.aten.full.default([8, 906, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_116: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_21, sigmoid_33);  full_default_21 = None
    mul_593: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(clone_13, sub_116);  clone_13 = sub_116 = None
    add_332: "f32[8, 906, 7, 7]" = torch.ops.aten.add.Scalar(mul_593, 1);  mul_593 = None
    mul_594: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_33, add_332);  sigmoid_33 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_392: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_393: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_404: "f32[1, 151]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_405: "f32[1, 151, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_428: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_429: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_34: "f32[8, 840, 7, 7]" = torch.ops.aten.sigmoid.default(clone_12)
    full_default_28: "f32[8, 840, 7, 7]" = torch.ops.aten.full.default([8, 840, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_134: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_28, sigmoid_34);  full_default_28 = None
    mul_636: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(clone_12, sub_134);  clone_12 = sub_134 = None
    add_336: "f32[8, 840, 7, 7]" = torch.ops.aten.add.Scalar(mul_636, 1);  mul_636 = None
    mul_637: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_34, add_336);  sigmoid_34 = add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_440: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_441: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_452: "f32[1, 140]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_453: "f32[1, 140, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_476: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_477: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_35: "f32[8, 768, 14, 14]" = torch.ops.aten.sigmoid.default(clone_11)
    full_default_31: "f32[8, 768, 14, 14]" = torch.ops.aten.full.default([8, 768, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_152: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_31, sigmoid_35);  full_default_31 = None
    mul_679: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(clone_11, sub_152);  clone_11 = sub_152 = None
    add_339: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Scalar(mul_679, 1);  mul_679 = None
    mul_680: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_35, add_339);  sigmoid_35 = add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_488: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_489: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_500: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_501: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_524: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_525: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 702, 14, 14]" = torch.ops.aten.sigmoid.default(clone_10)
    full_default_38: "f32[8, 702, 14, 14]" = torch.ops.aten.full.default([8, 702, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_170: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_38, sigmoid_36);  full_default_38 = None
    mul_722: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(clone_10, sub_170);  clone_10 = sub_170 = None
    add_342: "f32[8, 702, 14, 14]" = torch.ops.aten.add.Scalar(mul_722, 1);  mul_722 = None
    mul_723: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_36, add_342);  sigmoid_36 = add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_536: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_537: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_548: "f32[1, 117]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_549: "f32[1, 117, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_572: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_573: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 636, 14, 14]" = torch.ops.aten.sigmoid.default(clone_9)
    full_default_45: "f32[8, 636, 14, 14]" = torch.ops.aten.full.default([8, 636, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_188: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_45, sigmoid_37);  full_default_45 = None
    mul_765: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(clone_9, sub_188);  clone_9 = sub_188 = None
    add_346: "f32[8, 636, 14, 14]" = torch.ops.aten.add.Scalar(mul_765, 1);  mul_765 = None
    mul_766: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_37, add_346);  sigmoid_37 = add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_584: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_585: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_596: "f32[1, 106]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_597: "f32[1, 106, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_620: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_621: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_38: "f32[8, 570, 14, 14]" = torch.ops.aten.sigmoid.default(clone_8)
    full_default_52: "f32[8, 570, 14, 14]" = torch.ops.aten.full.default([8, 570, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_206: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_52, sigmoid_38);  full_default_52 = None
    mul_808: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(clone_8, sub_206);  clone_8 = sub_206 = None
    add_350: "f32[8, 570, 14, 14]" = torch.ops.aten.add.Scalar(mul_808, 1);  mul_808 = None
    mul_809: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_38, add_350);  sigmoid_38 = add_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_632: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_633: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_644: "f32[1, 95]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_645: "f32[1, 95, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_668: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_669: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_39: "f32[8, 504, 14, 14]" = torch.ops.aten.sigmoid.default(clone_7)
    full_default_59: "f32[8, 504, 14, 14]" = torch.ops.aten.full.default([8, 504, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_224: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_59, sigmoid_39);  full_default_59 = None
    mul_851: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(clone_7, sub_224);  clone_7 = sub_224 = None
    add_354: "f32[8, 504, 14, 14]" = torch.ops.aten.add.Scalar(mul_851, 1);  mul_851 = None
    mul_852: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_39, add_354);  sigmoid_39 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_680: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_681: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_692: "f32[1, 84]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_693: "f32[1, 84, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_716: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_717: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 432, 14, 14]" = torch.ops.aten.sigmoid.default(clone_6)
    full_default_66: "f32[8, 432, 14, 14]" = torch.ops.aten.full.default([8, 432, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_242: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_66, sigmoid_40);  full_default_66 = None
    mul_894: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(clone_6, sub_242);  clone_6 = sub_242 = None
    add_358: "f32[8, 432, 14, 14]" = torch.ops.aten.add.Scalar(mul_894, 1);  mul_894 = None
    mul_895: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_40, add_358);  sigmoid_40 = add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_728: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_729: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_740: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_741: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_764: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_765: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[8, 366, 28, 28]" = torch.ops.aten.sigmoid.default(clone_5)
    full_default_69: "f32[8, 366, 28, 28]" = torch.ops.aten.full.default([8, 366, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_260: "f32[8, 366, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_69, sigmoid_41);  full_default_69 = None
    mul_937: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(clone_5, sub_260);  clone_5 = sub_260 = None
    add_361: "f32[8, 366, 28, 28]" = torch.ops.aten.add.Scalar(mul_937, 1);  mul_937 = None
    mul_938: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_41, add_361);  sigmoid_41 = add_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_776: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_777: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_788: "f32[1, 61]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_789: "f32[1, 61, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_812: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_813: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_42: "f32[8, 300, 28, 28]" = torch.ops.aten.sigmoid.default(clone_4)
    full_default_76: "f32[8, 300, 28, 28]" = torch.ops.aten.full.default([8, 300, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_278: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_76, sigmoid_42);  full_default_76 = None
    mul_980: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(clone_4, sub_278);  clone_4 = sub_278 = None
    add_364: "f32[8, 300, 28, 28]" = torch.ops.aten.add.Scalar(mul_980, 1);  mul_980 = None
    mul_981: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_42, add_364);  sigmoid_42 = add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_824: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_825: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_836: "f32[1, 50]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_837: "f32[1, 50, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_860: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_861: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_43: "f32[8, 228, 56, 56]" = torch.ops.aten.sigmoid.default(clone_3)
    full_default_79: "f32[8, 228, 56, 56]" = torch.ops.aten.full.default([8, 228, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_296: "f32[8, 228, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_79, sigmoid_43);  full_default_79 = None
    mul_1023: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(clone_3, sub_296);  clone_3 = sub_296 = None
    add_367: "f32[8, 228, 56, 56]" = torch.ops.aten.add.Scalar(mul_1023, 1);  mul_1023 = None
    mul_1024: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_43, add_367);  sigmoid_43 = add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_872: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_873: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_884: "f32[1, 38]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_885: "f32[1, 38, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_26: "b8[8, 162, 56, 56]" = torch.ops.aten.le.Scalar(add_39, 0.0)
    ge_13: "b8[8, 162, 56, 56]" = torch.ops.aten.ge.Scalar(add_39, 6.0);  add_39 = None
    bitwise_or_13: "b8[8, 162, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_26, ge_13);  le_26 = ge_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_896: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_897: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_44: "f32[8, 162, 56, 56]" = torch.ops.aten.sigmoid.default(clone_2)
    full_default_85: "f32[8, 162, 56, 56]" = torch.ops.aten.full.default([8, 162, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_309: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_85, sigmoid_44);  full_default_85 = None
    mul_1053: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(clone_2, sub_309);  clone_2 = sub_309 = None
    add_369: "f32[8, 162, 56, 56]" = torch.ops.aten.add.Scalar(mul_1053, 1);  mul_1053 = None
    mul_1054: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_44, add_369);  sigmoid_44 = add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_908: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_909: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_920: "f32[1, 27]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_921: "f32[1, 27, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_27: "b8[8, 96, 56, 56]" = torch.ops.aten.le.Scalar(add_24, 0.0)
    ge_14: "b8[8, 96, 56, 56]" = torch.ops.aten.ge.Scalar(add_24, 6.0);  add_24 = None
    bitwise_or_14: "b8[8, 96, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_27, ge_14);  le_27 = ge_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_932: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_933: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 96, 112, 112]" = torch.ops.aten.sigmoid.default(clone_1)
    full_default_87: "f32[8, 96, 112, 112]" = torch.ops.aten.full.default([8, 96, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_322: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_87, sigmoid_45);  full_default_87 = None
    mul_1083: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(clone_1, sub_322);  clone_1 = sub_322 = None
    add_371: "f32[8, 96, 112, 112]" = torch.ops.aten.add.Scalar(mul_1083, 1);  mul_1083 = None
    mul_1084: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_45, add_371);  sigmoid_45 = add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_944: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_945: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_956: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_957: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_28: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(add_9, 0.0)
    ge_15: "b8[8, 32, 112, 112]" = torch.ops.aten.ge.Scalar(add_9, 6.0);  add_9 = None
    bitwise_or_15: "b8[8, 32, 112, 112]" = torch.ops.aten.bitwise_or.Tensor(le_28, ge_15);  le_28 = ge_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_968: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_969: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_46: "f32[8, 32, 112, 112]" = torch.ops.aten.sigmoid.default(clone)
    full_default_89: "f32[8, 32, 112, 112]" = torch.ops.aten.full.default([8, 32, 112, 112], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_335: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(full_default_89, sigmoid_46);  full_default_89 = None
    mul_1113: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(clone, sub_335);  clone = sub_335 = None
    add_372: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Scalar(mul_1113, 1);  mul_1113 = None
    mul_1114: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sigmoid_46, add_372);  sigmoid_46 = add_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_980: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_981: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_228, add);  primals_228 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_229, add_2);  primals_229 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_230, add_3);  primals_230 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_5);  primals_231 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_232, add_7);  primals_232 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_233, add_8);  primals_233 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_10);  primals_234 = add_10 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_235, add_12);  primals_235 = add_12 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_236, add_13);  primals_236 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_15);  primals_237 = add_15 = None
    copy__10: "f32[96]" = torch.ops.aten.copy_.default(primals_238, add_17);  primals_238 = add_17 = None
    copy__11: "f32[96]" = torch.ops.aten.copy_.default(primals_239, add_18);  primals_239 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_20);  primals_240 = add_20 = None
    copy__13: "f32[96]" = torch.ops.aten.copy_.default(primals_241, add_22);  primals_241 = add_22 = None
    copy__14: "f32[96]" = torch.ops.aten.copy_.default(primals_242, add_23);  primals_242 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_25);  primals_243 = add_25 = None
    copy__16: "f32[27]" = torch.ops.aten.copy_.default(primals_244, add_27);  primals_244 = add_27 = None
    copy__17: "f32[27]" = torch.ops.aten.copy_.default(primals_245, add_28);  primals_245 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_30);  primals_246 = add_30 = None
    copy__19: "f32[162]" = torch.ops.aten.copy_.default(primals_247, add_32);  primals_247 = add_32 = None
    copy__20: "f32[162]" = torch.ops.aten.copy_.default(primals_248, add_33);  primals_248 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_35);  primals_249 = add_35 = None
    copy__22: "f32[162]" = torch.ops.aten.copy_.default(primals_250, add_37);  primals_250 = add_37 = None
    copy__23: "f32[162]" = torch.ops.aten.copy_.default(primals_251, add_38);  primals_251 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_40);  primals_252 = add_40 = None
    copy__25: "f32[38]" = torch.ops.aten.copy_.default(primals_253, add_42);  primals_253 = add_42 = None
    copy__26: "f32[38]" = torch.ops.aten.copy_.default(primals_254, add_43);  primals_254 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_46);  primals_255 = add_46 = None
    copy__28: "f32[228]" = torch.ops.aten.copy_.default(primals_256, add_48);  primals_256 = add_48 = None
    copy__29: "f32[228]" = torch.ops.aten.copy_.default(primals_257, add_49);  primals_257 = add_49 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_51);  primals_258 = add_51 = None
    copy__31: "f32[228]" = torch.ops.aten.copy_.default(primals_259, add_53);  primals_259 = add_53 = None
    copy__32: "f32[228]" = torch.ops.aten.copy_.default(primals_260, add_54);  primals_260 = add_54 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_61);  primals_261 = add_61 = None
    copy__34: "f32[50]" = torch.ops.aten.copy_.default(primals_262, add_63);  primals_262 = add_63 = None
    copy__35: "f32[50]" = torch.ops.aten.copy_.default(primals_263, add_64);  primals_263 = add_64 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_66);  primals_264 = add_66 = None
    copy__37: "f32[300]" = torch.ops.aten.copy_.default(primals_265, add_68);  primals_265 = add_68 = None
    copy__38: "f32[300]" = torch.ops.aten.copy_.default(primals_266, add_69);  primals_266 = add_69 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_71);  primals_267 = add_71 = None
    copy__40: "f32[300]" = torch.ops.aten.copy_.default(primals_268, add_73);  primals_268 = add_73 = None
    copy__41: "f32[300]" = torch.ops.aten.copy_.default(primals_269, add_74);  primals_269 = add_74 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_81);  primals_270 = add_81 = None
    copy__43: "f32[61]" = torch.ops.aten.copy_.default(primals_271, add_83);  primals_271 = add_83 = None
    copy__44: "f32[61]" = torch.ops.aten.copy_.default(primals_272, add_84);  primals_272 = add_84 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_87);  primals_273 = add_87 = None
    copy__46: "f32[366]" = torch.ops.aten.copy_.default(primals_274, add_89);  primals_274 = add_89 = None
    copy__47: "f32[366]" = torch.ops.aten.copy_.default(primals_275, add_90);  primals_275 = add_90 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_92);  primals_276 = add_92 = None
    copy__49: "f32[366]" = torch.ops.aten.copy_.default(primals_277, add_94);  primals_277 = add_94 = None
    copy__50: "f32[366]" = torch.ops.aten.copy_.default(primals_278, add_95);  primals_278 = add_95 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_102);  primals_279 = add_102 = None
    copy__52: "f32[72]" = torch.ops.aten.copy_.default(primals_280, add_104);  primals_280 = add_104 = None
    copy__53: "f32[72]" = torch.ops.aten.copy_.default(primals_281, add_105);  primals_281 = add_105 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_107);  primals_282 = add_107 = None
    copy__55: "f32[432]" = torch.ops.aten.copy_.default(primals_283, add_109);  primals_283 = add_109 = None
    copy__56: "f32[432]" = torch.ops.aten.copy_.default(primals_284, add_110);  primals_284 = add_110 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_112);  primals_285 = add_112 = None
    copy__58: "f32[432]" = torch.ops.aten.copy_.default(primals_286, add_114);  primals_286 = add_114 = None
    copy__59: "f32[432]" = torch.ops.aten.copy_.default(primals_287, add_115);  primals_287 = add_115 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_122);  primals_288 = add_122 = None
    copy__61: "f32[84]" = torch.ops.aten.copy_.default(primals_289, add_124);  primals_289 = add_124 = None
    copy__62: "f32[84]" = torch.ops.aten.copy_.default(primals_290, add_125);  primals_290 = add_125 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_128);  primals_291 = add_128 = None
    copy__64: "f32[504]" = torch.ops.aten.copy_.default(primals_292, add_130);  primals_292 = add_130 = None
    copy__65: "f32[504]" = torch.ops.aten.copy_.default(primals_293, add_131);  primals_293 = add_131 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_133);  primals_294 = add_133 = None
    copy__67: "f32[504]" = torch.ops.aten.copy_.default(primals_295, add_135);  primals_295 = add_135 = None
    copy__68: "f32[504]" = torch.ops.aten.copy_.default(primals_296, add_136);  primals_296 = add_136 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_143);  primals_297 = add_143 = None
    copy__70: "f32[95]" = torch.ops.aten.copy_.default(primals_298, add_145);  primals_298 = add_145 = None
    copy__71: "f32[95]" = torch.ops.aten.copy_.default(primals_299, add_146);  primals_299 = add_146 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_149);  primals_300 = add_149 = None
    copy__73: "f32[570]" = torch.ops.aten.copy_.default(primals_301, add_151);  primals_301 = add_151 = None
    copy__74: "f32[570]" = torch.ops.aten.copy_.default(primals_302, add_152);  primals_302 = add_152 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_154);  primals_303 = add_154 = None
    copy__76: "f32[570]" = torch.ops.aten.copy_.default(primals_304, add_156);  primals_304 = add_156 = None
    copy__77: "f32[570]" = torch.ops.aten.copy_.default(primals_305, add_157);  primals_305 = add_157 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_164);  primals_306 = add_164 = None
    copy__79: "f32[106]" = torch.ops.aten.copy_.default(primals_307, add_166);  primals_307 = add_166 = None
    copy__80: "f32[106]" = torch.ops.aten.copy_.default(primals_308, add_167);  primals_308 = add_167 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_170);  primals_309 = add_170 = None
    copy__82: "f32[636]" = torch.ops.aten.copy_.default(primals_310, add_172);  primals_310 = add_172 = None
    copy__83: "f32[636]" = torch.ops.aten.copy_.default(primals_311, add_173);  primals_311 = add_173 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_175);  primals_312 = add_175 = None
    copy__85: "f32[636]" = torch.ops.aten.copy_.default(primals_313, add_177);  primals_313 = add_177 = None
    copy__86: "f32[636]" = torch.ops.aten.copy_.default(primals_314, add_178);  primals_314 = add_178 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_185);  primals_315 = add_185 = None
    copy__88: "f32[117]" = torch.ops.aten.copy_.default(primals_316, add_187);  primals_316 = add_187 = None
    copy__89: "f32[117]" = torch.ops.aten.copy_.default(primals_317, add_188);  primals_317 = add_188 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_191);  primals_318 = add_191 = None
    copy__91: "f32[702]" = torch.ops.aten.copy_.default(primals_319, add_193);  primals_319 = add_193 = None
    copy__92: "f32[702]" = torch.ops.aten.copy_.default(primals_320, add_194);  primals_320 = add_194 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_196);  primals_321 = add_196 = None
    copy__94: "f32[702]" = torch.ops.aten.copy_.default(primals_322, add_198);  primals_322 = add_198 = None
    copy__95: "f32[702]" = torch.ops.aten.copy_.default(primals_323, add_199);  primals_323 = add_199 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_206);  primals_324 = add_206 = None
    copy__97: "f32[128]" = torch.ops.aten.copy_.default(primals_325, add_208);  primals_325 = add_208 = None
    copy__98: "f32[128]" = torch.ops.aten.copy_.default(primals_326, add_209);  primals_326 = add_209 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_212);  primals_327 = add_212 = None
    copy__100: "f32[768]" = torch.ops.aten.copy_.default(primals_328, add_214);  primals_328 = add_214 = None
    copy__101: "f32[768]" = torch.ops.aten.copy_.default(primals_329, add_215);  primals_329 = add_215 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_217);  primals_330 = add_217 = None
    copy__103: "f32[768]" = torch.ops.aten.copy_.default(primals_331, add_219);  primals_331 = add_219 = None
    copy__104: "f32[768]" = torch.ops.aten.copy_.default(primals_332, add_220);  primals_332 = add_220 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_227);  primals_333 = add_227 = None
    copy__106: "f32[140]" = torch.ops.aten.copy_.default(primals_334, add_229);  primals_334 = add_229 = None
    copy__107: "f32[140]" = torch.ops.aten.copy_.default(primals_335, add_230);  primals_335 = add_230 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_232);  primals_336 = add_232 = None
    copy__109: "f32[840]" = torch.ops.aten.copy_.default(primals_337, add_234);  primals_337 = add_234 = None
    copy__110: "f32[840]" = torch.ops.aten.copy_.default(primals_338, add_235);  primals_338 = add_235 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_237);  primals_339 = add_237 = None
    copy__112: "f32[840]" = torch.ops.aten.copy_.default(primals_340, add_239);  primals_340 = add_239 = None
    copy__113: "f32[840]" = torch.ops.aten.copy_.default(primals_341, add_240);  primals_341 = add_240 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_247);  primals_342 = add_247 = None
    copy__115: "f32[151]" = torch.ops.aten.copy_.default(primals_343, add_249);  primals_343 = add_249 = None
    copy__116: "f32[151]" = torch.ops.aten.copy_.default(primals_344, add_250);  primals_344 = add_250 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_253);  primals_345 = add_253 = None
    copy__118: "f32[906]" = torch.ops.aten.copy_.default(primals_346, add_255);  primals_346 = add_255 = None
    copy__119: "f32[906]" = torch.ops.aten.copy_.default(primals_347, add_256);  primals_347 = add_256 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_258);  primals_348 = add_258 = None
    copy__121: "f32[906]" = torch.ops.aten.copy_.default(primals_349, add_260);  primals_349 = add_260 = None
    copy__122: "f32[906]" = torch.ops.aten.copy_.default(primals_350, add_261);  primals_350 = add_261 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_268);  primals_351 = add_268 = None
    copy__124: "f32[162]" = torch.ops.aten.copy_.default(primals_352, add_270);  primals_352 = add_270 = None
    copy__125: "f32[162]" = torch.ops.aten.copy_.default(primals_353, add_271);  primals_353 = add_271 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_274);  primals_354 = add_274 = None
    copy__127: "f32[972]" = torch.ops.aten.copy_.default(primals_355, add_276);  primals_355 = add_276 = None
    copy__128: "f32[972]" = torch.ops.aten.copy_.default(primals_356, add_277);  primals_356 = add_277 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_279);  primals_357 = add_279 = None
    copy__130: "f32[972]" = torch.ops.aten.copy_.default(primals_358, add_281);  primals_358 = add_281 = None
    copy__131: "f32[972]" = torch.ops.aten.copy_.default(primals_359, add_282);  primals_359 = add_282 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_289);  primals_360 = add_289 = None
    copy__133: "f32[174]" = torch.ops.aten.copy_.default(primals_361, add_291);  primals_361 = add_291 = None
    copy__134: "f32[174]" = torch.ops.aten.copy_.default(primals_362, add_292);  primals_362 = add_292 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_295);  primals_363 = add_295 = None
    copy__136: "f32[1044]" = torch.ops.aten.copy_.default(primals_364, add_297);  primals_364 = add_297 = None
    copy__137: "f32[1044]" = torch.ops.aten.copy_.default(primals_365, add_298);  primals_365 = add_298 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_300);  primals_366 = add_300 = None
    copy__139: "f32[1044]" = torch.ops.aten.copy_.default(primals_367, add_302);  primals_367 = add_302 = None
    copy__140: "f32[1044]" = torch.ops.aten.copy_.default(primals_368, add_303);  primals_368 = add_303 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_310);  primals_369 = add_310 = None
    copy__142: "f32[185]" = torch.ops.aten.copy_.default(primals_370, add_312);  primals_370 = add_312 = None
    copy__143: "f32[185]" = torch.ops.aten.copy_.default(primals_371, add_313);  primals_371 = add_313 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_316);  primals_372 = add_316 = None
    copy__145: "f32[1280]" = torch.ops.aten.copy_.default(primals_373, add_318);  primals_373 = add_318 = None
    copy__146: "f32[1280]" = torch.ops.aten.copy_.default(primals_374, add_319);  primals_374 = add_319 = None
    copy__147: "f32[19]" = torch.ops.aten.copy_.default(primals_375, add_58);  primals_375 = add_58 = None
    copy__148: "f32[19]" = torch.ops.aten.copy_.default(primals_376, add_59);  primals_376 = add_59 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_377, add_56);  primals_377 = add_56 = None
    copy__150: "f32[25]" = torch.ops.aten.copy_.default(primals_378, add_78);  primals_378 = add_78 = None
    copy__151: "f32[25]" = torch.ops.aten.copy_.default(primals_379, add_79);  primals_379 = add_79 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_380, add_76);  primals_380 = add_76 = None
    copy__153: "f32[30]" = torch.ops.aten.copy_.default(primals_381, add_99);  primals_381 = add_99 = None
    copy__154: "f32[30]" = torch.ops.aten.copy_.default(primals_382, add_100);  primals_382 = add_100 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_383, add_97);  primals_383 = add_97 = None
    copy__156: "f32[36]" = torch.ops.aten.copy_.default(primals_384, add_119);  primals_384 = add_119 = None
    copy__157: "f32[36]" = torch.ops.aten.copy_.default(primals_385, add_120);  primals_385 = add_120 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_386, add_117);  primals_386 = add_117 = None
    copy__159: "f32[42]" = torch.ops.aten.copy_.default(primals_387, add_140);  primals_387 = add_140 = None
    copy__160: "f32[42]" = torch.ops.aten.copy_.default(primals_388, add_141);  primals_388 = add_141 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_389, add_138);  primals_389 = add_138 = None
    copy__162: "f32[47]" = torch.ops.aten.copy_.default(primals_390, add_161);  primals_390 = add_161 = None
    copy__163: "f32[47]" = torch.ops.aten.copy_.default(primals_391, add_162);  primals_391 = add_162 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_392, add_159);  primals_392 = add_159 = None
    copy__165: "f32[53]" = torch.ops.aten.copy_.default(primals_393, add_182);  primals_393 = add_182 = None
    copy__166: "f32[53]" = torch.ops.aten.copy_.default(primals_394, add_183);  primals_394 = add_183 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_180);  primals_395 = add_180 = None
    copy__168: "f32[58]" = torch.ops.aten.copy_.default(primals_396, add_203);  primals_396 = add_203 = None
    copy__169: "f32[58]" = torch.ops.aten.copy_.default(primals_397, add_204);  primals_397 = add_204 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_398, add_201);  primals_398 = add_201 = None
    copy__171: "f32[64]" = torch.ops.aten.copy_.default(primals_399, add_224);  primals_399 = add_224 = None
    copy__172: "f32[64]" = torch.ops.aten.copy_.default(primals_400, add_225);  primals_400 = add_225 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_222);  primals_401 = add_222 = None
    copy__174: "f32[70]" = torch.ops.aten.copy_.default(primals_402, add_244);  primals_402 = add_244 = None
    copy__175: "f32[70]" = torch.ops.aten.copy_.default(primals_403, add_245);  primals_403 = add_245 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_404, add_242);  primals_404 = add_242 = None
    copy__177: "f32[75]" = torch.ops.aten.copy_.default(primals_405, add_265);  primals_405 = add_265 = None
    copy__178: "f32[75]" = torch.ops.aten.copy_.default(primals_406, add_266);  primals_406 = add_266 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_263);  primals_407 = add_263 = None
    copy__180: "f32[81]" = torch.ops.aten.copy_.default(primals_408, add_286);  primals_408 = add_286 = None
    copy__181: "f32[81]" = torch.ops.aten.copy_.default(primals_409, add_287);  primals_409 = add_287 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_410, add_284);  primals_410 = add_284 = None
    copy__183: "f32[87]" = torch.ops.aten.copy_.default(primals_411, add_307);  primals_411 = add_307 = None
    copy__184: "f32[87]" = torch.ops.aten.copy_.default(primals_412, add_308);  primals_412 = add_308 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_413, add_305);  primals_413 = add_305 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_220, primals_222, primals_224, primals_225, primals_414, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, clamp_max, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, mul_29, convolution_4, squeeze_13, clamp_max_1, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, mul_51, convolution_7, squeeze_22, clamp_max_2, convolution_8, squeeze_25, cat, convolution_9, squeeze_28, mul_73, convolution_10, squeeze_31, add_55, mean, convolution_11, relu, convolution_12, clamp_max_3, convolution_13, squeeze_37, add_65, convolution_14, squeeze_40, mul_103, convolution_15, squeeze_43, add_75, mean_1, convolution_16, relu_1, convolution_17, clamp_max_4, convolution_18, squeeze_49, cat_1, convolution_19, squeeze_52, mul_133, convolution_20, squeeze_55, add_96, mean_2, convolution_21, relu_2, convolution_22, clamp_max_5, convolution_23, squeeze_61, add_106, convolution_24, squeeze_64, mul_163, convolution_25, squeeze_67, add_116, mean_3, convolution_26, relu_3, convolution_27, clamp_max_6, convolution_28, squeeze_73, cat_2, convolution_29, squeeze_76, mul_193, convolution_30, squeeze_79, add_137, mean_4, convolution_31, relu_4, convolution_32, clamp_max_7, convolution_33, squeeze_85, cat_3, convolution_34, squeeze_88, mul_223, convolution_35, squeeze_91, add_158, mean_5, convolution_36, relu_5, convolution_37, clamp_max_8, convolution_38, squeeze_97, cat_4, convolution_39, squeeze_100, mul_253, convolution_40, squeeze_103, add_179, mean_6, convolution_41, relu_6, convolution_42, clamp_max_9, convolution_43, squeeze_109, cat_5, convolution_44, squeeze_112, mul_283, convolution_45, squeeze_115, add_200, mean_7, convolution_46, relu_7, convolution_47, clamp_max_10, convolution_48, squeeze_121, cat_6, convolution_49, squeeze_124, mul_313, convolution_50, squeeze_127, add_221, mean_8, convolution_51, relu_8, convolution_52, clamp_max_11, convolution_53, squeeze_133, add_231, convolution_54, squeeze_136, mul_343, convolution_55, squeeze_139, add_241, mean_9, convolution_56, relu_9, convolution_57, clamp_max_12, convolution_58, squeeze_145, cat_7, convolution_59, squeeze_148, mul_373, convolution_60, squeeze_151, add_262, mean_10, convolution_61, relu_10, convolution_62, clamp_max_13, convolution_63, squeeze_157, cat_8, convolution_64, squeeze_160, mul_403, convolution_65, squeeze_163, add_283, mean_11, convolution_66, relu_11, convolution_67, clamp_max_14, convolution_68, squeeze_169, cat_9, convolution_69, squeeze_172, mul_433, convolution_70, squeeze_175, add_304, mean_12, convolution_71, relu_12, convolution_72, clamp_max_15, convolution_73, squeeze_181, cat_10, convolution_74, squeeze_184, clone_17, permute_1, mul_465, unsqueeze_250, unsqueeze_262, unsqueeze_286, mul_508, unsqueeze_298, unsqueeze_310, unsqueeze_334, mul_551, unsqueeze_346, unsqueeze_358, unsqueeze_382, mul_594, unsqueeze_394, unsqueeze_406, unsqueeze_430, mul_637, unsqueeze_442, unsqueeze_454, unsqueeze_478, mul_680, unsqueeze_490, unsqueeze_502, unsqueeze_526, mul_723, unsqueeze_538, unsqueeze_550, unsqueeze_574, mul_766, unsqueeze_586, unsqueeze_598, unsqueeze_622, mul_809, unsqueeze_634, unsqueeze_646, unsqueeze_670, mul_852, unsqueeze_682, unsqueeze_694, unsqueeze_718, mul_895, unsqueeze_730, unsqueeze_742, unsqueeze_766, mul_938, unsqueeze_778, unsqueeze_790, unsqueeze_814, mul_981, unsqueeze_826, unsqueeze_838, unsqueeze_862, mul_1024, unsqueeze_874, unsqueeze_886, bitwise_or_13, unsqueeze_898, mul_1054, unsqueeze_910, unsqueeze_922, bitwise_or_14, unsqueeze_934, mul_1084, unsqueeze_946, unsqueeze_958, bitwise_or_15, unsqueeze_970, mul_1114, unsqueeze_982]
    