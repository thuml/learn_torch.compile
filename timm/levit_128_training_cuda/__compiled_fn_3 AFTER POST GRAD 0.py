from __future__ import annotations



def forward(self, primals_1: "f32[4, 196]", primals_2: "f32[4, 196]", primals_3: "f32[4, 196]", primals_4: "f32[4, 196]", primals_5: "f32[8, 196]", primals_6: "f32[8, 49]", primals_7: "f32[8, 49]", primals_8: "f32[8, 49]", primals_9: "f32[8, 49]", primals_10: "f32[16, 49]", primals_11: "f32[12, 16]", primals_12: "f32[12, 16]", primals_13: "f32[12, 16]", primals_14: "f32[12, 16]", primals_15: "f32[16, 3, 3, 3]", primals_16: "f32[16]", primals_17: "f32[16]", primals_18: "f32[32, 16, 3, 3]", primals_19: "f32[32]", primals_20: "f32[32]", primals_21: "f32[64, 32, 3, 3]", primals_22: "f32[64]", primals_23: "f32[64]", primals_24: "f32[128, 64, 3, 3]", primals_25: "f32[128]", primals_26: "f32[128]", primals_27: "f32[256, 128]", primals_28: "f32[256]", primals_29: "f32[256]", primals_30: "f32[128, 128]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[256, 128]", primals_34: "f32[256]", primals_35: "f32[256]", primals_36: "f32[128, 256]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[256, 128]", primals_40: "f32[256]", primals_41: "f32[256]", primals_42: "f32[128, 128]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[256, 128]", primals_46: "f32[256]", primals_47: "f32[256]", primals_48: "f32[128, 256]", primals_49: "f32[128]", primals_50: "f32[128]", primals_51: "f32[256, 128]", primals_52: "f32[256]", primals_53: "f32[256]", primals_54: "f32[128, 128]", primals_55: "f32[128]", primals_56: "f32[128]", primals_57: "f32[256, 128]", primals_58: "f32[256]", primals_59: "f32[256]", primals_60: "f32[128, 256]", primals_61: "f32[128]", primals_62: "f32[128]", primals_63: "f32[256, 128]", primals_64: "f32[256]", primals_65: "f32[256]", primals_66: "f32[128, 128]", primals_67: "f32[128]", primals_68: "f32[128]", primals_69: "f32[256, 128]", primals_70: "f32[256]", primals_71: "f32[256]", primals_72: "f32[128, 256]", primals_73: "f32[128]", primals_74: "f32[128]", primals_75: "f32[640, 128]", primals_76: "f32[640]", primals_77: "f32[640]", primals_78: "f32[128, 128]", primals_79: "f32[128]", primals_80: "f32[128]", primals_81: "f32[256, 512]", primals_82: "f32[256]", primals_83: "f32[256]", primals_84: "f32[512, 256]", primals_85: "f32[512]", primals_86: "f32[512]", primals_87: "f32[256, 512]", primals_88: "f32[256]", primals_89: "f32[256]", primals_90: "f32[512, 256]", primals_91: "f32[512]", primals_92: "f32[512]", primals_93: "f32[256, 256]", primals_94: "f32[256]", primals_95: "f32[256]", primals_96: "f32[512, 256]", primals_97: "f32[512]", primals_98: "f32[512]", primals_99: "f32[256, 512]", primals_100: "f32[256]", primals_101: "f32[256]", primals_102: "f32[512, 256]", primals_103: "f32[512]", primals_104: "f32[512]", primals_105: "f32[256, 256]", primals_106: "f32[256]", primals_107: "f32[256]", primals_108: "f32[512, 256]", primals_109: "f32[512]", primals_110: "f32[512]", primals_111: "f32[256, 512]", primals_112: "f32[256]", primals_113: "f32[256]", primals_114: "f32[512, 256]", primals_115: "f32[512]", primals_116: "f32[512]", primals_117: "f32[256, 256]", primals_118: "f32[256]", primals_119: "f32[256]", primals_120: "f32[512, 256]", primals_121: "f32[512]", primals_122: "f32[512]", primals_123: "f32[256, 512]", primals_124: "f32[256]", primals_125: "f32[256]", primals_126: "f32[512, 256]", primals_127: "f32[512]", primals_128: "f32[512]", primals_129: "f32[256, 256]", primals_130: "f32[256]", primals_131: "f32[256]", primals_132: "f32[512, 256]", primals_133: "f32[512]", primals_134: "f32[512]", primals_135: "f32[256, 512]", primals_136: "f32[256]", primals_137: "f32[256]", primals_138: "f32[1280, 256]", primals_139: "f32[1280]", primals_140: "f32[1280]", primals_141: "f32[256, 256]", primals_142: "f32[256]", primals_143: "f32[256]", primals_144: "f32[384, 1024]", primals_145: "f32[384]", primals_146: "f32[384]", primals_147: "f32[768, 384]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[384, 768]", primals_151: "f32[384]", primals_152: "f32[384]", primals_153: "f32[768, 384]", primals_154: "f32[768]", primals_155: "f32[768]", primals_156: "f32[384, 384]", primals_157: "f32[384]", primals_158: "f32[384]", primals_159: "f32[768, 384]", primals_160: "f32[768]", primals_161: "f32[768]", primals_162: "f32[384, 768]", primals_163: "f32[384]", primals_164: "f32[384]", primals_165: "f32[768, 384]", primals_166: "f32[768]", primals_167: "f32[768]", primals_168: "f32[384, 384]", primals_169: "f32[384]", primals_170: "f32[384]", primals_171: "f32[768, 384]", primals_172: "f32[768]", primals_173: "f32[768]", primals_174: "f32[384, 768]", primals_175: "f32[384]", primals_176: "f32[384]", primals_177: "f32[768, 384]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[384, 384]", primals_181: "f32[384]", primals_182: "f32[384]", primals_183: "f32[768, 384]", primals_184: "f32[768]", primals_185: "f32[768]", primals_186: "f32[384, 768]", primals_187: "f32[384]", primals_188: "f32[384]", primals_189: "f32[768, 384]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[384, 384]", primals_193: "f32[384]", primals_194: "f32[384]", primals_195: "f32[768, 384]", primals_196: "f32[768]", primals_197: "f32[768]", primals_198: "f32[384, 768]", primals_199: "f32[384]", primals_200: "f32[384]", primals_201: "f32[384]", primals_202: "f32[384]", primals_203: "f32[1000, 384]", primals_204: "f32[1000]", primals_205: "f32[384]", primals_206: "f32[384]", primals_207: "f32[1000, 384]", primals_208: "f32[1000]", primals_209: "i64[196, 196]", primals_210: "i64[196, 196]", primals_211: "i64[196, 196]", primals_212: "i64[196, 196]", primals_213: "i64[49, 196]", primals_214: "i64[49, 49]", primals_215: "i64[49, 49]", primals_216: "i64[49, 49]", primals_217: "i64[49, 49]", primals_218: "i64[16, 49]", primals_219: "i64[16, 16]", primals_220: "i64[16, 16]", primals_221: "i64[16, 16]", primals_222: "i64[16, 16]", primals_223: "f32[16]", primals_224: "f32[16]", primals_225: "i64[]", primals_226: "f32[32]", primals_227: "f32[32]", primals_228: "i64[]", primals_229: "f32[64]", primals_230: "f32[64]", primals_231: "i64[]", primals_232: "f32[128]", primals_233: "f32[128]", primals_234: "i64[]", primals_235: "f32[256]", primals_236: "f32[256]", primals_237: "i64[]", primals_238: "f32[128]", primals_239: "f32[128]", primals_240: "i64[]", primals_241: "f32[256]", primals_242: "f32[256]", primals_243: "i64[]", primals_244: "f32[128]", primals_245: "f32[128]", primals_246: "i64[]", primals_247: "f32[256]", primals_248: "f32[256]", primals_249: "i64[]", primals_250: "f32[128]", primals_251: "f32[128]", primals_252: "i64[]", primals_253: "f32[256]", primals_254: "f32[256]", primals_255: "i64[]", primals_256: "f32[128]", primals_257: "f32[128]", primals_258: "i64[]", primals_259: "f32[256]", primals_260: "f32[256]", primals_261: "i64[]", primals_262: "f32[128]", primals_263: "f32[128]", primals_264: "i64[]", primals_265: "f32[256]", primals_266: "f32[256]", primals_267: "i64[]", primals_268: "f32[128]", primals_269: "f32[128]", primals_270: "i64[]", primals_271: "f32[256]", primals_272: "f32[256]", primals_273: "i64[]", primals_274: "f32[128]", primals_275: "f32[128]", primals_276: "i64[]", primals_277: "f32[256]", primals_278: "f32[256]", primals_279: "i64[]", primals_280: "f32[128]", primals_281: "f32[128]", primals_282: "i64[]", primals_283: "f32[640]", primals_284: "f32[640]", primals_285: "i64[]", primals_286: "f32[128]", primals_287: "f32[128]", primals_288: "i64[]", primals_289: "f32[256]", primals_290: "f32[256]", primals_291: "i64[]", primals_292: "f32[512]", primals_293: "f32[512]", primals_294: "i64[]", primals_295: "f32[256]", primals_296: "f32[256]", primals_297: "i64[]", primals_298: "f32[512]", primals_299: "f32[512]", primals_300: "i64[]", primals_301: "f32[256]", primals_302: "f32[256]", primals_303: "i64[]", primals_304: "f32[512]", primals_305: "f32[512]", primals_306: "i64[]", primals_307: "f32[256]", primals_308: "f32[256]", primals_309: "i64[]", primals_310: "f32[512]", primals_311: "f32[512]", primals_312: "i64[]", primals_313: "f32[256]", primals_314: "f32[256]", primals_315: "i64[]", primals_316: "f32[512]", primals_317: "f32[512]", primals_318: "i64[]", primals_319: "f32[256]", primals_320: "f32[256]", primals_321: "i64[]", primals_322: "f32[512]", primals_323: "f32[512]", primals_324: "i64[]", primals_325: "f32[256]", primals_326: "f32[256]", primals_327: "i64[]", primals_328: "f32[512]", primals_329: "f32[512]", primals_330: "i64[]", primals_331: "f32[256]", primals_332: "f32[256]", primals_333: "i64[]", primals_334: "f32[512]", primals_335: "f32[512]", primals_336: "i64[]", primals_337: "f32[256]", primals_338: "f32[256]", primals_339: "i64[]", primals_340: "f32[512]", primals_341: "f32[512]", primals_342: "i64[]", primals_343: "f32[256]", primals_344: "f32[256]", primals_345: "i64[]", primals_346: "f32[1280]", primals_347: "f32[1280]", primals_348: "i64[]", primals_349: "f32[256]", primals_350: "f32[256]", primals_351: "i64[]", primals_352: "f32[384]", primals_353: "f32[384]", primals_354: "i64[]", primals_355: "f32[768]", primals_356: "f32[768]", primals_357: "i64[]", primals_358: "f32[384]", primals_359: "f32[384]", primals_360: "i64[]", primals_361: "f32[768]", primals_362: "f32[768]", primals_363: "i64[]", primals_364: "f32[384]", primals_365: "f32[384]", primals_366: "i64[]", primals_367: "f32[768]", primals_368: "f32[768]", primals_369: "i64[]", primals_370: "f32[384]", primals_371: "f32[384]", primals_372: "i64[]", primals_373: "f32[768]", primals_374: "f32[768]", primals_375: "i64[]", primals_376: "f32[384]", primals_377: "f32[384]", primals_378: "i64[]", primals_379: "f32[768]", primals_380: "f32[768]", primals_381: "i64[]", primals_382: "f32[384]", primals_383: "f32[384]", primals_384: "i64[]", primals_385: "f32[768]", primals_386: "f32[768]", primals_387: "i64[]", primals_388: "f32[384]", primals_389: "f32[384]", primals_390: "i64[]", primals_391: "f32[768]", primals_392: "f32[768]", primals_393: "i64[]", primals_394: "f32[384]", primals_395: "f32[384]", primals_396: "i64[]", primals_397: "f32[768]", primals_398: "f32[768]", primals_399: "i64[]", primals_400: "f32[384]", primals_401: "f32[384]", primals_402: "i64[]", primals_403: "f32[768]", primals_404: "f32[768]", primals_405: "i64[]", primals_406: "f32[384]", primals_407: "f32[384]", primals_408: "i64[]", primals_409: "f32[384]", primals_410: "f32[384]", primals_411: "i64[]", primals_412: "f32[384]", primals_413: "f32[384]", primals_414: "i64[]", primals_415: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_415, primals_15, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_224, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1);  primals_17 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_5: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_7: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max);  clamp_max = None
    div: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_1: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(div, primals_18, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_228, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[32]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
    add_9: "f32[32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_10: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_11: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(add_10, 3)
    clamp_min_1: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
    clamp_max_1: "f32[8, 32, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_15: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, clamp_max_1);  clamp_max_1 = None
    div_1: "f32[8, 32, 56, 56]" = torch.ops.aten.div.Tensor(mul_15, 6);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_2: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(div_1, primals_21, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_13: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_2: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
    add_14: "f32[64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0001594642002871);  squeeze_8 = None
    mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(primals_230, 0.9)
    add_15: "f32[64]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1);  primals_23 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_16: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    add_17: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(add_16, 3)
    clamp_min_2: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_min.default(add_17, 0);  add_17 = None
    clamp_max_2: "f32[8, 64, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_23: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(add_16, clamp_max_2);  clamp_max_2 = None
    div_2: "f32[8, 64, 28, 28]" = torch.ops.aten.div.Tensor(mul_23, 6);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    convolution_3: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(div_2, primals_24, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_234, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_19: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_3: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_24: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
    add_20: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_27: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0006381620931717);  squeeze_11 = None
    mul_28: "f32[128]" = torch.ops.aten.mul.Tensor(mul_27, 0.1);  mul_27 = None
    mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
    add_21: "f32[128]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_30: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_22: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_15);  mul_30 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:639, code: x = x.flatten(2).transpose(1, 2)
    view: "f32[8, 128, 196]" = torch.ops.aten.reshape.default(add_22, [8, 128, 196]);  add_22 = None
    permute: "f32[8, 196, 128]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_1: "f32[128, 256]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    clone: "f32[8, 196, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
    view_1: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone, [1568, 128]);  clone = None
    mm: "f32[1568, 256]" = torch.ops.aten.mm.default(view_1, permute_1)
    view_2: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_3: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_2, [1568, 256]);  view_2 = None
    add_23: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(view_3, [0], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256]" = var_mean_4[0]
    getitem_9: "f32[1, 256]" = var_mean_4[1];  var_mean_4 = None
    add_24: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_4: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_3, getitem_9);  view_3 = None
    mul_31: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0]);  getitem_9 = None
    squeeze_13: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0]);  rsqrt_4 = None
    mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
    add_25: "f32[256]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    squeeze_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_8, [0]);  getitem_8 = None
    mul_34: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0006381620931717);  squeeze_14 = None
    mul_35: "f32[256]" = torch.ops.aten.mul.Tensor(mul_34, 0.1);  mul_34 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(primals_236, 0.9)
    add_26: "f32[256]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
    mul_37: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_31, primals_28);  mul_31 = None
    add_27: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_37, primals_29);  mul_37 = primals_29 = None
    view_4: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_27, [8, 196, 256]);  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_5: "f32[8, 196, 4, 64]" = torch.ops.aten.reshape.default(view_4, [8, 196, 4, -1]);  view_4 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_5, [16, 16, 32], 3);  view_5 = None
    getitem_10: "f32[8, 196, 4, 16]" = split_with_sizes[0]
    getitem_11: "f32[8, 196, 4, 16]" = split_with_sizes[1]
    getitem_12: "f32[8, 196, 4, 32]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_2: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_3: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 3, 1]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_4: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_2, [8, 4, 196, 16]);  permute_2 = None
    clone_1: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_6: "f32[32, 196, 16]" = torch.ops.aten.reshape.default(clone_1, [32, 196, 16]);  clone_1 = None
    expand_1: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 196]);  permute_3 = None
    clone_2: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_7: "f32[32, 16, 196]" = torch.ops.aten.reshape.default(clone_2, [32, 16, 196]);  clone_2 = None
    bmm: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_6, view_7)
    view_8: "f32[8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm, [8, 4, 196, 196]);  bmm = None
    mul_38: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_8, 0.25);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(primals_1, [None, primals_209]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_28: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_38, index);  mul_38 = index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
    sub_5: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_28, amax);  add_28 = amax = None
    exp: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_1: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_3: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_2: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_3, [8, 4, 196, 196]);  div_3 = None
    view_9: "f32[32, 196, 196]" = torch.ops.aten.reshape.default(expand_2, [32, 196, 196]);  expand_2 = None
    expand_3: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_4, [8, 4, 196, 32]);  permute_4 = None
    clone_3: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_10: "f32[32, 196, 32]" = torch.ops.aten.reshape.default(clone_3, [32, 196, 32]);  clone_3 = None
    bmm_1: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_1, [8, 4, 196, 32]);  bmm_1 = None
    permute_5: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_4: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_12: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(clone_4, [8, 196, 128]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_29: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_12, 3)
    clamp_min_3: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_29, 0);  add_29 = None
    clamp_max_3: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_39: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_12, clamp_max_3);  clamp_max_3 = None
    div_4: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_39, 6);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_6: "f32[128, 128]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    view_13: "f32[1568, 128]" = torch.ops.aten.reshape.default(div_4, [1568, 128]);  div_4 = None
    mm_1: "f32[1568, 128]" = torch.ops.aten.mm.default(view_13, permute_6)
    view_14: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_1, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_15: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_14, [1568, 128]);  view_14 = None
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_240, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(view_15, [0], correction = 0, keepdim = True)
    getitem_13: "f32[1, 128]" = var_mean_5[0]
    getitem_14: "f32[1, 128]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05)
    rsqrt_5: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_15, getitem_14);  view_15 = None
    mul_40: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    squeeze_15: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_14, [0]);  getitem_14 = None
    squeeze_16: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0]);  rsqrt_5 = None
    mul_41: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_42: "f32[128]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
    add_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
    squeeze_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0]);  getitem_13 = None
    mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0006381620931717);  squeeze_17 = None
    mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(mul_43, 0.1);  mul_43 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
    add_33: "f32[128]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    mul_46: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_40, primals_31);  mul_40 = None
    add_34: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_46, primals_32);  mul_46 = primals_32 = None
    view_16: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_34, [8, 196, 128]);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_35: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(permute, view_16);  permute = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_7: "f32[128, 256]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    clone_5: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
    view_17: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_5, [1568, 128]);  clone_5 = None
    mm_2: "f32[1568, 256]" = torch.ops.aten.mm.default(view_17, permute_7)
    view_18: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_2, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_19: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_18, [1568, 256]);  view_18 = None
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(view_19, [0], correction = 0, keepdim = True)
    getitem_15: "f32[1, 256]" = var_mean_6[0]
    getitem_16: "f32[1, 256]" = var_mean_6[1];  var_mean_6 = None
    add_37: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05)
    rsqrt_6: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_19, getitem_16);  view_19 = None
    mul_47: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0]);  getitem_16 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0]);  rsqrt_6 = None
    mul_48: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_49: "f32[256]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_48, mul_49);  mul_48 = mul_49 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0]);  getitem_15 = None
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0006381620931717);  squeeze_20 = None
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(mul_50, 0.1);  mul_50 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
    add_39: "f32[256]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    mul_53: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_47, primals_34);  mul_47 = None
    add_40: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_53, primals_35);  mul_53 = primals_35 = None
    view_20: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_40, [8, 196, 256]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_41: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_20, 3)
    clamp_min_4: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_41, 0);  add_41 = None
    clamp_max_4: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_54: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_20, clamp_max_4);  clamp_max_4 = None
    div_5: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_54, 6);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_8: "f32[256, 128]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    view_21: "f32[1568, 256]" = torch.ops.aten.reshape.default(div_5, [1568, 256]);  div_5 = None
    mm_3: "f32[1568, 128]" = torch.ops.aten.mm.default(view_21, permute_8)
    view_22: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_3, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_23: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_22, [1568, 128]);  view_22 = None
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_246, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(view_23, [0], correction = 0, keepdim = True)
    getitem_17: "f32[1, 128]" = var_mean_7[0]
    getitem_18: "f32[1, 128]" = var_mean_7[1];  var_mean_7 = None
    add_43: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_17, 1e-05)
    rsqrt_7: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_8: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_23, getitem_18);  view_23 = None
    mul_55: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_18, [0]);  getitem_18 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0]);  rsqrt_7 = None
    mul_56: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
    add_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_17, [0]);  getitem_17 = None
    mul_58: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0006381620931717);  squeeze_23 = None
    mul_59: "f32[128]" = torch.ops.aten.mul.Tensor(mul_58, 0.1);  mul_58 = None
    mul_60: "f32[128]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
    add_45: "f32[128]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
    mul_61: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_55, primals_37);  mul_55 = None
    add_46: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_61, primals_38);  mul_61 = primals_38 = None
    view_24: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_46, [8, 196, 128]);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_47: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_35, view_24);  add_35 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_9: "f32[128, 256]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    clone_7: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_47, memory_format = torch.contiguous_format)
    view_25: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_7, [1568, 128]);  clone_7 = None
    mm_4: "f32[1568, 256]" = torch.ops.aten.mm.default(view_25, permute_9)
    view_26: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_4, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_27: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_26, [1568, 256]);  view_26 = None
    add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(view_27, [0], correction = 0, keepdim = True)
    getitem_19: "f32[1, 256]" = var_mean_8[0]
    getitem_20: "f32[1, 256]" = var_mean_8[1];  var_mean_8 = None
    add_49: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_19, 1e-05)
    rsqrt_8: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_9: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_27, getitem_20);  view_27 = None
    mul_62: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
    squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_20, [0]);  getitem_20 = None
    squeeze_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0]);  rsqrt_8 = None
    mul_63: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_64: "f32[256]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
    add_50: "f32[256]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    squeeze_26: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_19, [0]);  getitem_19 = None
    mul_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0006381620931717);  squeeze_26 = None
    mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(mul_65, 0.1);  mul_65 = None
    mul_67: "f32[256]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
    add_51: "f32[256]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    mul_68: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_62, primals_40);  mul_62 = None
    add_52: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_68, primals_41);  mul_68 = primals_41 = None
    view_28: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_52, [8, 196, 256]);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_29: "f32[8, 196, 4, 64]" = torch.ops.aten.reshape.default(view_28, [8, 196, 4, -1]);  view_28 = None
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_29, [16, 16, 32], 3);  view_29 = None
    getitem_21: "f32[8, 196, 4, 16]" = split_with_sizes_1[0]
    getitem_22: "f32[8, 196, 4, 16]" = split_with_sizes_1[1]
    getitem_23: "f32[8, 196, 4, 32]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_10: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_11: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 3, 1]);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_12: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_23, [0, 2, 1, 3]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_4: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_10, [8, 4, 196, 16]);  permute_10 = None
    clone_8: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_30: "f32[32, 196, 16]" = torch.ops.aten.reshape.default(clone_8, [32, 196, 16]);  clone_8 = None
    expand_5: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_11, [8, 4, 16, 196]);  permute_11 = None
    clone_9: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_31: "f32[32, 16, 196]" = torch.ops.aten.reshape.default(clone_9, [32, 16, 196]);  clone_9 = None
    bmm_2: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_30, view_31)
    view_32: "f32[8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_2, [8, 4, 196, 196]);  bmm_2 = None
    mul_69: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_32, 0.25);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_1: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(primals_2, [None, primals_210]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_53: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_69, index_1);  mul_69 = index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_53, [-1], True)
    sub_10: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_53, amax_1);  add_53 = amax_1 = None
    exp_1: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_2: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_6: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_6, [8, 4, 196, 196]);  div_6 = None
    view_33: "f32[32, 196, 196]" = torch.ops.aten.reshape.default(expand_6, [32, 196, 196]);  expand_6 = None
    expand_7: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_12, [8, 4, 196, 32]);  permute_12 = None
    clone_10: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_34: "f32[32, 196, 32]" = torch.ops.aten.reshape.default(clone_10, [32, 196, 32]);  clone_10 = None
    bmm_3: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_3, [8, 4, 196, 32]);  bmm_3 = None
    permute_13: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    clone_11: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_36: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(clone_11, [8, 196, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_54: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_36, 3)
    clamp_min_5: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_54, 0);  add_54 = None
    clamp_max_5: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_70: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_36, clamp_max_5);  clamp_max_5 = None
    div_7: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_70, 6);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_14: "f32[128, 128]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    view_37: "f32[1568, 128]" = torch.ops.aten.reshape.default(div_7, [1568, 128]);  div_7 = None
    mm_5: "f32[1568, 128]" = torch.ops.aten.mm.default(view_37, permute_14)
    view_38: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_5, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_39: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_38, [1568, 128]);  view_38 = None
    add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_252, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(view_39, [0], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128]" = var_mean_9[0]
    getitem_25: "f32[1, 128]" = var_mean_9[1];  var_mean_9 = None
    add_56: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_9: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_11: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_39, getitem_25);  view_39 = None
    mul_71: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
    squeeze_27: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0]);  getitem_25 = None
    squeeze_28: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0]);  rsqrt_9 = None
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_73: "f32[128]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
    add_57: "f32[128]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_29: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0]);  getitem_24 = None
    mul_74: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0006381620931717);  squeeze_29 = None
    mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[128]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
    add_58: "f32[128]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    mul_77: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_71, primals_43);  mul_71 = None
    add_59: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_77, primals_44);  mul_77 = primals_44 = None
    view_40: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_59, [8, 196, 128]);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_60: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_47, view_40);  add_47 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_15: "f32[128, 256]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    clone_12: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    view_41: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_12, [1568, 128]);  clone_12 = None
    mm_6: "f32[1568, 256]" = torch.ops.aten.mm.default(view_41, permute_15)
    view_42: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_6, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_43: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_42, [1568, 256]);  view_42 = None
    add_61: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(view_43, [0], correction = 0, keepdim = True)
    getitem_26: "f32[1, 256]" = var_mean_10[0]
    getitem_27: "f32[1, 256]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_10: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_12: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_43, getitem_27);  view_43 = None
    mul_78: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
    squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_27, [0]);  getitem_27 = None
    squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0]);  rsqrt_10 = None
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
    add_63: "f32[256]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
    squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_26, [0]);  getitem_26 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0006381620931717);  squeeze_32 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(mul_81, 0.1);  mul_81 = None
    mul_83: "f32[256]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
    add_64: "f32[256]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    mul_84: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_78, primals_46);  mul_78 = None
    add_65: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_84, primals_47);  mul_84 = primals_47 = None
    view_44: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_65, [8, 196, 256]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_66: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_44, 3)
    clamp_min_6: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_66, 0);  add_66 = None
    clamp_max_6: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_85: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_44, clamp_max_6);  clamp_max_6 = None
    div_8: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_85, 6);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_16: "f32[256, 128]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_45: "f32[1568, 256]" = torch.ops.aten.reshape.default(div_8, [1568, 256]);  div_8 = None
    mm_7: "f32[1568, 128]" = torch.ops.aten.mm.default(view_45, permute_16)
    view_46: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_7, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_47: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_46, [1568, 128]);  view_46 = None
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_258, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(view_47, [0], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128]" = var_mean_11[0]
    getitem_29: "f32[1, 128]" = var_mean_11[1];  var_mean_11 = None
    add_68: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_11: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_47, getitem_29);  view_47 = None
    mul_86: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
    squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0]);  getitem_29 = None
    squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0]);  rsqrt_11 = None
    mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
    add_69: "f32[128]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_28, [0]);  getitem_28 = None
    mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0006381620931717);  squeeze_35 = None
    mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(mul_89, 0.1);  mul_89 = None
    mul_91: "f32[128]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
    add_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    mul_92: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_86, primals_49);  mul_86 = None
    add_71: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_92, primals_50);  mul_92 = primals_50 = None
    view_48: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_71, [8, 196, 128]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_72: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_60, view_48);  add_60 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_17: "f32[128, 256]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    clone_14: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_72, memory_format = torch.contiguous_format)
    view_49: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_14, [1568, 128]);  clone_14 = None
    mm_8: "f32[1568, 256]" = torch.ops.aten.mm.default(view_49, permute_17)
    view_50: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_8, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_51: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_50, [1568, 256]);  view_50 = None
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(view_51, [0], correction = 0, keepdim = True)
    getitem_30: "f32[1, 256]" = var_mean_12[0]
    getitem_31: "f32[1, 256]" = var_mean_12[1];  var_mean_12 = None
    add_74: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_12: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_51, getitem_31);  view_51 = None
    mul_93: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
    squeeze_36: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0]);  getitem_31 = None
    squeeze_37: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0]);  rsqrt_12 = None
    mul_94: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_95: "f32[256]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_75: "f32[256]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_38: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_30, [0]);  getitem_30 = None
    mul_96: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0006381620931717);  squeeze_38 = None
    mul_97: "f32[256]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[256]" = torch.ops.aten.mul.Tensor(primals_260, 0.9)
    add_76: "f32[256]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    mul_99: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_93, primals_52);  mul_93 = None
    add_77: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_99, primals_53);  mul_99 = primals_53 = None
    view_52: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_77, [8, 196, 256]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_53: "f32[8, 196, 4, 64]" = torch.ops.aten.reshape.default(view_52, [8, 196, 4, -1]);  view_52 = None
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_53, [16, 16, 32], 3);  view_53 = None
    getitem_32: "f32[8, 196, 4, 16]" = split_with_sizes_2[0]
    getitem_33: "f32[8, 196, 4, 16]" = split_with_sizes_2[1]
    getitem_34: "f32[8, 196, 4, 32]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_18: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_19: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_33, [0, 2, 3, 1]);  getitem_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_20: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_8: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_18, [8, 4, 196, 16]);  permute_18 = None
    clone_15: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_54: "f32[32, 196, 16]" = torch.ops.aten.reshape.default(clone_15, [32, 196, 16]);  clone_15 = None
    expand_9: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_19, [8, 4, 16, 196]);  permute_19 = None
    clone_16: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_55: "f32[32, 16, 196]" = torch.ops.aten.reshape.default(clone_16, [32, 16, 196]);  clone_16 = None
    bmm_4: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_54, view_55)
    view_56: "f32[8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_4, [8, 4, 196, 196]);  bmm_4 = None
    mul_100: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_56, 0.25);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_2: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(primals_3, [None, primals_211]);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_78: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_100, index_2);  mul_100 = index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
    sub_15: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_78, amax_2);  add_78 = amax_2 = None
    exp_2: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_3: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_9: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_10: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_9, [8, 4, 196, 196]);  div_9 = None
    view_57: "f32[32, 196, 196]" = torch.ops.aten.reshape.default(expand_10, [32, 196, 196]);  expand_10 = None
    expand_11: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_20, [8, 4, 196, 32]);  permute_20 = None
    clone_17: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_58: "f32[32, 196, 32]" = torch.ops.aten.reshape.default(clone_17, [32, 196, 32]);  clone_17 = None
    bmm_5: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_57, view_58)
    view_59: "f32[8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_5, [8, 4, 196, 32]);  bmm_5 = None
    permute_21: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    clone_18: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_60: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(clone_18, [8, 196, 128]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_79: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_60, 3)
    clamp_min_7: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_79, 0);  add_79 = None
    clamp_max_7: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_101: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_60, clamp_max_7);  clamp_max_7 = None
    div_10: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_101, 6);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_22: "f32[128, 128]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    view_61: "f32[1568, 128]" = torch.ops.aten.reshape.default(div_10, [1568, 128]);  div_10 = None
    mm_9: "f32[1568, 128]" = torch.ops.aten.mm.default(view_61, permute_22)
    view_62: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_9, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_63: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_62, [1568, 128]);  view_62 = None
    add_80: "i64[]" = torch.ops.aten.add.Tensor(primals_264, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(view_63, [0], correction = 0, keepdim = True)
    getitem_35: "f32[1, 128]" = var_mean_13[0]
    getitem_36: "f32[1, 128]" = var_mean_13[1];  var_mean_13 = None
    add_81: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05)
    rsqrt_13: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_16: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_63, getitem_36);  view_63 = None
    mul_102: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_13);  sub_16 = None
    squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0]);  getitem_36 = None
    squeeze_40: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0]);  rsqrt_13 = None
    mul_103: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_104: "f32[128]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_82: "f32[128]" = torch.ops.aten.add.Tensor(mul_103, mul_104);  mul_103 = mul_104 = None
    squeeze_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0]);  getitem_35 = None
    mul_105: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0006381620931717);  squeeze_41 = None
    mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(mul_105, 0.1);  mul_105 = None
    mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_83: "f32[128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    mul_108: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_102, primals_55);  mul_102 = None
    add_84: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_108, primals_56);  mul_108 = primals_56 = None
    view_64: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_84, [8, 196, 128]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_85: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_72, view_64);  add_72 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_23: "f32[128, 256]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    clone_19: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_85, memory_format = torch.contiguous_format)
    view_65: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_19, [1568, 128]);  clone_19 = None
    mm_10: "f32[1568, 256]" = torch.ops.aten.mm.default(view_65, permute_23)
    view_66: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_10, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_67: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_66, [1568, 256]);  view_66 = None
    add_86: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(view_67, [0], correction = 0, keepdim = True)
    getitem_37: "f32[1, 256]" = var_mean_14[0]
    getitem_38: "f32[1, 256]" = var_mean_14[1];  var_mean_14 = None
    add_87: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05)
    rsqrt_14: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_17: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_67, getitem_38);  view_67 = None
    mul_109: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_14);  sub_17 = None
    squeeze_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_38, [0]);  getitem_38 = None
    squeeze_43: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0]);  rsqrt_14 = None
    mul_110: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_88: "f32[256]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
    squeeze_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_37, [0]);  getitem_37 = None
    mul_112: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0006381620931717);  squeeze_44 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(mul_112, 0.1);  mul_112 = None
    mul_114: "f32[256]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_89: "f32[256]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    mul_115: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_109, primals_58);  mul_109 = None
    add_90: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_115, primals_59);  mul_115 = primals_59 = None
    view_68: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_90, [8, 196, 256]);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_91: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_68, 3)
    clamp_min_8: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_91, 0);  add_91 = None
    clamp_max_8: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_116: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_68, clamp_max_8);  clamp_max_8 = None
    div_11: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_116, 6);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_24: "f32[256, 128]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    view_69: "f32[1568, 256]" = torch.ops.aten.reshape.default(div_11, [1568, 256]);  div_11 = None
    mm_11: "f32[1568, 128]" = torch.ops.aten.mm.default(view_69, permute_24)
    view_70: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_11, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_71: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_70, [1568, 128]);  view_70 = None
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_270, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(view_71, [0], correction = 0, keepdim = True)
    getitem_39: "f32[1, 128]" = var_mean_15[0]
    getitem_40: "f32[1, 128]" = var_mean_15[1];  var_mean_15 = None
    add_93: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05)
    rsqrt_15: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_18: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_71, getitem_40);  view_71 = None
    mul_117: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_15);  sub_18 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0]);  getitem_40 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0]);  rsqrt_15 = None
    mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_119: "f32[128]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_94: "f32[128]" = torch.ops.aten.add.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0]);  getitem_39 = None
    mul_120: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0006381620931717);  squeeze_47 = None
    mul_121: "f32[128]" = torch.ops.aten.mul.Tensor(mul_120, 0.1);  mul_120 = None
    mul_122: "f32[128]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_95: "f32[128]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
    mul_123: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_117, primals_61);  mul_117 = None
    add_96: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_123, primals_62);  mul_123 = primals_62 = None
    view_72: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_96, [8, 196, 128]);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_97: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_85, view_72);  add_85 = view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_25: "f32[128, 256]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    clone_21: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format)
    view_73: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_21, [1568, 128]);  clone_21 = None
    mm_12: "f32[1568, 256]" = torch.ops.aten.mm.default(view_73, permute_25)
    view_74: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_12, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_75: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_74, [1568, 256]);  view_74 = None
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(view_75, [0], correction = 0, keepdim = True)
    getitem_41: "f32[1, 256]" = var_mean_16[0]
    getitem_42: "f32[1, 256]" = var_mean_16[1];  var_mean_16 = None
    add_99: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05)
    rsqrt_16: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_19: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_75, getitem_42);  view_75 = None
    mul_124: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_16);  sub_19 = None
    squeeze_48: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_42, [0]);  getitem_42 = None
    squeeze_49: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0]);  rsqrt_16 = None
    mul_125: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_126: "f32[256]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_100: "f32[256]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    squeeze_50: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_41, [0]);  getitem_41 = None
    mul_127: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0006381620931717);  squeeze_50 = None
    mul_128: "f32[256]" = torch.ops.aten.mul.Tensor(mul_127, 0.1);  mul_127 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_101: "f32[256]" = torch.ops.aten.add.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
    mul_130: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_124, primals_64);  mul_124 = None
    add_102: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_130, primals_65);  mul_130 = primals_65 = None
    view_76: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_102, [8, 196, 256]);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_77: "f32[8, 196, 4, 64]" = torch.ops.aten.reshape.default(view_76, [8, 196, 4, -1]);  view_76 = None
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_77, [16, 16, 32], 3);  view_77 = None
    getitem_43: "f32[8, 196, 4, 16]" = split_with_sizes_3[0]
    getitem_44: "f32[8, 196, 4, 16]" = split_with_sizes_3[1]
    getitem_45: "f32[8, 196, 4, 32]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_26: "f32[8, 4, 196, 16]" = torch.ops.aten.permute.default(getitem_43, [0, 2, 1, 3]);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_27: "f32[8, 4, 16, 196]" = torch.ops.aten.permute.default(getitem_44, [0, 2, 3, 1]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_28: "f32[8, 4, 196, 32]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_12: "f32[8, 4, 196, 16]" = torch.ops.aten.expand.default(permute_26, [8, 4, 196, 16]);  permute_26 = None
    clone_22: "f32[8, 4, 196, 16]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_78: "f32[32, 196, 16]" = torch.ops.aten.reshape.default(clone_22, [32, 196, 16]);  clone_22 = None
    expand_13: "f32[8, 4, 16, 196]" = torch.ops.aten.expand.default(permute_27, [8, 4, 16, 196]);  permute_27 = None
    clone_23: "f32[8, 4, 16, 196]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_79: "f32[32, 16, 196]" = torch.ops.aten.reshape.default(clone_23, [32, 16, 196]);  clone_23 = None
    bmm_6: "f32[32, 196, 196]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 4, 196, 196]" = torch.ops.aten.reshape.default(bmm_6, [8, 4, 196, 196]);  bmm_6 = None
    mul_131: "f32[8, 4, 196, 196]" = torch.ops.aten.mul.Tensor(view_80, 0.25);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_3: "f32[4, 196, 196]" = torch.ops.aten.index.Tensor(primals_4, [None, primals_212]);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_103: "f32[8, 4, 196, 196]" = torch.ops.aten.add.Tensor(mul_131, index_3);  mul_131 = index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 4, 196, 1]" = torch.ops.aten.amax.default(add_103, [-1], True)
    sub_20: "f32[8, 4, 196, 196]" = torch.ops.aten.sub.Tensor(add_103, amax_3);  add_103 = amax_3 = None
    exp_3: "f32[8, 4, 196, 196]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_4: "f32[8, 4, 196, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_12: "f32[8, 4, 196, 196]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_14: "f32[8, 4, 196, 196]" = torch.ops.aten.expand.default(div_12, [8, 4, 196, 196]);  div_12 = None
    view_81: "f32[32, 196, 196]" = torch.ops.aten.reshape.default(expand_14, [32, 196, 196]);  expand_14 = None
    expand_15: "f32[8, 4, 196, 32]" = torch.ops.aten.expand.default(permute_28, [8, 4, 196, 32]);  permute_28 = None
    clone_24: "f32[8, 4, 196, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_82: "f32[32, 196, 32]" = torch.ops.aten.reshape.default(clone_24, [32, 196, 32]);  clone_24 = None
    bmm_7: "f32[32, 196, 32]" = torch.ops.aten.bmm.default(view_81, view_82)
    view_83: "f32[8, 4, 196, 32]" = torch.ops.aten.reshape.default(bmm_7, [8, 4, 196, 32]);  bmm_7 = None
    permute_29: "f32[8, 196, 4, 32]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_25: "f32[8, 196, 4, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_84: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(clone_25, [8, 196, 128]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_104: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(view_84, 3)
    clamp_min_9: "f32[8, 196, 128]" = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
    clamp_max_9: "f32[8, 196, 128]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_132: "f32[8, 196, 128]" = torch.ops.aten.mul.Tensor(view_84, clamp_max_9);  clamp_max_9 = None
    div_13: "f32[8, 196, 128]" = torch.ops.aten.div.Tensor(mul_132, 6);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_30: "f32[128, 128]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    view_85: "f32[1568, 128]" = torch.ops.aten.reshape.default(div_13, [1568, 128]);  div_13 = None
    mm_13: "f32[1568, 128]" = torch.ops.aten.mm.default(view_85, permute_30)
    view_86: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_13, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_87: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_86, [1568, 128]);  view_86 = None
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_276, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(view_87, [0], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128]" = var_mean_17[0]
    getitem_47: "f32[1, 128]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_17: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_21: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_87, getitem_47);  view_87 = None
    mul_133: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_17);  sub_21 = None
    squeeze_51: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_47, [0]);  getitem_47 = None
    squeeze_52: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0]);  rsqrt_17 = None
    mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_107: "f32[128]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_53: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_46, [0]);  getitem_46 = None
    mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0006381620931717);  squeeze_53 = None
    mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_108: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    mul_139: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_133, primals_67);  mul_133 = None
    add_109: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_139, primals_68);  mul_139 = primals_68 = None
    view_88: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_109, [8, 196, 128]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_110: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_97, view_88);  add_97 = view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_31: "f32[128, 256]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    clone_26: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_110, memory_format = torch.contiguous_format)
    view_89: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_26, [1568, 128]);  clone_26 = None
    mm_14: "f32[1568, 256]" = torch.ops.aten.mm.default(view_89, permute_31)
    view_90: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(mm_14, [8, 196, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_91: "f32[1568, 256]" = torch.ops.aten.reshape.default(view_90, [1568, 256]);  view_90 = None
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(view_91, [0], correction = 0, keepdim = True)
    getitem_48: "f32[1, 256]" = var_mean_18[0]
    getitem_49: "f32[1, 256]" = var_mean_18[1];  var_mean_18 = None
    add_112: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_18: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_22: "f32[1568, 256]" = torch.ops.aten.sub.Tensor(view_91, getitem_49);  view_91 = None
    mul_140: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = None
    squeeze_54: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_49, [0]);  getitem_49 = None
    squeeze_55: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0]);  rsqrt_18 = None
    mul_141: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_142: "f32[256]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_56: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_48, [0]);  getitem_48 = None
    mul_143: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0006381620931717);  squeeze_56 = None
    mul_144: "f32[256]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[256]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_114: "f32[256]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    mul_146: "f32[1568, 256]" = torch.ops.aten.mul.Tensor(mul_140, primals_70);  mul_140 = None
    add_115: "f32[1568, 256]" = torch.ops.aten.add.Tensor(mul_146, primals_71);  mul_146 = primals_71 = None
    view_92: "f32[8, 196, 256]" = torch.ops.aten.reshape.default(add_115, [8, 196, 256]);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_116: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(view_92, 3)
    clamp_min_10: "f32[8, 196, 256]" = torch.ops.aten.clamp_min.default(add_116, 0);  add_116 = None
    clamp_max_10: "f32[8, 196, 256]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_147: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_92, clamp_max_10);  clamp_max_10 = None
    div_14: "f32[8, 196, 256]" = torch.ops.aten.div.Tensor(mul_147, 6);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_32: "f32[256, 128]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    view_93: "f32[1568, 256]" = torch.ops.aten.reshape.default(div_14, [1568, 256]);  div_14 = None
    mm_15: "f32[1568, 128]" = torch.ops.aten.mm.default(view_93, permute_32)
    view_94: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(mm_15, [8, 196, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_95: "f32[1568, 128]" = torch.ops.aten.reshape.default(view_94, [1568, 128]);  view_94 = None
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_282, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(view_95, [0], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128]" = var_mean_19[0]
    getitem_51: "f32[1, 128]" = var_mean_19[1];  var_mean_19 = None
    add_118: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_19: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_23: "f32[1568, 128]" = torch.ops.aten.sub.Tensor(view_95, getitem_51);  view_95 = None
    mul_148: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_19);  sub_23 = None
    squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_51, [0]);  getitem_51 = None
    squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0]);  rsqrt_19 = None
    mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_119: "f32[128]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_50, [0]);  getitem_50 = None
    mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0006381620931717);  squeeze_59 = None
    mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(mul_151, 0.1);  mul_151 = None
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_120: "f32[128]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    mul_154: "f32[1568, 128]" = torch.ops.aten.mul.Tensor(mul_148, primals_73);  mul_148 = None
    add_121: "f32[1568, 128]" = torch.ops.aten.add.Tensor(mul_154, primals_74);  mul_154 = primals_74 = None
    view_96: "f32[8, 196, 128]" = torch.ops.aten.reshape.default(add_121, [8, 196, 128]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_122: "f32[8, 196, 128]" = torch.ops.aten.add.Tensor(add_110, view_96);  add_110 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_33: "f32[128, 640]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    clone_28: "f32[8, 196, 128]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    view_97: "f32[1568, 128]" = torch.ops.aten.reshape.default(clone_28, [1568, 128]);  clone_28 = None
    mm_16: "f32[1568, 640]" = torch.ops.aten.mm.default(view_97, permute_33)
    view_98: "f32[8, 196, 640]" = torch.ops.aten.reshape.default(mm_16, [8, 196, 640])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_99: "f32[1568, 640]" = torch.ops.aten.reshape.default(view_98, [1568, 640]);  view_98 = None
    add_123: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(view_99, [0], correction = 0, keepdim = True)
    getitem_52: "f32[1, 640]" = var_mean_20[0]
    getitem_53: "f32[1, 640]" = var_mean_20[1];  var_mean_20 = None
    add_124: "f32[1, 640]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_20: "f32[1, 640]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_24: "f32[1568, 640]" = torch.ops.aten.sub.Tensor(view_99, getitem_53);  view_99 = None
    mul_155: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_20);  sub_24 = None
    squeeze_60: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_53, [0]);  getitem_53 = None
    squeeze_61: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0]);  rsqrt_20 = None
    mul_156: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_157: "f32[640]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_125: "f32[640]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_62: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_52, [0]);  getitem_52 = None
    mul_158: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0006381620931717);  squeeze_62 = None
    mul_159: "f32[640]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[640]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_126: "f32[640]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    mul_161: "f32[1568, 640]" = torch.ops.aten.mul.Tensor(mul_155, primals_76);  mul_155 = None
    add_127: "f32[1568, 640]" = torch.ops.aten.add.Tensor(mul_161, primals_77);  mul_161 = primals_77 = None
    view_100: "f32[8, 196, 640]" = torch.ops.aten.reshape.default(add_127, [8, 196, 640]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_101: "f32[8, 196, 8, 80]" = torch.ops.aten.reshape.default(view_100, [8, 196, 8, -1]);  view_100 = None
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_101, [16, 64], 3);  view_101 = None
    getitem_54: "f32[8, 196, 8, 16]" = split_with_sizes_4[0]
    getitem_55: "f32[8, 196, 8, 64]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_34: "f32[8, 8, 16, 196]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 3, 1]);  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_35: "f32[8, 8, 196, 64]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_102: "f32[8, 14, 14, 128]" = torch.ops.aten.reshape.default(add_122, [8, 14, 14, 128]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    slice_6: "f32[8, 7, 14, 128]" = torch.ops.aten.slice.Tensor(view_102, 1, 0, 9223372036854775807, 2);  view_102 = None
    slice_7: "f32[8, 7, 7, 128]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807, 2);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    clone_29: "f32[8, 7, 7, 128]" = torch.ops.aten.clone.default(slice_7, memory_format = torch.contiguous_format);  slice_7 = None
    view_103: "f32[8, 49, 128]" = torch.ops.aten.reshape.default(clone_29, [8, 49, 128]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_36: "f32[128, 128]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    view_104: "f32[392, 128]" = torch.ops.aten.reshape.default(view_103, [392, 128]);  view_103 = None
    mm_17: "f32[392, 128]" = torch.ops.aten.mm.default(view_104, permute_36)
    view_105: "f32[8, 49, 128]" = torch.ops.aten.reshape.default(mm_17, [8, 49, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_106: "f32[392, 128]" = torch.ops.aten.reshape.default(view_105, [392, 128]);  view_105 = None
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(view_106, [0], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128]" = var_mean_21[0]
    getitem_57: "f32[1, 128]" = var_mean_21[1];  var_mean_21 = None
    add_129: "f32[1, 128]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_21: "f32[1, 128]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_25: "f32[392, 128]" = torch.ops.aten.sub.Tensor(view_106, getitem_57);  view_106 = None
    mul_162: "f32[392, 128]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_21);  sub_25 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_57, [0]);  getitem_57 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0]);  rsqrt_21 = None
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_164: "f32[128]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_130: "f32[128]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_56, [0]);  getitem_56 = None
    mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0025575447570332);  squeeze_65 = None
    mul_166: "f32[128]" = torch.ops.aten.mul.Tensor(mul_165, 0.1);  mul_165 = None
    mul_167: "f32[128]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_131: "f32[128]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
    mul_168: "f32[392, 128]" = torch.ops.aten.mul.Tensor(mul_162, primals_79);  mul_162 = None
    add_132: "f32[392, 128]" = torch.ops.aten.add.Tensor(mul_168, primals_80);  mul_168 = primals_80 = None
    view_107: "f32[8, 49, 128]" = torch.ops.aten.reshape.default(add_132, [8, 49, 128]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_108: "f32[8, 49, 8, 16]" = torch.ops.aten.reshape.default(view_107, [8, -1, 8, 16]);  view_107 = None
    permute_37: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_16: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_37, [8, 8, 49, 16]);  permute_37 = None
    clone_30: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_109: "f32[64, 49, 16]" = torch.ops.aten.reshape.default(clone_30, [64, 49, 16]);  clone_30 = None
    expand_17: "f32[8, 8, 16, 196]" = torch.ops.aten.expand.default(permute_34, [8, 8, 16, 196]);  permute_34 = None
    clone_31: "f32[8, 8, 16, 196]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_110: "f32[64, 16, 196]" = torch.ops.aten.reshape.default(clone_31, [64, 16, 196]);  clone_31 = None
    bmm_8: "f32[64, 49, 196]" = torch.ops.aten.bmm.default(view_109, view_110)
    view_111: "f32[8, 8, 49, 196]" = torch.ops.aten.reshape.default(bmm_8, [8, 8, 49, 196]);  bmm_8 = None
    mul_169: "f32[8, 8, 49, 196]" = torch.ops.aten.mul.Tensor(view_111, 0.25);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_4: "f32[8, 49, 196]" = torch.ops.aten.index.Tensor(primals_5, [None, primals_213]);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_133: "f32[8, 8, 49, 196]" = torch.ops.aten.add.Tensor(mul_169, index_4);  mul_169 = index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_133, [-1], True)
    sub_26: "f32[8, 8, 49, 196]" = torch.ops.aten.sub.Tensor(add_133, amax_4);  add_133 = amax_4 = None
    exp_4: "f32[8, 8, 49, 196]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_5: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_15: "f32[8, 8, 49, 196]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 8, 49, 196]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    expand_18: "f32[8, 8, 49, 196]" = torch.ops.aten.expand.default(div_15, [8, 8, 49, 196]);  div_15 = None
    view_112: "f32[64, 49, 196]" = torch.ops.aten.reshape.default(expand_18, [64, 49, 196]);  expand_18 = None
    expand_19: "f32[8, 8, 196, 64]" = torch.ops.aten.expand.default(permute_35, [8, 8, 196, 64]);  permute_35 = None
    clone_32: "f32[8, 8, 196, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_113: "f32[64, 196, 64]" = torch.ops.aten.reshape.default(clone_32, [64, 196, 64]);  clone_32 = None
    bmm_9: "f32[64, 49, 64]" = torch.ops.aten.bmm.default(view_112, view_113)
    view_114: "f32[8, 8, 49, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 8, 49, 64]);  bmm_9 = None
    permute_38: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    clone_33: "f32[8, 49, 8, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_115: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(clone_33, [8, 49, 512]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    add_134: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_115, 3)
    clamp_min_11: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_134, 0);  add_134 = None
    clamp_max_11: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_170: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_115, clamp_max_11);  clamp_max_11 = None
    div_16: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_170, 6);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_39: "f32[512, 256]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_116: "f32[392, 512]" = torch.ops.aten.reshape.default(div_16, [392, 512]);  div_16 = None
    mm_18: "f32[392, 256]" = torch.ops.aten.mm.default(view_116, permute_39)
    view_117: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_18, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_118: "f32[392, 256]" = torch.ops.aten.reshape.default(view_117, [392, 256]);  view_117 = None
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(view_118, [0], correction = 0, keepdim = True)
    getitem_58: "f32[1, 256]" = var_mean_22[0]
    getitem_59: "f32[1, 256]" = var_mean_22[1];  var_mean_22 = None
    add_136: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_22: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_27: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_118, getitem_59);  view_118 = None
    mul_171: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_22);  sub_27 = None
    squeeze_66: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_59, [0]);  getitem_59 = None
    squeeze_67: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0]);  rsqrt_22 = None
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_137: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_68: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_58, [0]);  getitem_58 = None
    mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0025575447570332);  squeeze_68 = None
    mul_175: "f32[256]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_138: "f32[256]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    mul_177: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_171, primals_82);  mul_171 = None
    add_139: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_177, primals_83);  mul_177 = primals_83 = None
    view_119: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_139, [8, 49, 256]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_40: "f32[256, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_120: "f32[392, 256]" = torch.ops.aten.reshape.default(view_119, [392, 256])
    mm_19: "f32[392, 512]" = torch.ops.aten.mm.default(view_120, permute_40)
    view_121: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_19, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_122: "f32[392, 512]" = torch.ops.aten.reshape.default(view_121, [392, 512]);  view_121 = None
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(view_122, [0], correction = 0, keepdim = True)
    getitem_60: "f32[1, 512]" = var_mean_23[0]
    getitem_61: "f32[1, 512]" = var_mean_23[1];  var_mean_23 = None
    add_141: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_23: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_28: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_122, getitem_61);  view_122 = None
    mul_178: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_23);  sub_28 = None
    squeeze_69: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_61, [0]);  getitem_61 = None
    squeeze_70: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0]);  rsqrt_23 = None
    mul_179: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_180: "f32[512]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_142: "f32[512]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    squeeze_71: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_60, [0]);  getitem_60 = None
    mul_181: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0025575447570332);  squeeze_71 = None
    mul_182: "f32[512]" = torch.ops.aten.mul.Tensor(mul_181, 0.1);  mul_181 = None
    mul_183: "f32[512]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_143: "f32[512]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    mul_184: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_178, primals_85);  mul_178 = None
    add_144: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_184, primals_86);  mul_184 = primals_86 = None
    view_123: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_144, [8, 49, 512]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_145: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_123, 3)
    clamp_min_12: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_145, 0);  add_145 = None
    clamp_max_12: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_185: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_123, clamp_max_12);  clamp_max_12 = None
    div_17: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_185, 6);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_41: "f32[512, 256]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    view_124: "f32[392, 512]" = torch.ops.aten.reshape.default(div_17, [392, 512]);  div_17 = None
    mm_20: "f32[392, 256]" = torch.ops.aten.mm.default(view_124, permute_41)
    view_125: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_20, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_126: "f32[392, 256]" = torch.ops.aten.reshape.default(view_125, [392, 256]);  view_125 = None
    add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(view_126, [0], correction = 0, keepdim = True)
    getitem_62: "f32[1, 256]" = var_mean_24[0]
    getitem_63: "f32[1, 256]" = var_mean_24[1];  var_mean_24 = None
    add_147: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_24: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_29: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_126, getitem_63);  view_126 = None
    mul_186: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_24);  sub_29 = None
    squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_63, [0]);  getitem_63 = None
    squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0]);  rsqrt_24 = None
    mul_187: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_188: "f32[256]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_148: "f32[256]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
    squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_62, [0]);  getitem_62 = None
    mul_189: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0025575447570332);  squeeze_74 = None
    mul_190: "f32[256]" = torch.ops.aten.mul.Tensor(mul_189, 0.1);  mul_189 = None
    mul_191: "f32[256]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_149: "f32[256]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    mul_192: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_186, primals_88);  mul_186 = None
    add_150: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_192, primals_89);  mul_192 = primals_89 = None
    view_127: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_150, [8, 49, 256]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    add_151: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_119, view_127);  view_119 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_42: "f32[256, 512]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    view_128: "f32[392, 256]" = torch.ops.aten.reshape.default(add_151, [392, 256])
    mm_21: "f32[392, 512]" = torch.ops.aten.mm.default(view_128, permute_42)
    view_129: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_21, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_130: "f32[392, 512]" = torch.ops.aten.reshape.default(view_129, [392, 512]);  view_129 = None
    add_152: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(view_130, [0], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512]" = var_mean_25[0]
    getitem_65: "f32[1, 512]" = var_mean_25[1];  var_mean_25 = None
    add_153: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_25: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_30: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_130, getitem_65);  view_130 = None
    mul_193: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_25);  sub_30 = None
    squeeze_75: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_65, [0]);  getitem_65 = None
    squeeze_76: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0]);  rsqrt_25 = None
    mul_194: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_195: "f32[512]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_154: "f32[512]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    squeeze_77: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_64, [0]);  getitem_64 = None
    mul_196: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0025575447570332);  squeeze_77 = None
    mul_197: "f32[512]" = torch.ops.aten.mul.Tensor(mul_196, 0.1);  mul_196 = None
    mul_198: "f32[512]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_155: "f32[512]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    mul_199: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_193, primals_91);  mul_193 = None
    add_156: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_199, primals_92);  mul_199 = primals_92 = None
    view_131: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_156, [8, 49, 512]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_132: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_131, [8, 49, 8, -1]);  view_131 = None
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_132, [16, 16, 32], 3);  view_132 = None
    getitem_66: "f32[8, 49, 8, 16]" = split_with_sizes_5[0]
    getitem_67: "f32[8, 49, 8, 16]" = split_with_sizes_5[1]
    getitem_68: "f32[8, 49, 8, 32]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_43: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_44: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_67, [0, 2, 3, 1]);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_45: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3]);  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_20: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_43, [8, 8, 49, 16]);  permute_43 = None
    clone_35: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_133: "f32[64, 49, 16]" = torch.ops.aten.reshape.default(clone_35, [64, 49, 16]);  clone_35 = None
    expand_21: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_44, [8, 8, 16, 49]);  permute_44 = None
    clone_36: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_134: "f32[64, 16, 49]" = torch.ops.aten.reshape.default(clone_36, [64, 16, 49]);  clone_36 = None
    bmm_10: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_133, view_134)
    view_135: "f32[8, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_10, [8, 8, 49, 49]);  bmm_10 = None
    mul_200: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_135, 0.25);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_5: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(primals_6, [None, primals_214]);  primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_157: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_200, index_5);  mul_200 = index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
    sub_31: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_157, amax_5);  add_157 = amax_5 = None
    exp_5: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_6: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_18: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_22: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_18, [8, 8, 49, 49]);  div_18 = None
    view_136: "f32[64, 49, 49]" = torch.ops.aten.reshape.default(expand_22, [64, 49, 49]);  expand_22 = None
    expand_23: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_45, [8, 8, 49, 32]);  permute_45 = None
    clone_37: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_137: "f32[64, 49, 32]" = torch.ops.aten.reshape.default(clone_37, [64, 49, 32]);  clone_37 = None
    bmm_11: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_136, view_137)
    view_138: "f32[8, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_11, [8, 8, 49, 32]);  bmm_11 = None
    permute_46: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_38: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_139: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(clone_38, [8, 49, 256]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_158: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_139, 3)
    clamp_min_13: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_158, 0);  add_158 = None
    clamp_max_13: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_201: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_139, clamp_max_13);  clamp_max_13 = None
    div_19: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_201, 6);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    view_140: "f32[392, 256]" = torch.ops.aten.reshape.default(div_19, [392, 256]);  div_19 = None
    mm_22: "f32[392, 256]" = torch.ops.aten.mm.default(view_140, permute_47)
    view_141: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_22, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_142: "f32[392, 256]" = torch.ops.aten.reshape.default(view_141, [392, 256]);  view_141 = None
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(view_142, [0], correction = 0, keepdim = True)
    getitem_69: "f32[1, 256]" = var_mean_26[0]
    getitem_70: "f32[1, 256]" = var_mean_26[1];  var_mean_26 = None
    add_160: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05)
    rsqrt_26: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_32: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_142, getitem_70);  view_142 = None
    mul_202: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_26);  sub_32 = None
    squeeze_78: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_70, [0]);  getitem_70 = None
    squeeze_79: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0]);  rsqrt_26 = None
    mul_203: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_161: "f32[256]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    squeeze_80: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_69, [0]);  getitem_69 = None
    mul_205: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0025575447570332);  squeeze_80 = None
    mul_206: "f32[256]" = torch.ops.aten.mul.Tensor(mul_205, 0.1);  mul_205 = None
    mul_207: "f32[256]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_162: "f32[256]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    mul_208: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_202, primals_94);  mul_202 = None
    add_163: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_208, primals_95);  mul_208 = primals_95 = None
    view_143: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_163, [8, 49, 256]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_164: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_151, view_143);  add_151 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_48: "f32[256, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    view_144: "f32[392, 256]" = torch.ops.aten.reshape.default(add_164, [392, 256])
    mm_23: "f32[392, 512]" = torch.ops.aten.mm.default(view_144, permute_48)
    view_145: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_23, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_146: "f32[392, 512]" = torch.ops.aten.reshape.default(view_145, [392, 512]);  view_145 = None
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(view_146, [0], correction = 0, keepdim = True)
    getitem_71: "f32[1, 512]" = var_mean_27[0]
    getitem_72: "f32[1, 512]" = var_mean_27[1];  var_mean_27 = None
    add_166: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_71, 1e-05)
    rsqrt_27: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_33: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_146, getitem_72);  view_146 = None
    mul_209: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_27);  sub_33 = None
    squeeze_81: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_72, [0]);  getitem_72 = None
    squeeze_82: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0]);  rsqrt_27 = None
    mul_210: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_211: "f32[512]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_167: "f32[512]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    squeeze_83: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_71, [0]);  getitem_71 = None
    mul_212: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0025575447570332);  squeeze_83 = None
    mul_213: "f32[512]" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
    mul_214: "f32[512]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_168: "f32[512]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    mul_215: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_209, primals_97);  mul_209 = None
    add_169: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_215, primals_98);  mul_215 = primals_98 = None
    view_147: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_169, [8, 49, 512]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_170: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_147, 3)
    clamp_min_14: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_170, 0);  add_170 = None
    clamp_max_14: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    mul_216: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_147, clamp_max_14);  clamp_max_14 = None
    div_20: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_216, 6);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_49: "f32[512, 256]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    view_148: "f32[392, 512]" = torch.ops.aten.reshape.default(div_20, [392, 512]);  div_20 = None
    mm_24: "f32[392, 256]" = torch.ops.aten.mm.default(view_148, permute_49)
    view_149: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_24, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_150: "f32[392, 256]" = torch.ops.aten.reshape.default(view_149, [392, 256]);  view_149 = None
    add_171: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(view_150, [0], correction = 0, keepdim = True)
    getitem_73: "f32[1, 256]" = var_mean_28[0]
    getitem_74: "f32[1, 256]" = var_mean_28[1];  var_mean_28 = None
    add_172: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_73, 1e-05)
    rsqrt_28: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_34: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_150, getitem_74);  view_150 = None
    mul_217: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_28);  sub_34 = None
    squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_74, [0]);  getitem_74 = None
    squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0]);  rsqrt_28 = None
    mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_173: "f32[256]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_73, [0]);  getitem_73 = None
    mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0025575447570332);  squeeze_86 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_174: "f32[256]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    mul_223: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_217, primals_100);  mul_217 = None
    add_175: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_223, primals_101);  mul_223 = primals_101 = None
    view_151: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_175, [8, 49, 256]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_176: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_164, view_151);  add_164 = view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_50: "f32[256, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    view_152: "f32[392, 256]" = torch.ops.aten.reshape.default(add_176, [392, 256])
    mm_25: "f32[392, 512]" = torch.ops.aten.mm.default(view_152, permute_50)
    view_153: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_25, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_154: "f32[392, 512]" = torch.ops.aten.reshape.default(view_153, [392, 512]);  view_153 = None
    add_177: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(view_154, [0], correction = 0, keepdim = True)
    getitem_75: "f32[1, 512]" = var_mean_29[0]
    getitem_76: "f32[1, 512]" = var_mean_29[1];  var_mean_29 = None
    add_178: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05)
    rsqrt_29: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_35: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_154, getitem_76);  view_154 = None
    mul_224: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_29);  sub_35 = None
    squeeze_87: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_76, [0]);  getitem_76 = None
    squeeze_88: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0]);  rsqrt_29 = None
    mul_225: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_226: "f32[512]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_179: "f32[512]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_89: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_75, [0]);  getitem_75 = None
    mul_227: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0025575447570332);  squeeze_89 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_180: "f32[512]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    mul_230: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_224, primals_103);  mul_224 = None
    add_181: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_230, primals_104);  mul_230 = primals_104 = None
    view_155: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_181, [8, 49, 512]);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_156: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_155, [8, 49, 8, -1]);  view_155 = None
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_156, [16, 16, 32], 3);  view_156 = None
    getitem_77: "f32[8, 49, 8, 16]" = split_with_sizes_6[0]
    getitem_78: "f32[8, 49, 8, 16]" = split_with_sizes_6[1]
    getitem_79: "f32[8, 49, 8, 32]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_51: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_77, [0, 2, 1, 3]);  getitem_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_52: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 3, 1]);  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_53: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_24: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_51, [8, 8, 49, 16]);  permute_51 = None
    clone_40: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_157: "f32[64, 49, 16]" = torch.ops.aten.reshape.default(clone_40, [64, 49, 16]);  clone_40 = None
    expand_25: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_52, [8, 8, 16, 49]);  permute_52 = None
    clone_41: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_158: "f32[64, 16, 49]" = torch.ops.aten.reshape.default(clone_41, [64, 16, 49]);  clone_41 = None
    bmm_12: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_157, view_158)
    view_159: "f32[8, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 49, 49]);  bmm_12 = None
    mul_231: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_159, 0.25);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_6: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(primals_7, [None, primals_215]);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_182: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_231, index_6);  mul_231 = index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_182, [-1], True)
    sub_36: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_182, amax_6);  add_182 = amax_6 = None
    exp_6: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_7: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_21: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_26: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_21, [8, 8, 49, 49]);  div_21 = None
    view_160: "f32[64, 49, 49]" = torch.ops.aten.reshape.default(expand_26, [64, 49, 49]);  expand_26 = None
    expand_27: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_53, [8, 8, 49, 32]);  permute_53 = None
    clone_42: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_161: "f32[64, 49, 32]" = torch.ops.aten.reshape.default(clone_42, [64, 49, 32]);  clone_42 = None
    bmm_13: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_160, view_161)
    view_162: "f32[8, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_13, [8, 8, 49, 32]);  bmm_13 = None
    permute_54: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_43: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    view_163: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(clone_43, [8, 49, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_183: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_163, 3)
    clamp_min_15: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_183, 0);  add_183 = None
    clamp_max_15: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_232: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_163, clamp_max_15);  clamp_max_15 = None
    div_22: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_232, 6);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_55: "f32[256, 256]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_164: "f32[392, 256]" = torch.ops.aten.reshape.default(div_22, [392, 256]);  div_22 = None
    mm_26: "f32[392, 256]" = torch.ops.aten.mm.default(view_164, permute_55)
    view_165: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_26, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_166: "f32[392, 256]" = torch.ops.aten.reshape.default(view_165, [392, 256]);  view_165 = None
    add_184: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(view_166, [0], correction = 0, keepdim = True)
    getitem_80: "f32[1, 256]" = var_mean_30[0]
    getitem_81: "f32[1, 256]" = var_mean_30[1];  var_mean_30 = None
    add_185: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_30: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_37: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_166, getitem_81);  view_166 = None
    mul_233: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_30);  sub_37 = None
    squeeze_90: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_81, [0]);  getitem_81 = None
    squeeze_91: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0]);  rsqrt_30 = None
    mul_234: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_235: "f32[256]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_186: "f32[256]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_92: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_80, [0]);  getitem_80 = None
    mul_236: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0025575447570332);  squeeze_92 = None
    mul_237: "f32[256]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[256]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_187: "f32[256]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    mul_239: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_233, primals_106);  mul_233 = None
    add_188: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_239, primals_107);  mul_239 = primals_107 = None
    view_167: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_188, [8, 49, 256]);  add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_189: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_176, view_167);  add_176 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_56: "f32[256, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    view_168: "f32[392, 256]" = torch.ops.aten.reshape.default(add_189, [392, 256])
    mm_27: "f32[392, 512]" = torch.ops.aten.mm.default(view_168, permute_56)
    view_169: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_27, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_170: "f32[392, 512]" = torch.ops.aten.reshape.default(view_169, [392, 512]);  view_169 = None
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(view_170, [0], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512]" = var_mean_31[0]
    getitem_83: "f32[1, 512]" = var_mean_31[1];  var_mean_31 = None
    add_191: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_31: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_38: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_170, getitem_83);  view_170 = None
    mul_240: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_31);  sub_38 = None
    squeeze_93: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_83, [0]);  getitem_83 = None
    squeeze_94: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0]);  rsqrt_31 = None
    mul_241: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_242: "f32[512]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_192: "f32[512]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_95: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_82, [0]);  getitem_82 = None
    mul_243: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0025575447570332);  squeeze_95 = None
    mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[512]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_193: "f32[512]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    mul_246: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_240, primals_109);  mul_240 = None
    add_194: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_246, primals_110);  mul_246 = primals_110 = None
    view_171: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_194, [8, 49, 512]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_195: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_171, 3)
    clamp_min_16: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_195, 0);  add_195 = None
    clamp_max_16: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_247: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_171, clamp_max_16);  clamp_max_16 = None
    div_23: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_247, 6);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_57: "f32[512, 256]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    view_172: "f32[392, 512]" = torch.ops.aten.reshape.default(div_23, [392, 512]);  div_23 = None
    mm_28: "f32[392, 256]" = torch.ops.aten.mm.default(view_172, permute_57)
    view_173: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_28, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_174: "f32[392, 256]" = torch.ops.aten.reshape.default(view_173, [392, 256]);  view_173 = None
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(view_174, [0], correction = 0, keepdim = True)
    getitem_84: "f32[1, 256]" = var_mean_32[0]
    getitem_85: "f32[1, 256]" = var_mean_32[1];  var_mean_32 = None
    add_197: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_32: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_39: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_174, getitem_85);  view_174 = None
    mul_248: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_32);  sub_39 = None
    squeeze_96: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_85, [0]);  getitem_85 = None
    squeeze_97: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0]);  rsqrt_32 = None
    mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_250: "f32[256]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_198: "f32[256]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_98: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_84, [0]);  getitem_84 = None
    mul_251: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0025575447570332);  squeeze_98 = None
    mul_252: "f32[256]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[256]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_199: "f32[256]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    mul_254: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_248, primals_112);  mul_248 = None
    add_200: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_254, primals_113);  mul_254 = primals_113 = None
    view_175: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_200, [8, 49, 256]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_201: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_189, view_175);  add_189 = view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_58: "f32[256, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    view_176: "f32[392, 256]" = torch.ops.aten.reshape.default(add_201, [392, 256])
    mm_29: "f32[392, 512]" = torch.ops.aten.mm.default(view_176, permute_58)
    view_177: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_29, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_178: "f32[392, 512]" = torch.ops.aten.reshape.default(view_177, [392, 512]);  view_177 = None
    add_202: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(view_178, [0], correction = 0, keepdim = True)
    getitem_86: "f32[1, 512]" = var_mean_33[0]
    getitem_87: "f32[1, 512]" = var_mean_33[1];  var_mean_33 = None
    add_203: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_33: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    sub_40: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_178, getitem_87);  view_178 = None
    mul_255: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_33);  sub_40 = None
    squeeze_99: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_87, [0]);  getitem_87 = None
    squeeze_100: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0]);  rsqrt_33 = None
    mul_256: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_257: "f32[512]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_204: "f32[512]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    squeeze_101: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_86, [0]);  getitem_86 = None
    mul_258: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0025575447570332);  squeeze_101 = None
    mul_259: "f32[512]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
    mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_205: "f32[512]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    mul_261: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_255, primals_115);  mul_255 = None
    add_206: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_261, primals_116);  mul_261 = primals_116 = None
    view_179: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_206, [8, 49, 512]);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_180: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_179, [8, 49, 8, -1]);  view_179 = None
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_180, [16, 16, 32], 3);  view_180 = None
    getitem_88: "f32[8, 49, 8, 16]" = split_with_sizes_7[0]
    getitem_89: "f32[8, 49, 8, 16]" = split_with_sizes_7[1]
    getitem_90: "f32[8, 49, 8, 32]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_59: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_88, [0, 2, 1, 3]);  getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_60: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 3, 1]);  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_61: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_28: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_59, [8, 8, 49, 16]);  permute_59 = None
    clone_45: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_181: "f32[64, 49, 16]" = torch.ops.aten.reshape.default(clone_45, [64, 49, 16]);  clone_45 = None
    expand_29: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_60, [8, 8, 16, 49]);  permute_60 = None
    clone_46: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_182: "f32[64, 16, 49]" = torch.ops.aten.reshape.default(clone_46, [64, 16, 49]);  clone_46 = None
    bmm_14: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_181, view_182)
    view_183: "f32[8, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_14, [8, 8, 49, 49]);  bmm_14 = None
    mul_262: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_183, 0.25);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_7: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(primals_8, [None, primals_216]);  primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_207: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_262, index_7);  mul_262 = index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_207, [-1], True)
    sub_41: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_207, amax_7);  add_207 = amax_7 = None
    exp_7: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_8: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_24: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_30: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_24, [8, 8, 49, 49]);  div_24 = None
    view_184: "f32[64, 49, 49]" = torch.ops.aten.reshape.default(expand_30, [64, 49, 49]);  expand_30 = None
    expand_31: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_61, [8, 8, 49, 32]);  permute_61 = None
    clone_47: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_185: "f32[64, 49, 32]" = torch.ops.aten.reshape.default(clone_47, [64, 49, 32]);  clone_47 = None
    bmm_15: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_184, view_185)
    view_186: "f32[8, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_15, [8, 8, 49, 32]);  bmm_15 = None
    permute_62: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_48: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_187: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(clone_48, [8, 49, 256]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_208: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_187, 3)
    clamp_min_17: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_208, 0);  add_208 = None
    clamp_max_17: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_263: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_187, clamp_max_17);  clamp_max_17 = None
    div_25: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_263, 6);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    view_188: "f32[392, 256]" = torch.ops.aten.reshape.default(div_25, [392, 256]);  div_25 = None
    mm_30: "f32[392, 256]" = torch.ops.aten.mm.default(view_188, permute_63)
    view_189: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_30, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_190: "f32[392, 256]" = torch.ops.aten.reshape.default(view_189, [392, 256]);  view_189 = None
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(view_190, [0], correction = 0, keepdim = True)
    getitem_91: "f32[1, 256]" = var_mean_34[0]
    getitem_92: "f32[1, 256]" = var_mean_34[1];  var_mean_34 = None
    add_210: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_91, 1e-05)
    rsqrt_34: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_42: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_190, getitem_92);  view_190 = None
    mul_264: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_34);  sub_42 = None
    squeeze_102: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_92, [0]);  getitem_92 = None
    squeeze_103: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0]);  rsqrt_34 = None
    mul_265: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_266: "f32[256]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_211: "f32[256]" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    squeeze_104: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_91, [0]);  getitem_91 = None
    mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0025575447570332);  squeeze_104 = None
    mul_268: "f32[256]" = torch.ops.aten.mul.Tensor(mul_267, 0.1);  mul_267 = None
    mul_269: "f32[256]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_212: "f32[256]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    mul_270: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_264, primals_118);  mul_264 = None
    add_213: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_270, primals_119);  mul_270 = primals_119 = None
    view_191: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_213, [8, 49, 256]);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_214: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_201, view_191);  add_201 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_64: "f32[256, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    view_192: "f32[392, 256]" = torch.ops.aten.reshape.default(add_214, [392, 256])
    mm_31: "f32[392, 512]" = torch.ops.aten.mm.default(view_192, permute_64)
    view_193: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_31, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_194: "f32[392, 512]" = torch.ops.aten.reshape.default(view_193, [392, 512]);  view_193 = None
    add_215: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(view_194, [0], correction = 0, keepdim = True)
    getitem_93: "f32[1, 512]" = var_mean_35[0]
    getitem_94: "f32[1, 512]" = var_mean_35[1];  var_mean_35 = None
    add_216: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_93, 1e-05)
    rsqrt_35: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_43: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_194, getitem_94);  view_194 = None
    mul_271: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_35);  sub_43 = None
    squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_94, [0]);  getitem_94 = None
    squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0]);  rsqrt_35 = None
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_273: "f32[512]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_217: "f32[512]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_93, [0]);  getitem_93 = None
    mul_274: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0025575447570332);  squeeze_107 = None
    mul_275: "f32[512]" = torch.ops.aten.mul.Tensor(mul_274, 0.1);  mul_274 = None
    mul_276: "f32[512]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_218: "f32[512]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    mul_277: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_271, primals_121);  mul_271 = None
    add_219: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_277, primals_122);  mul_277 = primals_122 = None
    view_195: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_219, [8, 49, 512]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_220: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_195, 3)
    clamp_min_18: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_220, 0);  add_220 = None
    clamp_max_18: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    mul_278: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_195, clamp_max_18);  clamp_max_18 = None
    div_26: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_278, 6);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_65: "f32[512, 256]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    view_196: "f32[392, 512]" = torch.ops.aten.reshape.default(div_26, [392, 512]);  div_26 = None
    mm_32: "f32[392, 256]" = torch.ops.aten.mm.default(view_196, permute_65)
    view_197: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_32, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_198: "f32[392, 256]" = torch.ops.aten.reshape.default(view_197, [392, 256]);  view_197 = None
    add_221: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(view_198, [0], correction = 0, keepdim = True)
    getitem_95: "f32[1, 256]" = var_mean_36[0]
    getitem_96: "f32[1, 256]" = var_mean_36[1];  var_mean_36 = None
    add_222: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_95, 1e-05)
    rsqrt_36: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    sub_44: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_198, getitem_96);  view_198 = None
    mul_279: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_36);  sub_44 = None
    squeeze_108: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_96, [0]);  getitem_96 = None
    squeeze_109: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0]);  rsqrt_36 = None
    mul_280: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_281: "f32[256]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_223: "f32[256]" = torch.ops.aten.add.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
    squeeze_110: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_95, [0]);  getitem_95 = None
    mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0025575447570332);  squeeze_110 = None
    mul_283: "f32[256]" = torch.ops.aten.mul.Tensor(mul_282, 0.1);  mul_282 = None
    mul_284: "f32[256]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_224: "f32[256]" = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    mul_285: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_279, primals_124);  mul_279 = None
    add_225: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_285, primals_125);  mul_285 = primals_125 = None
    view_199: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_225, [8, 49, 256]);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_226: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_214, view_199);  add_214 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_66: "f32[256, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_200: "f32[392, 256]" = torch.ops.aten.reshape.default(add_226, [392, 256])
    mm_33: "f32[392, 512]" = torch.ops.aten.mm.default(view_200, permute_66)
    view_201: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_33, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_202: "f32[392, 512]" = torch.ops.aten.reshape.default(view_201, [392, 512]);  view_201 = None
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(view_202, [0], correction = 0, keepdim = True)
    getitem_97: "f32[1, 512]" = var_mean_37[0]
    getitem_98: "f32[1, 512]" = var_mean_37[1];  var_mean_37 = None
    add_228: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05)
    rsqrt_37: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_45: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_202, getitem_98);  view_202 = None
    mul_286: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_37);  sub_45 = None
    squeeze_111: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_98, [0]);  getitem_98 = None
    squeeze_112: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0]);  rsqrt_37 = None
    mul_287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_229: "f32[512]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    squeeze_113: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_97, [0]);  getitem_97 = None
    mul_289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0025575447570332);  squeeze_113 = None
    mul_290: "f32[512]" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
    mul_291: "f32[512]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_230: "f32[512]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    mul_292: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_286, primals_127);  mul_286 = None
    add_231: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_292, primals_128);  mul_292 = primals_128 = None
    view_203: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_231, [8, 49, 512]);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_204: "f32[8, 49, 8, 64]" = torch.ops.aten.reshape.default(view_203, [8, 49, 8, -1]);  view_203 = None
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_204, [16, 16, 32], 3);  view_204 = None
    getitem_99: "f32[8, 49, 8, 16]" = split_with_sizes_8[0]
    getitem_100: "f32[8, 49, 8, 16]" = split_with_sizes_8[1]
    getitem_101: "f32[8, 49, 8, 32]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_67: "f32[8, 8, 49, 16]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_68: "f32[8, 8, 16, 49]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 3, 1]);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_69: "f32[8, 8, 49, 32]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_32: "f32[8, 8, 49, 16]" = torch.ops.aten.expand.default(permute_67, [8, 8, 49, 16]);  permute_67 = None
    clone_50: "f32[8, 8, 49, 16]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_205: "f32[64, 49, 16]" = torch.ops.aten.reshape.default(clone_50, [64, 49, 16]);  clone_50 = None
    expand_33: "f32[8, 8, 16, 49]" = torch.ops.aten.expand.default(permute_68, [8, 8, 16, 49]);  permute_68 = None
    clone_51: "f32[8, 8, 16, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_206: "f32[64, 16, 49]" = torch.ops.aten.reshape.default(clone_51, [64, 16, 49]);  clone_51 = None
    bmm_16: "f32[64, 49, 49]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[8, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_16, [8, 8, 49, 49]);  bmm_16 = None
    mul_293: "f32[8, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_207, 0.25);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_8: "f32[8, 49, 49]" = torch.ops.aten.index.Tensor(primals_9, [None, primals_217]);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_232: "f32[8, 8, 49, 49]" = torch.ops.aten.add.Tensor(mul_293, index_8);  mul_293 = index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 8, 49, 1]" = torch.ops.aten.amax.default(add_232, [-1], True)
    sub_46: "f32[8, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_232, amax_8);  add_232 = amax_8 = None
    exp_8: "f32[8, 8, 49, 49]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_9: "f32[8, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_27: "f32[8, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_34: "f32[8, 8, 49, 49]" = torch.ops.aten.expand.default(div_27, [8, 8, 49, 49]);  div_27 = None
    view_208: "f32[64, 49, 49]" = torch.ops.aten.reshape.default(expand_34, [64, 49, 49]);  expand_34 = None
    expand_35: "f32[8, 8, 49, 32]" = torch.ops.aten.expand.default(permute_69, [8, 8, 49, 32]);  permute_69 = None
    clone_52: "f32[8, 8, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_209: "f32[64, 49, 32]" = torch.ops.aten.reshape.default(clone_52, [64, 49, 32]);  clone_52 = None
    bmm_17: "f32[64, 49, 32]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[8, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_17, [8, 8, 49, 32]);  bmm_17 = None
    permute_70: "f32[8, 49, 8, 32]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_53: "f32[8, 49, 8, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    view_211: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(clone_53, [8, 49, 256]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_233: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(view_211, 3)
    clamp_min_19: "f32[8, 49, 256]" = torch.ops.aten.clamp_min.default(add_233, 0);  add_233 = None
    clamp_max_19: "f32[8, 49, 256]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_294: "f32[8, 49, 256]" = torch.ops.aten.mul.Tensor(view_211, clamp_max_19);  clamp_max_19 = None
    div_28: "f32[8, 49, 256]" = torch.ops.aten.div.Tensor(mul_294, 6);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_71: "f32[256, 256]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    view_212: "f32[392, 256]" = torch.ops.aten.reshape.default(div_28, [392, 256]);  div_28 = None
    mm_34: "f32[392, 256]" = torch.ops.aten.mm.default(view_212, permute_71)
    view_213: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_34, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_214: "f32[392, 256]" = torch.ops.aten.reshape.default(view_213, [392, 256]);  view_213 = None
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(view_214, [0], correction = 0, keepdim = True)
    getitem_102: "f32[1, 256]" = var_mean_38[0]
    getitem_103: "f32[1, 256]" = var_mean_38[1];  var_mean_38 = None
    add_235: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_38: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_47: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_214, getitem_103);  view_214 = None
    mul_295: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_38);  sub_47 = None
    squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_103, [0]);  getitem_103 = None
    squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0]);  rsqrt_38 = None
    mul_296: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_297: "f32[256]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_236: "f32[256]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_102, [0]);  getitem_102 = None
    mul_298: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0025575447570332);  squeeze_116 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(mul_298, 0.1);  mul_298 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_237: "f32[256]" = torch.ops.aten.add.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    mul_301: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_295, primals_130);  mul_295 = None
    add_238: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_301, primals_131);  mul_301 = primals_131 = None
    view_215: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_238, [8, 49, 256]);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_239: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_226, view_215);  add_226 = view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_72: "f32[256, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    view_216: "f32[392, 256]" = torch.ops.aten.reshape.default(add_239, [392, 256])
    mm_35: "f32[392, 512]" = torch.ops.aten.mm.default(view_216, permute_72)
    view_217: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(mm_35, [8, 49, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_218: "f32[392, 512]" = torch.ops.aten.reshape.default(view_217, [392, 512]);  view_217 = None
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(view_218, [0], correction = 0, keepdim = True)
    getitem_104: "f32[1, 512]" = var_mean_39[0]
    getitem_105: "f32[1, 512]" = var_mean_39[1];  var_mean_39 = None
    add_241: "f32[1, 512]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_39: "f32[1, 512]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_48: "f32[392, 512]" = torch.ops.aten.sub.Tensor(view_218, getitem_105);  view_218 = None
    mul_302: "f32[392, 512]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_39);  sub_48 = None
    squeeze_117: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_105, [0]);  getitem_105 = None
    squeeze_118: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0]);  rsqrt_39 = None
    mul_303: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_242: "f32[512]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    squeeze_119: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_104, [0]);  getitem_104 = None
    mul_305: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0025575447570332);  squeeze_119 = None
    mul_306: "f32[512]" = torch.ops.aten.mul.Tensor(mul_305, 0.1);  mul_305 = None
    mul_307: "f32[512]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_243: "f32[512]" = torch.ops.aten.add.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    mul_308: "f32[392, 512]" = torch.ops.aten.mul.Tensor(mul_302, primals_133);  mul_302 = None
    add_244: "f32[392, 512]" = torch.ops.aten.add.Tensor(mul_308, primals_134);  mul_308 = primals_134 = None
    view_219: "f32[8, 49, 512]" = torch.ops.aten.reshape.default(add_244, [8, 49, 512]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_245: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_219, 3)
    clamp_min_20: "f32[8, 49, 512]" = torch.ops.aten.clamp_min.default(add_245, 0);  add_245 = None
    clamp_max_20: "f32[8, 49, 512]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_309: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_219, clamp_max_20);  clamp_max_20 = None
    div_29: "f32[8, 49, 512]" = torch.ops.aten.div.Tensor(mul_309, 6);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_73: "f32[512, 256]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_220: "f32[392, 512]" = torch.ops.aten.reshape.default(div_29, [392, 512]);  div_29 = None
    mm_36: "f32[392, 256]" = torch.ops.aten.mm.default(view_220, permute_73)
    view_221: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(mm_36, [8, 49, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_222: "f32[392, 256]" = torch.ops.aten.reshape.default(view_221, [392, 256]);  view_221 = None
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(view_222, [0], correction = 0, keepdim = True)
    getitem_106: "f32[1, 256]" = var_mean_40[0]
    getitem_107: "f32[1, 256]" = var_mean_40[1];  var_mean_40 = None
    add_247: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_40: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_49: "f32[392, 256]" = torch.ops.aten.sub.Tensor(view_222, getitem_107);  view_222 = None
    mul_310: "f32[392, 256]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_40);  sub_49 = None
    squeeze_120: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_107, [0]);  getitem_107 = None
    squeeze_121: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0]);  rsqrt_40 = None
    mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_248: "f32[256]" = torch.ops.aten.add.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    squeeze_122: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_106, [0]);  getitem_106 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0025575447570332);  squeeze_122 = None
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(mul_313, 0.1);  mul_313 = None
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_249: "f32[256]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    mul_316: "f32[392, 256]" = torch.ops.aten.mul.Tensor(mul_310, primals_136);  mul_310 = None
    add_250: "f32[392, 256]" = torch.ops.aten.add.Tensor(mul_316, primals_137);  mul_316 = primals_137 = None
    view_223: "f32[8, 49, 256]" = torch.ops.aten.reshape.default(add_250, [8, 49, 256]);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_251: "f32[8, 49, 256]" = torch.ops.aten.add.Tensor(add_239, view_223);  add_239 = view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_74: "f32[256, 1280]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    view_224: "f32[392, 256]" = torch.ops.aten.reshape.default(add_251, [392, 256])
    mm_37: "f32[392, 1280]" = torch.ops.aten.mm.default(view_224, permute_74)
    view_225: "f32[8, 49, 1280]" = torch.ops.aten.reshape.default(mm_37, [8, 49, 1280])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_226: "f32[392, 1280]" = torch.ops.aten.reshape.default(view_225, [392, 1280]);  view_225 = None
    add_252: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(view_226, [0], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1280]" = var_mean_41[0]
    getitem_109: "f32[1, 1280]" = var_mean_41[1];  var_mean_41 = None
    add_253: "f32[1, 1280]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_41: "f32[1, 1280]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_50: "f32[392, 1280]" = torch.ops.aten.sub.Tensor(view_226, getitem_109);  view_226 = None
    mul_317: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_41);  sub_50 = None
    squeeze_123: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_109, [0]);  getitem_109 = None
    squeeze_124: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0]);  rsqrt_41 = None
    mul_318: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_319: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_254: "f32[1280]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    squeeze_125: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_108, [0]);  getitem_108 = None
    mul_320: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0025575447570332);  squeeze_125 = None
    mul_321: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_320, 0.1);  mul_320 = None
    mul_322: "f32[1280]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_255: "f32[1280]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    mul_323: "f32[392, 1280]" = torch.ops.aten.mul.Tensor(mul_317, primals_139);  mul_317 = None
    add_256: "f32[392, 1280]" = torch.ops.aten.add.Tensor(mul_323, primals_140);  mul_323 = primals_140 = None
    view_227: "f32[8, 49, 1280]" = torch.ops.aten.reshape.default(add_256, [8, 49, 1280]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_228: "f32[8, 49, 16, 80]" = torch.ops.aten.reshape.default(view_227, [8, 49, 16, -1]);  view_227 = None
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_228, [16, 64], 3);  view_228 = None
    getitem_110: "f32[8, 49, 16, 16]" = split_with_sizes_9[0]
    getitem_111: "f32[8, 49, 16, 64]" = split_with_sizes_9[1];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    permute_75: "f32[8, 16, 16, 49]" = torch.ops.aten.permute.default(getitem_110, [0, 2, 3, 1]);  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    permute_76: "f32[8, 16, 49, 64]" = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    view_229: "f32[8, 7, 7, 256]" = torch.ops.aten.reshape.default(add_251, [8, 7, 7, 256]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    slice_13: "f32[8, 7, 7, 256]" = torch.ops.aten.slice.Tensor(view_229, 0, 0, 9223372036854775807);  view_229 = None
    slice_14: "f32[8, 4, 7, 256]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807, 2);  slice_13 = None
    slice_15: "f32[8, 4, 4, 256]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807, 2);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    clone_55: "f32[8, 4, 4, 256]" = torch.ops.aten.clone.default(slice_15, memory_format = torch.contiguous_format);  slice_15 = None
    view_230: "f32[8, 16, 256]" = torch.ops.aten.reshape.default(clone_55, [8, 16, 256]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_77: "f32[256, 256]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    view_231: "f32[128, 256]" = torch.ops.aten.reshape.default(view_230, [128, 256]);  view_230 = None
    mm_38: "f32[128, 256]" = torch.ops.aten.mm.default(view_231, permute_77)
    view_232: "f32[8, 16, 256]" = torch.ops.aten.reshape.default(mm_38, [8, 16, 256])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_233: "f32[128, 256]" = torch.ops.aten.reshape.default(view_232, [128, 256]);  view_232 = None
    add_257: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(view_233, [0], correction = 0, keepdim = True)
    getitem_112: "f32[1, 256]" = var_mean_42[0]
    getitem_113: "f32[1, 256]" = var_mean_42[1];  var_mean_42 = None
    add_258: "f32[1, 256]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_42: "f32[1, 256]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_51: "f32[128, 256]" = torch.ops.aten.sub.Tensor(view_233, getitem_113);  view_233 = None
    mul_324: "f32[128, 256]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_42);  sub_51 = None
    squeeze_126: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_113, [0]);  getitem_113 = None
    squeeze_127: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0]);  rsqrt_42 = None
    mul_325: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_326: "f32[256]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_259: "f32[256]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    squeeze_128: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_112, [0]);  getitem_112 = None
    mul_327: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0078740157480315);  squeeze_128 = None
    mul_328: "f32[256]" = torch.ops.aten.mul.Tensor(mul_327, 0.1);  mul_327 = None
    mul_329: "f32[256]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_260: "f32[256]" = torch.ops.aten.add.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
    mul_330: "f32[128, 256]" = torch.ops.aten.mul.Tensor(mul_324, primals_142);  mul_324 = None
    add_261: "f32[128, 256]" = torch.ops.aten.add.Tensor(mul_330, primals_143);  mul_330 = primals_143 = None
    view_234: "f32[8, 16, 256]" = torch.ops.aten.reshape.default(add_261, [8, 16, 256]);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_235: "f32[8, 16, 16, 16]" = torch.ops.aten.reshape.default(view_234, [8, -1, 16, 16]);  view_234 = None
    permute_78: "f32[8, 16, 16, 16]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_36: "f32[8, 16, 16, 16]" = torch.ops.aten.expand.default(permute_78, [8, 16, 16, 16]);  permute_78 = None
    clone_56: "f32[8, 16, 16, 16]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_236: "f32[128, 16, 16]" = torch.ops.aten.reshape.default(clone_56, [128, 16, 16]);  clone_56 = None
    expand_37: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(permute_75, [8, 16, 16, 49]);  permute_75 = None
    clone_57: "f32[8, 16, 16, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_237: "f32[128, 16, 49]" = torch.ops.aten.reshape.default(clone_57, [128, 16, 49]);  clone_57 = None
    bmm_18: "f32[128, 16, 49]" = torch.ops.aten.bmm.default(view_236, view_237)
    view_238: "f32[8, 16, 16, 49]" = torch.ops.aten.reshape.default(bmm_18, [8, 16, 16, 49]);  bmm_18 = None
    mul_331: "f32[8, 16, 16, 49]" = torch.ops.aten.mul.Tensor(view_238, 0.25);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:311, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_9: "f32[16, 16, 49]" = torch.ops.aten.index.Tensor(primals_10, [None, primals_218]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_262: "f32[8, 16, 16, 49]" = torch.ops.aten.add.Tensor(mul_331, index_9);  mul_331 = index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 16, 16, 1]" = torch.ops.aten.amax.default(add_262, [-1], True)
    sub_52: "f32[8, 16, 16, 49]" = torch.ops.aten.sub.Tensor(add_262, amax_9);  add_262 = amax_9 = None
    exp_9: "f32[8, 16, 16, 49]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_10: "f32[8, 16, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_30: "f32[8, 16, 16, 49]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 16, 16, 49]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    expand_38: "f32[8, 16, 16, 49]" = torch.ops.aten.expand.default(div_30, [8, 16, 16, 49]);  div_30 = None
    view_239: "f32[128, 16, 49]" = torch.ops.aten.reshape.default(expand_38, [128, 16, 49]);  expand_38 = None
    expand_39: "f32[8, 16, 49, 64]" = torch.ops.aten.expand.default(permute_76, [8, 16, 49, 64]);  permute_76 = None
    clone_58: "f32[8, 16, 49, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_240: "f32[128, 49, 64]" = torch.ops.aten.reshape.default(clone_58, [128, 49, 64]);  clone_58 = None
    bmm_19: "f32[128, 16, 64]" = torch.ops.aten.bmm.default(view_239, view_240)
    view_241: "f32[8, 16, 16, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 16, 16, 64]);  bmm_19 = None
    permute_79: "f32[8, 16, 16, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_59: "f32[8, 16, 16, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_242: "f32[8, 16, 1024]" = torch.ops.aten.reshape.default(clone_59, [8, 16, 1024]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    add_263: "f32[8, 16, 1024]" = torch.ops.aten.add.Tensor(view_242, 3)
    clamp_min_21: "f32[8, 16, 1024]" = torch.ops.aten.clamp_min.default(add_263, 0);  add_263 = None
    clamp_max_21: "f32[8, 16, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_332: "f32[8, 16, 1024]" = torch.ops.aten.mul.Tensor(view_242, clamp_max_21);  clamp_max_21 = None
    div_31: "f32[8, 16, 1024]" = torch.ops.aten.div.Tensor(mul_332, 6);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_80: "f32[1024, 384]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    view_243: "f32[128, 1024]" = torch.ops.aten.reshape.default(div_31, [128, 1024]);  div_31 = None
    mm_39: "f32[128, 384]" = torch.ops.aten.mm.default(view_243, permute_80)
    view_244: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_39, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_245: "f32[128, 384]" = torch.ops.aten.reshape.default(view_244, [128, 384]);  view_244 = None
    add_264: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(view_245, [0], correction = 0, keepdim = True)
    getitem_114: "f32[1, 384]" = var_mean_43[0]
    getitem_115: "f32[1, 384]" = var_mean_43[1];  var_mean_43 = None
    add_265: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_43: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
    sub_53: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_245, getitem_115);  view_245 = None
    mul_333: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_43);  sub_53 = None
    squeeze_129: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_115, [0]);  getitem_115 = None
    squeeze_130: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0]);  rsqrt_43 = None
    mul_334: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_335: "f32[384]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_266: "f32[384]" = torch.ops.aten.add.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    squeeze_131: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_114, [0]);  getitem_114 = None
    mul_336: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0078740157480315);  squeeze_131 = None
    mul_337: "f32[384]" = torch.ops.aten.mul.Tensor(mul_336, 0.1);  mul_336 = None
    mul_338: "f32[384]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_267: "f32[384]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    mul_339: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_333, primals_145);  mul_333 = None
    add_268: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_339, primals_146);  mul_339 = primals_146 = None
    view_246: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_268, [8, 16, 384]);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_81: "f32[384, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_247: "f32[128, 384]" = torch.ops.aten.reshape.default(view_246, [128, 384])
    mm_40: "f32[128, 768]" = torch.ops.aten.mm.default(view_247, permute_81)
    view_248: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_40, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_249: "f32[128, 768]" = torch.ops.aten.reshape.default(view_248, [128, 768]);  view_248 = None
    add_269: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(view_249, [0], correction = 0, keepdim = True)
    getitem_116: "f32[1, 768]" = var_mean_44[0]
    getitem_117: "f32[1, 768]" = var_mean_44[1];  var_mean_44 = None
    add_270: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_44: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_270);  add_270 = None
    sub_54: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_249, getitem_117);  view_249 = None
    mul_340: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_44);  sub_54 = None
    squeeze_132: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_117, [0]);  getitem_117 = None
    squeeze_133: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0]);  rsqrt_44 = None
    mul_341: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_342: "f32[768]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_271: "f32[768]" = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
    squeeze_134: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_116, [0]);  getitem_116 = None
    mul_343: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0078740157480315);  squeeze_134 = None
    mul_344: "f32[768]" = torch.ops.aten.mul.Tensor(mul_343, 0.1);  mul_343 = None
    mul_345: "f32[768]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_272: "f32[768]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    mul_346: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_340, primals_148);  mul_340 = None
    add_273: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_346, primals_149);  mul_346 = primals_149 = None
    view_250: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_273, [8, 16, 768]);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_274: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_250, 3)
    clamp_min_22: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_274, 0);  add_274 = None
    clamp_max_22: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    mul_347: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_250, clamp_max_22);  clamp_max_22 = None
    div_32: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_347, 6);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_82: "f32[768, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    view_251: "f32[128, 768]" = torch.ops.aten.reshape.default(div_32, [128, 768]);  div_32 = None
    mm_41: "f32[128, 384]" = torch.ops.aten.mm.default(view_251, permute_82)
    view_252: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_41, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_253: "f32[128, 384]" = torch.ops.aten.reshape.default(view_252, [128, 384]);  view_252 = None
    add_275: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(view_253, [0], correction = 0, keepdim = True)
    getitem_118: "f32[1, 384]" = var_mean_45[0]
    getitem_119: "f32[1, 384]" = var_mean_45[1];  var_mean_45 = None
    add_276: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_45: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_55: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_253, getitem_119);  view_253 = None
    mul_348: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_45);  sub_55 = None
    squeeze_135: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_119, [0]);  getitem_119 = None
    squeeze_136: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0]);  rsqrt_45 = None
    mul_349: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_350: "f32[384]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_277: "f32[384]" = torch.ops.aten.add.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
    squeeze_137: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_118, [0]);  getitem_118 = None
    mul_351: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0078740157480315);  squeeze_137 = None
    mul_352: "f32[384]" = torch.ops.aten.mul.Tensor(mul_351, 0.1);  mul_351 = None
    mul_353: "f32[384]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_278: "f32[384]" = torch.ops.aten.add.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
    mul_354: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_348, primals_151);  mul_348 = None
    add_279: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_354, primals_152);  mul_354 = primals_152 = None
    view_254: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_279, [8, 16, 384]);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    add_280: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_246, view_254);  view_246 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_83: "f32[384, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    view_255: "f32[128, 384]" = torch.ops.aten.reshape.default(add_280, [128, 384])
    mm_42: "f32[128, 768]" = torch.ops.aten.mm.default(view_255, permute_83)
    view_256: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_42, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_257: "f32[128, 768]" = torch.ops.aten.reshape.default(view_256, [128, 768]);  view_256 = None
    add_281: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(view_257, [0], correction = 0, keepdim = True)
    getitem_120: "f32[1, 768]" = var_mean_46[0]
    getitem_121: "f32[1, 768]" = var_mean_46[1];  var_mean_46 = None
    add_282: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_46: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
    sub_56: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_257, getitem_121);  view_257 = None
    mul_355: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_46);  sub_56 = None
    squeeze_138: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_121, [0]);  getitem_121 = None
    squeeze_139: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0]);  rsqrt_46 = None
    mul_356: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_357: "f32[768]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_283: "f32[768]" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    squeeze_140: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_120, [0]);  getitem_120 = None
    mul_358: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0078740157480315);  squeeze_140 = None
    mul_359: "f32[768]" = torch.ops.aten.mul.Tensor(mul_358, 0.1);  mul_358 = None
    mul_360: "f32[768]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_284: "f32[768]" = torch.ops.aten.add.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    mul_361: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_355, primals_154);  mul_355 = None
    add_285: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_361, primals_155);  mul_361 = primals_155 = None
    view_258: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_285, [8, 16, 768]);  add_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_259: "f32[8, 16, 12, 64]" = torch.ops.aten.reshape.default(view_258, [8, 16, 12, -1]);  view_258 = None
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_259, [16, 16, 32], 3);  view_259 = None
    getitem_122: "f32[8, 16, 12, 16]" = split_with_sizes_10[0]
    getitem_123: "f32[8, 16, 12, 16]" = split_with_sizes_10[1]
    getitem_124: "f32[8, 16, 12, 32]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_84: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_85: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_123, [0, 2, 3, 1]);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_86: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_124, [0, 2, 1, 3]);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_40: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_84, [8, 12, 16, 16]);  permute_84 = None
    clone_61: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_260: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_61, [96, 16, 16]);  clone_61 = None
    expand_41: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_85, [8, 12, 16, 16]);  permute_85 = None
    clone_62: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_261: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_62, [96, 16, 16]);  clone_62 = None
    bmm_20: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_260, view_261)
    view_262: "f32[8, 12, 16, 16]" = torch.ops.aten.reshape.default(bmm_20, [8, 12, 16, 16]);  bmm_20 = None
    mul_362: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_262, 0.25);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_10: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(primals_11, [None, primals_219]);  primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_286: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_362, index_10);  mul_362 = index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_286, [-1], True)
    sub_57: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_286, amax_10);  add_286 = amax_10 = None
    exp_10: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_11: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_33: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_42: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_33, [8, 12, 16, 16]);  div_33 = None
    view_263: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(expand_42, [96, 16, 16]);  expand_42 = None
    expand_43: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_86, [8, 12, 16, 32]);  permute_86 = None
    clone_63: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_264: "f32[96, 16, 32]" = torch.ops.aten.reshape.default(clone_63, [96, 16, 32]);  clone_63 = None
    bmm_21: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_263, view_264)
    view_265: "f32[8, 12, 16, 32]" = torch.ops.aten.reshape.default(bmm_21, [8, 12, 16, 32]);  bmm_21 = None
    permute_87: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    clone_64: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    view_266: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(clone_64, [8, 16, 384]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_287: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_266, 3)
    clamp_min_23: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_287, 0);  add_287 = None
    clamp_max_23: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_363: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_266, clamp_max_23);  clamp_max_23 = None
    div_34: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_363, 6);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_88: "f32[384, 384]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    view_267: "f32[128, 384]" = torch.ops.aten.reshape.default(div_34, [128, 384]);  div_34 = None
    mm_43: "f32[128, 384]" = torch.ops.aten.mm.default(view_267, permute_88)
    view_268: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_43, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_269: "f32[128, 384]" = torch.ops.aten.reshape.default(view_268, [128, 384]);  view_268 = None
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(view_269, [0], correction = 0, keepdim = True)
    getitem_125: "f32[1, 384]" = var_mean_47[0]
    getitem_126: "f32[1, 384]" = var_mean_47[1];  var_mean_47 = None
    add_289: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_125, 1e-05)
    rsqrt_47: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_58: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_269, getitem_126);  view_269 = None
    mul_364: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_47);  sub_58 = None
    squeeze_141: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_126, [0]);  getitem_126 = None
    squeeze_142: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0]);  rsqrt_47 = None
    mul_365: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_366: "f32[384]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_290: "f32[384]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_143: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_125, [0]);  getitem_125 = None
    mul_367: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0078740157480315);  squeeze_143 = None
    mul_368: "f32[384]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[384]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_291: "f32[384]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    mul_370: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_364, primals_157);  mul_364 = None
    add_292: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_370, primals_158);  mul_370 = primals_158 = None
    view_270: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_292, [8, 16, 384]);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_293: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_280, view_270);  add_280 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_89: "f32[384, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    view_271: "f32[128, 384]" = torch.ops.aten.reshape.default(add_293, [128, 384])
    mm_44: "f32[128, 768]" = torch.ops.aten.mm.default(view_271, permute_89)
    view_272: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_44, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_273: "f32[128, 768]" = torch.ops.aten.reshape.default(view_272, [128, 768]);  view_272 = None
    add_294: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(view_273, [0], correction = 0, keepdim = True)
    getitem_127: "f32[1, 768]" = var_mean_48[0]
    getitem_128: "f32[1, 768]" = var_mean_48[1];  var_mean_48 = None
    add_295: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_127, 1e-05)
    rsqrt_48: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_59: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_273, getitem_128);  view_273 = None
    mul_371: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_48);  sub_59 = None
    squeeze_144: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_128, [0]);  getitem_128 = None
    squeeze_145: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0]);  rsqrt_48 = None
    mul_372: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_373: "f32[768]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_296: "f32[768]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_146: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_127, [0]);  getitem_127 = None
    mul_374: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0078740157480315);  squeeze_146 = None
    mul_375: "f32[768]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[768]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_297: "f32[768]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    mul_377: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_371, primals_160);  mul_371 = None
    add_298: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_377, primals_161);  mul_377 = primals_161 = None
    view_274: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_298, [8, 16, 768]);  add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_299: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_274, 3)
    clamp_min_24: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_299, 0);  add_299 = None
    clamp_max_24: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_378: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_274, clamp_max_24);  clamp_max_24 = None
    div_35: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_378, 6);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_90: "f32[768, 384]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    view_275: "f32[128, 768]" = torch.ops.aten.reshape.default(div_35, [128, 768]);  div_35 = None
    mm_45: "f32[128, 384]" = torch.ops.aten.mm.default(view_275, permute_90)
    view_276: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_45, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_277: "f32[128, 384]" = torch.ops.aten.reshape.default(view_276, [128, 384]);  view_276 = None
    add_300: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(view_277, [0], correction = 0, keepdim = True)
    getitem_129: "f32[1, 384]" = var_mean_49[0]
    getitem_130: "f32[1, 384]" = var_mean_49[1];  var_mean_49 = None
    add_301: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_129, 1e-05)
    rsqrt_49: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
    sub_60: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_277, getitem_130);  view_277 = None
    mul_379: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_49);  sub_60 = None
    squeeze_147: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_130, [0]);  getitem_130 = None
    squeeze_148: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0]);  rsqrt_49 = None
    mul_380: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_381: "f32[384]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_302: "f32[384]" = torch.ops.aten.add.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    squeeze_149: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_129, [0]);  getitem_129 = None
    mul_382: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0078740157480315);  squeeze_149 = None
    mul_383: "f32[384]" = torch.ops.aten.mul.Tensor(mul_382, 0.1);  mul_382 = None
    mul_384: "f32[384]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_303: "f32[384]" = torch.ops.aten.add.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    mul_385: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_379, primals_163);  mul_379 = None
    add_304: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_385, primals_164);  mul_385 = primals_164 = None
    view_278: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_304, [8, 16, 384]);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_305: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_293, view_278);  add_293 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_91: "f32[384, 768]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_279: "f32[128, 384]" = torch.ops.aten.reshape.default(add_305, [128, 384])
    mm_46: "f32[128, 768]" = torch.ops.aten.mm.default(view_279, permute_91)
    view_280: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_281: "f32[128, 768]" = torch.ops.aten.reshape.default(view_280, [128, 768]);  view_280 = None
    add_306: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(view_281, [0], correction = 0, keepdim = True)
    getitem_131: "f32[1, 768]" = var_mean_50[0]
    getitem_132: "f32[1, 768]" = var_mean_50[1];  var_mean_50 = None
    add_307: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_131, 1e-05)
    rsqrt_50: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
    sub_61: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_281, getitem_132);  view_281 = None
    mul_386: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_50);  sub_61 = None
    squeeze_150: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_132, [0]);  getitem_132 = None
    squeeze_151: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0]);  rsqrt_50 = None
    mul_387: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_388: "f32[768]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_308: "f32[768]" = torch.ops.aten.add.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    squeeze_152: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_131, [0]);  getitem_131 = None
    mul_389: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0078740157480315);  squeeze_152 = None
    mul_390: "f32[768]" = torch.ops.aten.mul.Tensor(mul_389, 0.1);  mul_389 = None
    mul_391: "f32[768]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_309: "f32[768]" = torch.ops.aten.add.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    mul_392: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_386, primals_166);  mul_386 = None
    add_310: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_392, primals_167);  mul_392 = primals_167 = None
    view_282: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_310, [8, 16, 768]);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_283: "f32[8, 16, 12, 64]" = torch.ops.aten.reshape.default(view_282, [8, 16, 12, -1]);  view_282 = None
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_283, [16, 16, 32], 3);  view_283 = None
    getitem_133: "f32[8, 16, 12, 16]" = split_with_sizes_11[0]
    getitem_134: "f32[8, 16, 12, 16]" = split_with_sizes_11[1]
    getitem_135: "f32[8, 16, 12, 32]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_92: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_93: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 3, 1]);  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_94: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_44: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_92, [8, 12, 16, 16]);  permute_92 = None
    clone_66: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_284: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_66, [96, 16, 16]);  clone_66 = None
    expand_45: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_93, [8, 12, 16, 16]);  permute_93 = None
    clone_67: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_285: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_67, [96, 16, 16]);  clone_67 = None
    bmm_22: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_284, view_285)
    view_286: "f32[8, 12, 16, 16]" = torch.ops.aten.reshape.default(bmm_22, [8, 12, 16, 16]);  bmm_22 = None
    mul_393: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_286, 0.25);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_11: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(primals_12, [None, primals_220]);  primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_311: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_393, index_11);  mul_393 = index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_311, [-1], True)
    sub_62: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_311, amax_11);  add_311 = amax_11 = None
    exp_11: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_62);  sub_62 = None
    sum_12: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_36: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_46: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_36, [8, 12, 16, 16]);  div_36 = None
    view_287: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(expand_46, [96, 16, 16]);  expand_46 = None
    expand_47: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_94, [8, 12, 16, 32]);  permute_94 = None
    clone_68: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_288: "f32[96, 16, 32]" = torch.ops.aten.reshape.default(clone_68, [96, 16, 32]);  clone_68 = None
    bmm_23: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_287, view_288)
    view_289: "f32[8, 12, 16, 32]" = torch.ops.aten.reshape.default(bmm_23, [8, 12, 16, 32]);  bmm_23 = None
    permute_95: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    clone_69: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_290: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(clone_69, [8, 16, 384]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_312: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_290, 3)
    clamp_min_25: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_312, 0);  add_312 = None
    clamp_max_25: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_394: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_290, clamp_max_25);  clamp_max_25 = None
    div_37: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_394, 6);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_96: "f32[384, 384]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    view_291: "f32[128, 384]" = torch.ops.aten.reshape.default(div_37, [128, 384]);  div_37 = None
    mm_47: "f32[128, 384]" = torch.ops.aten.mm.default(view_291, permute_96)
    view_292: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_47, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_293: "f32[128, 384]" = torch.ops.aten.reshape.default(view_292, [128, 384]);  view_292 = None
    add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(view_293, [0], correction = 0, keepdim = True)
    getitem_136: "f32[1, 384]" = var_mean_51[0]
    getitem_137: "f32[1, 384]" = var_mean_51[1];  var_mean_51 = None
    add_314: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_51: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
    sub_63: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_293, getitem_137);  view_293 = None
    mul_395: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_51);  sub_63 = None
    squeeze_153: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_137, [0]);  getitem_137 = None
    squeeze_154: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0]);  rsqrt_51 = None
    mul_396: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_397: "f32[384]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_315: "f32[384]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    squeeze_155: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_136, [0]);  getitem_136 = None
    mul_398: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0078740157480315);  squeeze_155 = None
    mul_399: "f32[384]" = torch.ops.aten.mul.Tensor(mul_398, 0.1);  mul_398 = None
    mul_400: "f32[384]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_316: "f32[384]" = torch.ops.aten.add.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    mul_401: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_395, primals_169);  mul_395 = None
    add_317: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_401, primals_170);  mul_401 = primals_170 = None
    view_294: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_317, [8, 16, 384]);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_318: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_305, view_294);  add_305 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_97: "f32[384, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    view_295: "f32[128, 384]" = torch.ops.aten.reshape.default(add_318, [128, 384])
    mm_48: "f32[128, 768]" = torch.ops.aten.mm.default(view_295, permute_97)
    view_296: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_48, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_297: "f32[128, 768]" = torch.ops.aten.reshape.default(view_296, [128, 768]);  view_296 = None
    add_319: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(view_297, [0], correction = 0, keepdim = True)
    getitem_138: "f32[1, 768]" = var_mean_52[0]
    getitem_139: "f32[1, 768]" = var_mean_52[1];  var_mean_52 = None
    add_320: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_52: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
    sub_64: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_297, getitem_139);  view_297 = None
    mul_402: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_52);  sub_64 = None
    squeeze_156: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_139, [0]);  getitem_139 = None
    squeeze_157: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0]);  rsqrt_52 = None
    mul_403: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_404: "f32[768]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_321: "f32[768]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    squeeze_158: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_138, [0]);  getitem_138 = None
    mul_405: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0078740157480315);  squeeze_158 = None
    mul_406: "f32[768]" = torch.ops.aten.mul.Tensor(mul_405, 0.1);  mul_405 = None
    mul_407: "f32[768]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_322: "f32[768]" = torch.ops.aten.add.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    mul_408: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_402, primals_172);  mul_402 = None
    add_323: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_408, primals_173);  mul_408 = primals_173 = None
    view_298: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_323, [8, 16, 768]);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_324: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_298, 3)
    clamp_min_26: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_324, 0);  add_324 = None
    clamp_max_26: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    mul_409: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_298, clamp_max_26);  clamp_max_26 = None
    div_38: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_409, 6);  mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_98: "f32[768, 384]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    view_299: "f32[128, 768]" = torch.ops.aten.reshape.default(div_38, [128, 768]);  div_38 = None
    mm_49: "f32[128, 384]" = torch.ops.aten.mm.default(view_299, permute_98)
    view_300: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_49, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_301: "f32[128, 384]" = torch.ops.aten.reshape.default(view_300, [128, 384]);  view_300 = None
    add_325: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(view_301, [0], correction = 0, keepdim = True)
    getitem_140: "f32[1, 384]" = var_mean_53[0]
    getitem_141: "f32[1, 384]" = var_mean_53[1];  var_mean_53 = None
    add_326: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_53: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
    sub_65: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_301, getitem_141);  view_301 = None
    mul_410: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_53);  sub_65 = None
    squeeze_159: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_141, [0]);  getitem_141 = None
    squeeze_160: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0]);  rsqrt_53 = None
    mul_411: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_412: "f32[384]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_327: "f32[384]" = torch.ops.aten.add.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    squeeze_161: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_140, [0]);  getitem_140 = None
    mul_413: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0078740157480315);  squeeze_161 = None
    mul_414: "f32[384]" = torch.ops.aten.mul.Tensor(mul_413, 0.1);  mul_413 = None
    mul_415: "f32[384]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_328: "f32[384]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    mul_416: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_410, primals_175);  mul_410 = None
    add_329: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_416, primals_176);  mul_416 = primals_176 = None
    view_302: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_329, [8, 16, 384]);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_330: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_318, view_302);  add_318 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_99: "f32[384, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    view_303: "f32[128, 384]" = torch.ops.aten.reshape.default(add_330, [128, 384])
    mm_50: "f32[128, 768]" = torch.ops.aten.mm.default(view_303, permute_99)
    view_304: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_50, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_305: "f32[128, 768]" = torch.ops.aten.reshape.default(view_304, [128, 768]);  view_304 = None
    add_331: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(view_305, [0], correction = 0, keepdim = True)
    getitem_142: "f32[1, 768]" = var_mean_54[0]
    getitem_143: "f32[1, 768]" = var_mean_54[1];  var_mean_54 = None
    add_332: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_54: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_66: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_305, getitem_143);  view_305 = None
    mul_417: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_54);  sub_66 = None
    squeeze_162: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_143, [0]);  getitem_143 = None
    squeeze_163: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0]);  rsqrt_54 = None
    mul_418: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_419: "f32[768]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_333: "f32[768]" = torch.ops.aten.add.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
    squeeze_164: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_142, [0]);  getitem_142 = None
    mul_420: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0078740157480315);  squeeze_164 = None
    mul_421: "f32[768]" = torch.ops.aten.mul.Tensor(mul_420, 0.1);  mul_420 = None
    mul_422: "f32[768]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_334: "f32[768]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    mul_423: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_417, primals_178);  mul_417 = None
    add_335: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_423, primals_179);  mul_423 = primals_179 = None
    view_306: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_335, [8, 16, 768]);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_307: "f32[8, 16, 12, 64]" = torch.ops.aten.reshape.default(view_306, [8, 16, 12, -1]);  view_306 = None
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(view_307, [16, 16, 32], 3);  view_307 = None
    getitem_144: "f32[8, 16, 12, 16]" = split_with_sizes_12[0]
    getitem_145: "f32[8, 16, 12, 16]" = split_with_sizes_12[1]
    getitem_146: "f32[8, 16, 12, 32]" = split_with_sizes_12[2];  split_with_sizes_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_100: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_101: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 3, 1]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_102: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_48: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_100, [8, 12, 16, 16]);  permute_100 = None
    clone_71: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_308: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_71, [96, 16, 16]);  clone_71 = None
    expand_49: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_101, [8, 12, 16, 16]);  permute_101 = None
    clone_72: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_309: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_72, [96, 16, 16]);  clone_72 = None
    bmm_24: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_308, view_309)
    view_310: "f32[8, 12, 16, 16]" = torch.ops.aten.reshape.default(bmm_24, [8, 12, 16, 16]);  bmm_24 = None
    mul_424: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_310, 0.25);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_12: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(primals_13, [None, primals_221]);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_336: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_424, index_12);  mul_424 = index_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_336, [-1], True)
    sub_67: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_336, amax_12);  add_336 = amax_12 = None
    exp_12: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_13: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_39: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_50: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_39, [8, 12, 16, 16]);  div_39 = None
    view_311: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(expand_50, [96, 16, 16]);  expand_50 = None
    expand_51: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_102, [8, 12, 16, 32]);  permute_102 = None
    clone_73: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_312: "f32[96, 16, 32]" = torch.ops.aten.reshape.default(clone_73, [96, 16, 32]);  clone_73 = None
    bmm_25: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_311, view_312)
    view_313: "f32[8, 12, 16, 32]" = torch.ops.aten.reshape.default(bmm_25, [8, 12, 16, 32]);  bmm_25 = None
    permute_103: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_74: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    view_314: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(clone_74, [8, 16, 384]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_337: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_314, 3)
    clamp_min_27: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_337, 0);  add_337 = None
    clamp_max_27: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_425: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_314, clamp_max_27);  clamp_max_27 = None
    div_40: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_425, 6);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_104: "f32[384, 384]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    view_315: "f32[128, 384]" = torch.ops.aten.reshape.default(div_40, [128, 384]);  div_40 = None
    mm_51: "f32[128, 384]" = torch.ops.aten.mm.default(view_315, permute_104)
    view_316: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_51, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_317: "f32[128, 384]" = torch.ops.aten.reshape.default(view_316, [128, 384]);  view_316 = None
    add_338: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(view_317, [0], correction = 0, keepdim = True)
    getitem_147: "f32[1, 384]" = var_mean_55[0]
    getitem_148: "f32[1, 384]" = var_mean_55[1];  var_mean_55 = None
    add_339: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_147, 1e-05)
    rsqrt_55: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
    sub_68: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_317, getitem_148);  view_317 = None
    mul_426: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_55);  sub_68 = None
    squeeze_165: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_148, [0]);  getitem_148 = None
    squeeze_166: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0]);  rsqrt_55 = None
    mul_427: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_428: "f32[384]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_340: "f32[384]" = torch.ops.aten.add.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
    squeeze_167: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_147, [0]);  getitem_147 = None
    mul_429: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0078740157480315);  squeeze_167 = None
    mul_430: "f32[384]" = torch.ops.aten.mul.Tensor(mul_429, 0.1);  mul_429 = None
    mul_431: "f32[384]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_341: "f32[384]" = torch.ops.aten.add.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
    mul_432: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_426, primals_181);  mul_426 = None
    add_342: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_432, primals_182);  mul_432 = primals_182 = None
    view_318: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_342, [8, 16, 384]);  add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_343: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_330, view_318);  add_330 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_105: "f32[384, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    view_319: "f32[128, 384]" = torch.ops.aten.reshape.default(add_343, [128, 384])
    mm_52: "f32[128, 768]" = torch.ops.aten.mm.default(view_319, permute_105)
    view_320: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_52, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_321: "f32[128, 768]" = torch.ops.aten.reshape.default(view_320, [128, 768]);  view_320 = None
    add_344: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(view_321, [0], correction = 0, keepdim = True)
    getitem_149: "f32[1, 768]" = var_mean_56[0]
    getitem_150: "f32[1, 768]" = var_mean_56[1];  var_mean_56 = None
    add_345: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_149, 1e-05)
    rsqrt_56: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
    sub_69: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_321, getitem_150);  view_321 = None
    mul_433: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_56);  sub_69 = None
    squeeze_168: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_150, [0]);  getitem_150 = None
    squeeze_169: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0]);  rsqrt_56 = None
    mul_434: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_435: "f32[768]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_346: "f32[768]" = torch.ops.aten.add.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    squeeze_170: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_149, [0]);  getitem_149 = None
    mul_436: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0078740157480315);  squeeze_170 = None
    mul_437: "f32[768]" = torch.ops.aten.mul.Tensor(mul_436, 0.1);  mul_436 = None
    mul_438: "f32[768]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_347: "f32[768]" = torch.ops.aten.add.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    mul_439: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_433, primals_184);  mul_433 = None
    add_348: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_439, primals_185);  mul_439 = primals_185 = None
    view_322: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_348, [8, 16, 768]);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_349: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_322, 3)
    clamp_min_28: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_349, 0);  add_349 = None
    clamp_max_28: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_440: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_322, clamp_max_28);  clamp_max_28 = None
    div_41: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_440, 6);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_106: "f32[768, 384]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    view_323: "f32[128, 768]" = torch.ops.aten.reshape.default(div_41, [128, 768]);  div_41 = None
    mm_53: "f32[128, 384]" = torch.ops.aten.mm.default(view_323, permute_106)
    view_324: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_53, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_325: "f32[128, 384]" = torch.ops.aten.reshape.default(view_324, [128, 384]);  view_324 = None
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(view_325, [0], correction = 0, keepdim = True)
    getitem_151: "f32[1, 384]" = var_mean_57[0]
    getitem_152: "f32[1, 384]" = var_mean_57[1];  var_mean_57 = None
    add_351: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_151, 1e-05)
    rsqrt_57: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_70: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_325, getitem_152);  view_325 = None
    mul_441: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_57);  sub_70 = None
    squeeze_171: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_152, [0]);  getitem_152 = None
    squeeze_172: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0]);  rsqrt_57 = None
    mul_442: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_443: "f32[384]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_352: "f32[384]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_173: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_151, [0]);  getitem_151 = None
    mul_444: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0078740157480315);  squeeze_173 = None
    mul_445: "f32[384]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[384]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_353: "f32[384]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    mul_447: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_441, primals_187);  mul_441 = None
    add_354: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_447, primals_188);  mul_447 = primals_188 = None
    view_326: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_354, [8, 16, 384]);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_355: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_343, view_326);  add_343 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_107: "f32[384, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    view_327: "f32[128, 384]" = torch.ops.aten.reshape.default(add_355, [128, 384])
    mm_54: "f32[128, 768]" = torch.ops.aten.mm.default(view_327, permute_107)
    view_328: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_329: "f32[128, 768]" = torch.ops.aten.reshape.default(view_328, [128, 768]);  view_328 = None
    add_356: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(view_329, [0], correction = 0, keepdim = True)
    getitem_153: "f32[1, 768]" = var_mean_58[0]
    getitem_154: "f32[1, 768]" = var_mean_58[1];  var_mean_58 = None
    add_357: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_153, 1e-05)
    rsqrt_58: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_357);  add_357 = None
    sub_71: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_329, getitem_154);  view_329 = None
    mul_448: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_58);  sub_71 = None
    squeeze_174: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_154, [0]);  getitem_154 = None
    squeeze_175: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0]);  rsqrt_58 = None
    mul_449: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_450: "f32[768]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_358: "f32[768]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_176: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_153, [0]);  getitem_153 = None
    mul_451: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0078740157480315);  squeeze_176 = None
    mul_452: "f32[768]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[768]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_359: "f32[768]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    mul_454: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_448, primals_190);  mul_448 = None
    add_360: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_454, primals_191);  mul_454 = primals_191 = None
    view_330: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_360, [8, 16, 768]);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_331: "f32[8, 16, 12, 64]" = torch.ops.aten.reshape.default(view_330, [8, 16, 12, -1]);  view_330 = None
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(view_331, [16, 16, 32], 3);  view_331 = None
    getitem_155: "f32[8, 16, 12, 16]" = split_with_sizes_13[0]
    getitem_156: "f32[8, 16, 12, 16]" = split_with_sizes_13[1]
    getitem_157: "f32[8, 16, 12, 32]" = split_with_sizes_13[2];  split_with_sizes_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    permute_108: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    permute_109: "f32[8, 12, 16, 16]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 3, 1]);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    permute_110: "f32[8, 12, 16, 32]" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    expand_52: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_108, [8, 12, 16, 16]);  permute_108 = None
    clone_76: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_332: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_76, [96, 16, 16]);  clone_76 = None
    expand_53: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(permute_109, [8, 12, 16, 16]);  permute_109 = None
    clone_77: "f32[8, 12, 16, 16]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_333: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(clone_77, [96, 16, 16]);  clone_77 = None
    bmm_26: "f32[96, 16, 16]" = torch.ops.aten.bmm.default(view_332, view_333)
    view_334: "f32[8, 12, 16, 16]" = torch.ops.aten.reshape.default(bmm_26, [8, 12, 16, 16]);  bmm_26 = None
    mul_455: "f32[8, 12, 16, 16]" = torch.ops.aten.mul.Tensor(view_334, 0.25);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:211, code: return self.attention_biases[:, self.attention_bias_idxs]
    index_13: "f32[12, 16, 16]" = torch.ops.aten.index.Tensor(primals_14, [None, primals_222]);  primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    add_361: "f32[8, 12, 16, 16]" = torch.ops.aten.add.Tensor(mul_455, index_13);  mul_455 = index_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 12, 16, 1]" = torch.ops.aten.amax.default(add_361, [-1], True)
    sub_72: "f32[8, 12, 16, 16]" = torch.ops.aten.sub.Tensor(add_361, amax_13);  add_361 = amax_13 = None
    exp_13: "f32[8, 12, 16, 16]" = torch.ops.aten.exp.default(sub_72);  sub_72 = None
    sum_14: "f32[8, 12, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_42: "f32[8, 12, 16, 16]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(div_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    expand_54: "f32[8, 12, 16, 16]" = torch.ops.aten.expand.default(div_42, [8, 12, 16, 16]);  div_42 = None
    view_335: "f32[96, 16, 16]" = torch.ops.aten.reshape.default(expand_54, [96, 16, 16]);  expand_54 = None
    expand_55: "f32[8, 12, 16, 32]" = torch.ops.aten.expand.default(permute_110, [8, 12, 16, 32]);  permute_110 = None
    clone_78: "f32[8, 12, 16, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_336: "f32[96, 16, 32]" = torch.ops.aten.reshape.default(clone_78, [96, 16, 32]);  clone_78 = None
    bmm_27: "f32[96, 16, 32]" = torch.ops.aten.bmm.default(view_335, view_336)
    view_337: "f32[8, 12, 16, 32]" = torch.ops.aten.reshape.default(bmm_27, [8, 12, 16, 32]);  bmm_27 = None
    permute_111: "f32[8, 16, 12, 32]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    clone_79: "f32[8, 16, 12, 32]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_338: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(clone_79, [8, 16, 384]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    add_362: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(view_338, 3)
    clamp_min_29: "f32[8, 16, 384]" = torch.ops.aten.clamp_min.default(add_362, 0);  add_362 = None
    clamp_max_29: "f32[8, 16, 384]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_456: "f32[8, 16, 384]" = torch.ops.aten.mul.Tensor(view_338, clamp_max_29);  clamp_max_29 = None
    div_43: "f32[8, 16, 384]" = torch.ops.aten.div.Tensor(mul_456, 6);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_112: "f32[384, 384]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    view_339: "f32[128, 384]" = torch.ops.aten.reshape.default(div_43, [128, 384]);  div_43 = None
    mm_55: "f32[128, 384]" = torch.ops.aten.mm.default(view_339, permute_112)
    view_340: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_55, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_341: "f32[128, 384]" = torch.ops.aten.reshape.default(view_340, [128, 384]);  view_340 = None
    add_363: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(view_341, [0], correction = 0, keepdim = True)
    getitem_158: "f32[1, 384]" = var_mean_59[0]
    getitem_159: "f32[1, 384]" = var_mean_59[1];  var_mean_59 = None
    add_364: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_59: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_364);  add_364 = None
    sub_73: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_341, getitem_159);  view_341 = None
    mul_457: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_59);  sub_73 = None
    squeeze_177: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_159, [0]);  getitem_159 = None
    squeeze_178: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0]);  rsqrt_59 = None
    mul_458: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_459: "f32[384]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_365: "f32[384]" = torch.ops.aten.add.Tensor(mul_458, mul_459);  mul_458 = mul_459 = None
    squeeze_179: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_158, [0]);  getitem_158 = None
    mul_460: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0078740157480315);  squeeze_179 = None
    mul_461: "f32[384]" = torch.ops.aten.mul.Tensor(mul_460, 0.1);  mul_460 = None
    mul_462: "f32[384]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_366: "f32[384]" = torch.ops.aten.add.Tensor(mul_461, mul_462);  mul_461 = mul_462 = None
    mul_463: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_457, primals_193);  mul_457 = None
    add_367: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_463, primals_194);  mul_463 = primals_194 = None
    view_342: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_367, [8, 16, 384]);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    add_368: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_355, view_342);  add_355 = view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_113: "f32[384, 768]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    view_343: "f32[128, 384]" = torch.ops.aten.reshape.default(add_368, [128, 384])
    mm_56: "f32[128, 768]" = torch.ops.aten.mm.default(view_343, permute_113)
    view_344: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(mm_56, [8, 16, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_345: "f32[128, 768]" = torch.ops.aten.reshape.default(view_344, [128, 768]);  view_344 = None
    add_369: "i64[]" = torch.ops.aten.add.Tensor(primals_405, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(view_345, [0], correction = 0, keepdim = True)
    getitem_160: "f32[1, 768]" = var_mean_60[0]
    getitem_161: "f32[1, 768]" = var_mean_60[1];  var_mean_60 = None
    add_370: "f32[1, 768]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
    rsqrt_60: "f32[1, 768]" = torch.ops.aten.rsqrt.default(add_370);  add_370 = None
    sub_74: "f32[128, 768]" = torch.ops.aten.sub.Tensor(view_345, getitem_161);  view_345 = None
    mul_464: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_60);  sub_74 = None
    squeeze_180: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_161, [0]);  getitem_161 = None
    squeeze_181: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0]);  rsqrt_60 = None
    mul_465: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_466: "f32[768]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_371: "f32[768]" = torch.ops.aten.add.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    squeeze_182: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_160, [0]);  getitem_160 = None
    mul_467: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0078740157480315);  squeeze_182 = None
    mul_468: "f32[768]" = torch.ops.aten.mul.Tensor(mul_467, 0.1);  mul_467 = None
    mul_469: "f32[768]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_372: "f32[768]" = torch.ops.aten.add.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    mul_470: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_464, primals_196);  mul_464 = None
    add_373: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_470, primals_197);  mul_470 = primals_197 = None
    view_346: "f32[8, 16, 768]" = torch.ops.aten.reshape.default(add_373, [8, 16, 768]);  add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    add_374: "f32[8, 16, 768]" = torch.ops.aten.add.Tensor(view_346, 3)
    clamp_min_30: "f32[8, 16, 768]" = torch.ops.aten.clamp_min.default(add_374, 0);  add_374 = None
    clamp_max_30: "f32[8, 16, 768]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
    mul_471: "f32[8, 16, 768]" = torch.ops.aten.mul.Tensor(view_346, clamp_max_30);  clamp_max_30 = None
    div_44: "f32[8, 16, 768]" = torch.ops.aten.div.Tensor(mul_471, 6);  mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    view_347: "f32[128, 768]" = torch.ops.aten.reshape.default(div_44, [128, 768]);  div_44 = None
    mm_57: "f32[128, 384]" = torch.ops.aten.mm.default(view_347, permute_114)
    view_348: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(mm_57, [8, 16, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    view_349: "f32[128, 384]" = torch.ops.aten.reshape.default(view_348, [128, 384]);  view_348 = None
    add_375: "i64[]" = torch.ops.aten.add.Tensor(primals_408, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(view_349, [0], correction = 0, keepdim = True)
    getitem_162: "f32[1, 384]" = var_mean_61[0]
    getitem_163: "f32[1, 384]" = var_mean_61[1];  var_mean_61 = None
    add_376: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_61: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
    sub_75: "f32[128, 384]" = torch.ops.aten.sub.Tensor(view_349, getitem_163);  view_349 = None
    mul_472: "f32[128, 384]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_61);  sub_75 = None
    squeeze_183: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_163, [0]);  getitem_163 = None
    squeeze_184: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0]);  rsqrt_61 = None
    mul_473: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_474: "f32[384]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_377: "f32[384]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    squeeze_185: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_162, [0]);  getitem_162 = None
    mul_475: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0078740157480315);  squeeze_185 = None
    mul_476: "f32[384]" = torch.ops.aten.mul.Tensor(mul_475, 0.1);  mul_475 = None
    mul_477: "f32[384]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_378: "f32[384]" = torch.ops.aten.add.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    mul_478: "f32[128, 384]" = torch.ops.aten.mul.Tensor(mul_472, primals_199);  mul_472 = None
    add_379: "f32[128, 384]" = torch.ops.aten.add.Tensor(mul_478, primals_200);  mul_478 = primals_200 = None
    view_350: "f32[8, 16, 384]" = torch.ops.aten.reshape.default(add_379, [8, 16, 384]);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    add_380: "f32[8, 16, 384]" = torch.ops.aten.add.Tensor(add_368, view_350);  add_368 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:681, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
    mean: "f32[8, 384]" = torch.ops.aten.mean.dim(add_380, [1]);  add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    add_381: "i64[]" = torch.ops.aten.add.Tensor(primals_411, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(mean, [0], correction = 0, keepdim = True)
    getitem_164: "f32[1, 384]" = var_mean_62[0]
    getitem_165: "f32[1, 384]" = var_mean_62[1];  var_mean_62 = None
    add_382: "f32[1, 384]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
    rsqrt_62: "f32[1, 384]" = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
    sub_76: "f32[8, 384]" = torch.ops.aten.sub.Tensor(mean, getitem_165)
    mul_479: "f32[8, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_62);  sub_76 = rsqrt_62 = None
    squeeze_186: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_165, [0]);  getitem_165 = None
    mul_480: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1);  squeeze_186 = None
    mul_481: "f32[384]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_383: "f32[384]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_481 = None
    squeeze_188: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_164, [0]);  getitem_164 = None
    mul_482: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.1428571428571428);  squeeze_188 = None
    mul_483: "f32[384]" = torch.ops.aten.mul.Tensor(mul_482, 0.1);  mul_482 = None
    mul_484: "f32[384]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_384: "f32[384]" = torch.ops.aten.add.Tensor(mul_483, mul_484);  mul_484 = None
    mul_485: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_479, primals_201)
    add_385: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_485, primals_202);  mul_485 = primals_202 = None
    permute_115: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[8, 1000]" = torch.ops.aten.mm.default(add_385, permute_115)
    add_tensor_1: "f32[8, 1000]" = torch.ops.aten.add.Tensor(mm_default_1, primals_204);  mm_default_1 = primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    add_386: "i64[]" = torch.ops.aten.add.Tensor(primals_414, 1)
    mul_488: "f32[384]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_388: "f32[384]" = torch.ops.aten.add.Tensor(mul_480, mul_488);  mul_480 = mul_488 = None
    mul_491: "f32[384]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_389: "f32[384]" = torch.ops.aten.add.Tensor(mul_483, mul_491);  mul_483 = mul_491 = None
    mul_492: "f32[8, 384]" = torch.ops.aten.mul.Tensor(mul_479, primals_205);  mul_479 = None
    add_390: "f32[8, 384]" = torch.ops.aten.add.Tensor(mul_492, primals_206);  mul_492 = primals_206 = None
    permute_116: "f32[384, 1000]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[8, 1000]" = torch.ops.aten.mm.default(add_390, permute_116)
    add_tensor: "f32[8, 1000]" = torch.ops.aten.add.Tensor(mm_default, primals_208);  mm_default = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:690, code: return (x + x_dist) / 2
    add_391: "f32[8, 1000]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor);  add_tensor_1 = add_tensor = None
    div_45: "f32[8, 1000]" = torch.ops.aten.div.Tensor(add_391, 2);  add_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    permute_117: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    permute_121: "f32[1000, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_25: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_127: "f32[384, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_29: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_131: "f32[768, 384]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_33: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_135: "f32[384, 384]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_138: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
    permute_139: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_14: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_140: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    permute_141: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_37: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_147: "f32[768, 384]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_41: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_151: "f32[384, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_45: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_155: "f32[768, 384]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_49: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_159: "f32[384, 384]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_162: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
    permute_163: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_15: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_164: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
    permute_165: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_53: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_171: "f32[768, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_57: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_175: "f32[384, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_61: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_179: "f32[768, 384]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_65: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_183: "f32[384, 384]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_186: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    permute_187: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_16: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_188: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    permute_189: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_69: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_195: "f32[768, 384]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_73: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_199: "f32[384, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_77: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_203: "f32[768, 384]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_81: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_207: "f32[384, 384]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_210: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
    permute_211: "f32[96, 32, 16]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_17: "f32[8, 12, 16, 16]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_212: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
    permute_213: "f32[96, 16, 16]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_85: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_219: "f32[768, 384]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_89: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_223: "f32[384, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_93: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_227: "f32[768, 384]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_97: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_231: "f32[384, 1024]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    permute_234: "f32[128, 49, 16]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    permute_235: "f32[128, 64, 49]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    alias_18: "f32[8, 16, 16, 49]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_236: "f32[128, 16, 16]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
    permute_237: "f32[128, 49, 16]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_101: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_241: "f32[256, 256]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_105: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_247: "f32[1280, 256]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_109: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_251: "f32[256, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_113: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_255: "f32[512, 256]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_117: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_259: "f32[256, 256]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_262: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    permute_263: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_19: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_264: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    permute_265: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_121: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_271: "f32[512, 256]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_125: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_275: "f32[256, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_129: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_279: "f32[512, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_133: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_283: "f32[256, 256]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_286: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None
    permute_287: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_20: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_288: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    permute_289: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_137: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_295: "f32[512, 256]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_141: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_299: "f32[256, 512]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_303: "f32[512, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_149: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_307: "f32[256, 256]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_310: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
    permute_311: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_21: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_312: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    permute_313: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_153: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_319: "f32[512, 256]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_157: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_323: "f32[256, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_161: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_327: "f32[512, 256]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_165: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_331: "f32[256, 256]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_334: "f32[64, 49, 49]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    permute_335: "f32[64, 32, 49]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_22: "f32[8, 8, 49, 49]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_336: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    permute_337: "f32[64, 49, 16]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_169: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_343: "f32[512, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_173: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_347: "f32[256, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_177: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_351: "f32[512, 256]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_181: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_355: "f32[256, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    permute_358: "f32[64, 196, 49]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    permute_359: "f32[64, 64, 196]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    alias_23: "f32[8, 8, 49, 196]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_360: "f32[64, 16, 49]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
    permute_361: "f32[64, 196, 16]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_185: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_365: "f32[128, 128]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_189: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_371: "f32[640, 128]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_193: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_375: "f32[128, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_197: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_379: "f32[256, 128]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_201: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_386: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    permute_387: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_388: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    permute_389: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_205: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_395: "f32[256, 128]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_209: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_399: "f32[128, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_213: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_403: "f32[256, 128]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_217: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_407: "f32[128, 128]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_410: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    permute_411: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_412: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    permute_413: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_221: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_419: "f32[256, 128]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_225: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_423: "f32[128, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_427: "f32[256, 128]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_434: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    permute_435: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_26: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_436: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
    permute_437: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_237: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_443: "f32[256, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_241: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_447: "f32[128, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_245: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_451: "f32[256, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_249: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_455: "f32[128, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    permute_458: "f32[32, 196, 196]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_459: "f32[32, 32, 196]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    alias_27: "f32[8, 4, 196, 196]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    permute_460: "f32[32, 16, 196]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    permute_461: "f32[32, 196, 16]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    unsqueeze_253: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    permute_467: "f32[256, 128]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    unsqueeze_257: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_258: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    unsqueeze_269: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_270: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    unsqueeze_281: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_282: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    unsqueeze_293: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_294: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[16]" = torch.ops.aten.copy_.default(primals_223, add_2);  primals_223 = add_2 = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_224, add_3);  primals_224 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_225, add);  primals_225 = add = None
    copy__3: "f32[32]" = torch.ops.aten.copy_.default(primals_226, add_8);  primals_226 = add_8 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_227, add_9);  primals_227 = add_9 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_228, add_6);  primals_228 = add_6 = None
    copy__6: "f32[64]" = torch.ops.aten.copy_.default(primals_229, add_14);  primals_229 = add_14 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_230, add_15);  primals_230 = add_15 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_12);  primals_231 = add_12 = None
    copy__9: "f32[128]" = torch.ops.aten.copy_.default(primals_232, add_20);  primals_232 = add_20 = None
    copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_233, add_21);  primals_233 = add_21 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_234, add_18);  primals_234 = add_18 = None
    copy__12: "f32[256]" = torch.ops.aten.copy_.default(primals_235, add_25);  primals_235 = add_25 = None
    copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_236, add_26);  primals_236 = add_26 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_23);  primals_237 = add_23 = None
    copy__15: "f32[128]" = torch.ops.aten.copy_.default(primals_238, add_32);  primals_238 = add_32 = None
    copy__16: "f32[128]" = torch.ops.aten.copy_.default(primals_239, add_33);  primals_239 = add_33 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_240, add_30);  primals_240 = add_30 = None
    copy__18: "f32[256]" = torch.ops.aten.copy_.default(primals_241, add_38);  primals_241 = add_38 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_242, add_39);  primals_242 = add_39 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_36);  primals_243 = add_36 = None
    copy__21: "f32[128]" = torch.ops.aten.copy_.default(primals_244, add_44);  primals_244 = add_44 = None
    copy__22: "f32[128]" = torch.ops.aten.copy_.default(primals_245, add_45);  primals_245 = add_45 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_246, add_42);  primals_246 = add_42 = None
    copy__24: "f32[256]" = torch.ops.aten.copy_.default(primals_247, add_50);  primals_247 = add_50 = None
    copy__25: "f32[256]" = torch.ops.aten.copy_.default(primals_248, add_51);  primals_248 = add_51 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_48);  primals_249 = add_48 = None
    copy__27: "f32[128]" = torch.ops.aten.copy_.default(primals_250, add_57);  primals_250 = add_57 = None
    copy__28: "f32[128]" = torch.ops.aten.copy_.default(primals_251, add_58);  primals_251 = add_58 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_252, add_55);  primals_252 = add_55 = None
    copy__30: "f32[256]" = torch.ops.aten.copy_.default(primals_253, add_63);  primals_253 = add_63 = None
    copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_254, add_64);  primals_254 = add_64 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_61);  primals_255 = add_61 = None
    copy__33: "f32[128]" = torch.ops.aten.copy_.default(primals_256, add_69);  primals_256 = add_69 = None
    copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_257, add_70);  primals_257 = add_70 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_258, add_67);  primals_258 = add_67 = None
    copy__36: "f32[256]" = torch.ops.aten.copy_.default(primals_259, add_75);  primals_259 = add_75 = None
    copy__37: "f32[256]" = torch.ops.aten.copy_.default(primals_260, add_76);  primals_260 = add_76 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_73);  primals_261 = add_73 = None
    copy__39: "f32[128]" = torch.ops.aten.copy_.default(primals_262, add_82);  primals_262 = add_82 = None
    copy__40: "f32[128]" = torch.ops.aten.copy_.default(primals_263, add_83);  primals_263 = add_83 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_264, add_80);  primals_264 = add_80 = None
    copy__42: "f32[256]" = torch.ops.aten.copy_.default(primals_265, add_88);  primals_265 = add_88 = None
    copy__43: "f32[256]" = torch.ops.aten.copy_.default(primals_266, add_89);  primals_266 = add_89 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_86);  primals_267 = add_86 = None
    copy__45: "f32[128]" = torch.ops.aten.copy_.default(primals_268, add_94);  primals_268 = add_94 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_269, add_95);  primals_269 = add_95 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_270, add_92);  primals_270 = add_92 = None
    copy__48: "f32[256]" = torch.ops.aten.copy_.default(primals_271, add_100);  primals_271 = add_100 = None
    copy__49: "f32[256]" = torch.ops.aten.copy_.default(primals_272, add_101);  primals_272 = add_101 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_98);  primals_273 = add_98 = None
    copy__51: "f32[128]" = torch.ops.aten.copy_.default(primals_274, add_107);  primals_274 = add_107 = None
    copy__52: "f32[128]" = torch.ops.aten.copy_.default(primals_275, add_108);  primals_275 = add_108 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_276, add_105);  primals_276 = add_105 = None
    copy__54: "f32[256]" = torch.ops.aten.copy_.default(primals_277, add_113);  primals_277 = add_113 = None
    copy__55: "f32[256]" = torch.ops.aten.copy_.default(primals_278, add_114);  primals_278 = add_114 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_111);  primals_279 = add_111 = None
    copy__57: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_119);  primals_280 = add_119 = None
    copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_281, add_120);  primals_281 = add_120 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_282, add_117);  primals_282 = add_117 = None
    copy__60: "f32[640]" = torch.ops.aten.copy_.default(primals_283, add_125);  primals_283 = add_125 = None
    copy__61: "f32[640]" = torch.ops.aten.copy_.default(primals_284, add_126);  primals_284 = add_126 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_123);  primals_285 = add_123 = None
    copy__63: "f32[128]" = torch.ops.aten.copy_.default(primals_286, add_130);  primals_286 = add_130 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_287, add_131);  primals_287 = add_131 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_128);  primals_288 = add_128 = None
    copy__66: "f32[256]" = torch.ops.aten.copy_.default(primals_289, add_137);  primals_289 = add_137 = None
    copy__67: "f32[256]" = torch.ops.aten.copy_.default(primals_290, add_138);  primals_290 = add_138 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_135);  primals_291 = add_135 = None
    copy__69: "f32[512]" = torch.ops.aten.copy_.default(primals_292, add_142);  primals_292 = add_142 = None
    copy__70: "f32[512]" = torch.ops.aten.copy_.default(primals_293, add_143);  primals_293 = add_143 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_140);  primals_294 = add_140 = None
    copy__72: "f32[256]" = torch.ops.aten.copy_.default(primals_295, add_148);  primals_295 = add_148 = None
    copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_296, add_149);  primals_296 = add_149 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_146);  primals_297 = add_146 = None
    copy__75: "f32[512]" = torch.ops.aten.copy_.default(primals_298, add_154);  primals_298 = add_154 = None
    copy__76: "f32[512]" = torch.ops.aten.copy_.default(primals_299, add_155);  primals_299 = add_155 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_152);  primals_300 = add_152 = None
    copy__78: "f32[256]" = torch.ops.aten.copy_.default(primals_301, add_161);  primals_301 = add_161 = None
    copy__79: "f32[256]" = torch.ops.aten.copy_.default(primals_302, add_162);  primals_302 = add_162 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_159);  primals_303 = add_159 = None
    copy__81: "f32[512]" = torch.ops.aten.copy_.default(primals_304, add_167);  primals_304 = add_167 = None
    copy__82: "f32[512]" = torch.ops.aten.copy_.default(primals_305, add_168);  primals_305 = add_168 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_165);  primals_306 = add_165 = None
    copy__84: "f32[256]" = torch.ops.aten.copy_.default(primals_307, add_173);  primals_307 = add_173 = None
    copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_308, add_174);  primals_308 = add_174 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_171);  primals_309 = add_171 = None
    copy__87: "f32[512]" = torch.ops.aten.copy_.default(primals_310, add_179);  primals_310 = add_179 = None
    copy__88: "f32[512]" = torch.ops.aten.copy_.default(primals_311, add_180);  primals_311 = add_180 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_177);  primals_312 = add_177 = None
    copy__90: "f32[256]" = torch.ops.aten.copy_.default(primals_313, add_186);  primals_313 = add_186 = None
    copy__91: "f32[256]" = torch.ops.aten.copy_.default(primals_314, add_187);  primals_314 = add_187 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_184);  primals_315 = add_184 = None
    copy__93: "f32[512]" = torch.ops.aten.copy_.default(primals_316, add_192);  primals_316 = add_192 = None
    copy__94: "f32[512]" = torch.ops.aten.copy_.default(primals_317, add_193);  primals_317 = add_193 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_190);  primals_318 = add_190 = None
    copy__96: "f32[256]" = torch.ops.aten.copy_.default(primals_319, add_198);  primals_319 = add_198 = None
    copy__97: "f32[256]" = torch.ops.aten.copy_.default(primals_320, add_199);  primals_320 = add_199 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_196);  primals_321 = add_196 = None
    copy__99: "f32[512]" = torch.ops.aten.copy_.default(primals_322, add_204);  primals_322 = add_204 = None
    copy__100: "f32[512]" = torch.ops.aten.copy_.default(primals_323, add_205);  primals_323 = add_205 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_202);  primals_324 = add_202 = None
    copy__102: "f32[256]" = torch.ops.aten.copy_.default(primals_325, add_211);  primals_325 = add_211 = None
    copy__103: "f32[256]" = torch.ops.aten.copy_.default(primals_326, add_212);  primals_326 = add_212 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_209);  primals_327 = add_209 = None
    copy__105: "f32[512]" = torch.ops.aten.copy_.default(primals_328, add_217);  primals_328 = add_217 = None
    copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_329, add_218);  primals_329 = add_218 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_215);  primals_330 = add_215 = None
    copy__108: "f32[256]" = torch.ops.aten.copy_.default(primals_331, add_223);  primals_331 = add_223 = None
    copy__109: "f32[256]" = torch.ops.aten.copy_.default(primals_332, add_224);  primals_332 = add_224 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_221);  primals_333 = add_221 = None
    copy__111: "f32[512]" = torch.ops.aten.copy_.default(primals_334, add_229);  primals_334 = add_229 = None
    copy__112: "f32[512]" = torch.ops.aten.copy_.default(primals_335, add_230);  primals_335 = add_230 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_227);  primals_336 = add_227 = None
    copy__114: "f32[256]" = torch.ops.aten.copy_.default(primals_337, add_236);  primals_337 = add_236 = None
    copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_338, add_237);  primals_338 = add_237 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_234);  primals_339 = add_234 = None
    copy__117: "f32[512]" = torch.ops.aten.copy_.default(primals_340, add_242);  primals_340 = add_242 = None
    copy__118: "f32[512]" = torch.ops.aten.copy_.default(primals_341, add_243);  primals_341 = add_243 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_240);  primals_342 = add_240 = None
    copy__120: "f32[256]" = torch.ops.aten.copy_.default(primals_343, add_248);  primals_343 = add_248 = None
    copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_344, add_249);  primals_344 = add_249 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_246);  primals_345 = add_246 = None
    copy__123: "f32[1280]" = torch.ops.aten.copy_.default(primals_346, add_254);  primals_346 = add_254 = None
    copy__124: "f32[1280]" = torch.ops.aten.copy_.default(primals_347, add_255);  primals_347 = add_255 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_252);  primals_348 = add_252 = None
    copy__126: "f32[256]" = torch.ops.aten.copy_.default(primals_349, add_259);  primals_349 = add_259 = None
    copy__127: "f32[256]" = torch.ops.aten.copy_.default(primals_350, add_260);  primals_350 = add_260 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_257);  primals_351 = add_257 = None
    copy__129: "f32[384]" = torch.ops.aten.copy_.default(primals_352, add_266);  primals_352 = add_266 = None
    copy__130: "f32[384]" = torch.ops.aten.copy_.default(primals_353, add_267);  primals_353 = add_267 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_264);  primals_354 = add_264 = None
    copy__132: "f32[768]" = torch.ops.aten.copy_.default(primals_355, add_271);  primals_355 = add_271 = None
    copy__133: "f32[768]" = torch.ops.aten.copy_.default(primals_356, add_272);  primals_356 = add_272 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_269);  primals_357 = add_269 = None
    copy__135: "f32[384]" = torch.ops.aten.copy_.default(primals_358, add_277);  primals_358 = add_277 = None
    copy__136: "f32[384]" = torch.ops.aten.copy_.default(primals_359, add_278);  primals_359 = add_278 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_275);  primals_360 = add_275 = None
    copy__138: "f32[768]" = torch.ops.aten.copy_.default(primals_361, add_283);  primals_361 = add_283 = None
    copy__139: "f32[768]" = torch.ops.aten.copy_.default(primals_362, add_284);  primals_362 = add_284 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_281);  primals_363 = add_281 = None
    copy__141: "f32[384]" = torch.ops.aten.copy_.default(primals_364, add_290);  primals_364 = add_290 = None
    copy__142: "f32[384]" = torch.ops.aten.copy_.default(primals_365, add_291);  primals_365 = add_291 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_288);  primals_366 = add_288 = None
    copy__144: "f32[768]" = torch.ops.aten.copy_.default(primals_367, add_296);  primals_367 = add_296 = None
    copy__145: "f32[768]" = torch.ops.aten.copy_.default(primals_368, add_297);  primals_368 = add_297 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_294);  primals_369 = add_294 = None
    copy__147: "f32[384]" = torch.ops.aten.copy_.default(primals_370, add_302);  primals_370 = add_302 = None
    copy__148: "f32[384]" = torch.ops.aten.copy_.default(primals_371, add_303);  primals_371 = add_303 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_300);  primals_372 = add_300 = None
    copy__150: "f32[768]" = torch.ops.aten.copy_.default(primals_373, add_308);  primals_373 = add_308 = None
    copy__151: "f32[768]" = torch.ops.aten.copy_.default(primals_374, add_309);  primals_374 = add_309 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_306);  primals_375 = add_306 = None
    copy__153: "f32[384]" = torch.ops.aten.copy_.default(primals_376, add_315);  primals_376 = add_315 = None
    copy__154: "f32[384]" = torch.ops.aten.copy_.default(primals_377, add_316);  primals_377 = add_316 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_313);  primals_378 = add_313 = None
    copy__156: "f32[768]" = torch.ops.aten.copy_.default(primals_379, add_321);  primals_379 = add_321 = None
    copy__157: "f32[768]" = torch.ops.aten.copy_.default(primals_380, add_322);  primals_380 = add_322 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_319);  primals_381 = add_319 = None
    copy__159: "f32[384]" = torch.ops.aten.copy_.default(primals_382, add_327);  primals_382 = add_327 = None
    copy__160: "f32[384]" = torch.ops.aten.copy_.default(primals_383, add_328);  primals_383 = add_328 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_325);  primals_384 = add_325 = None
    copy__162: "f32[768]" = torch.ops.aten.copy_.default(primals_385, add_333);  primals_385 = add_333 = None
    copy__163: "f32[768]" = torch.ops.aten.copy_.default(primals_386, add_334);  primals_386 = add_334 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_331);  primals_387 = add_331 = None
    copy__165: "f32[384]" = torch.ops.aten.copy_.default(primals_388, add_340);  primals_388 = add_340 = None
    copy__166: "f32[384]" = torch.ops.aten.copy_.default(primals_389, add_341);  primals_389 = add_341 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_338);  primals_390 = add_338 = None
    copy__168: "f32[768]" = torch.ops.aten.copy_.default(primals_391, add_346);  primals_391 = add_346 = None
    copy__169: "f32[768]" = torch.ops.aten.copy_.default(primals_392, add_347);  primals_392 = add_347 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_344);  primals_393 = add_344 = None
    copy__171: "f32[384]" = torch.ops.aten.copy_.default(primals_394, add_352);  primals_394 = add_352 = None
    copy__172: "f32[384]" = torch.ops.aten.copy_.default(primals_395, add_353);  primals_395 = add_353 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_350);  primals_396 = add_350 = None
    copy__174: "f32[768]" = torch.ops.aten.copy_.default(primals_397, add_358);  primals_397 = add_358 = None
    copy__175: "f32[768]" = torch.ops.aten.copy_.default(primals_398, add_359);  primals_398 = add_359 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_356);  primals_399 = add_356 = None
    copy__177: "f32[384]" = torch.ops.aten.copy_.default(primals_400, add_365);  primals_400 = add_365 = None
    copy__178: "f32[384]" = torch.ops.aten.copy_.default(primals_401, add_366);  primals_401 = add_366 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_363);  primals_402 = add_363 = None
    copy__180: "f32[768]" = torch.ops.aten.copy_.default(primals_403, add_371);  primals_403 = add_371 = None
    copy__181: "f32[768]" = torch.ops.aten.copy_.default(primals_404, add_372);  primals_404 = add_372 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_405, add_369);  primals_405 = add_369 = None
    copy__183: "f32[384]" = torch.ops.aten.copy_.default(primals_406, add_377);  primals_406 = add_377 = None
    copy__184: "f32[384]" = torch.ops.aten.copy_.default(primals_407, add_378);  primals_407 = add_378 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_408, add_375);  primals_408 = add_375 = None
    copy__186: "f32[384]" = torch.ops.aten.copy_.default(primals_409, add_383);  primals_409 = add_383 = None
    copy__187: "f32[384]" = torch.ops.aten.copy_.default(primals_410, add_384);  primals_410 = add_384 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_411, add_381);  primals_411 = add_381 = None
    copy__189: "f32[384]" = torch.ops.aten.copy_.default(primals_412, add_388);  primals_412 = add_388 = None
    copy__190: "f32[384]" = torch.ops.aten.copy_.default(primals_413, add_389);  primals_413 = add_389 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_414, add_386);  primals_414 = add_386 = None
    return [div_45, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_415, convolution, squeeze_1, add_4, div, convolution_1, squeeze_4, add_10, div_1, convolution_2, squeeze_7, add_16, div_2, convolution_3, squeeze_10, view_1, mm, squeeze_13, view_12, view_13, mm_1, squeeze_16, view_17, mm_2, squeeze_19, view_20, view_21, mm_3, squeeze_22, view_25, mm_4, squeeze_25, view_36, view_37, mm_5, squeeze_28, view_41, mm_6, squeeze_31, view_44, view_45, mm_7, squeeze_34, view_49, mm_8, squeeze_37, view_60, view_61, mm_9, squeeze_40, view_65, mm_10, squeeze_43, view_68, view_69, mm_11, squeeze_46, view_73, mm_12, squeeze_49, view_84, view_85, mm_13, squeeze_52, view_89, mm_14, squeeze_55, view_92, view_93, mm_15, squeeze_58, view_97, mm_16, squeeze_61, view_104, mm_17, squeeze_64, view_115, view_116, mm_18, squeeze_67, view_120, mm_19, squeeze_70, view_123, view_124, mm_20, squeeze_73, view_128, mm_21, squeeze_76, view_139, view_140, mm_22, squeeze_79, view_144, mm_23, squeeze_82, view_147, view_148, mm_24, squeeze_85, view_152, mm_25, squeeze_88, view_163, view_164, mm_26, squeeze_91, view_168, mm_27, squeeze_94, view_171, view_172, mm_28, squeeze_97, view_176, mm_29, squeeze_100, view_187, view_188, mm_30, squeeze_103, view_192, mm_31, squeeze_106, view_195, view_196, mm_32, squeeze_109, view_200, mm_33, squeeze_112, view_211, view_212, mm_34, squeeze_115, view_216, mm_35, squeeze_118, view_219, view_220, mm_36, squeeze_121, view_224, mm_37, squeeze_124, view_231, mm_38, squeeze_127, view_242, view_243, mm_39, squeeze_130, view_247, mm_40, squeeze_133, view_250, view_251, mm_41, squeeze_136, view_255, mm_42, squeeze_139, view_266, view_267, mm_43, squeeze_142, view_271, mm_44, squeeze_145, view_274, view_275, mm_45, squeeze_148, view_279, mm_46, squeeze_151, view_290, view_291, mm_47, squeeze_154, view_295, mm_48, squeeze_157, view_298, view_299, mm_49, squeeze_160, view_303, mm_50, squeeze_163, view_314, view_315, mm_51, squeeze_166, view_319, mm_52, squeeze_169, view_322, view_323, mm_53, squeeze_172, view_327, mm_54, squeeze_175, view_338, view_339, mm_55, squeeze_178, view_343, mm_56, squeeze_181, view_346, view_347, mm_57, squeeze_184, mean, add_385, add_390, permute_117, permute_121, unsqueeze_25, permute_127, unsqueeze_29, permute_131, unsqueeze_33, permute_135, permute_138, permute_139, alias_14, permute_140, permute_141, unsqueeze_37, permute_147, unsqueeze_41, permute_151, unsqueeze_45, permute_155, unsqueeze_49, permute_159, permute_162, permute_163, alias_15, permute_164, permute_165, unsqueeze_53, permute_171, unsqueeze_57, permute_175, unsqueeze_61, permute_179, unsqueeze_65, permute_183, permute_186, permute_187, alias_16, permute_188, permute_189, unsqueeze_69, permute_195, unsqueeze_73, permute_199, unsqueeze_77, permute_203, unsqueeze_81, permute_207, permute_210, permute_211, alias_17, permute_212, permute_213, unsqueeze_85, permute_219, unsqueeze_89, permute_223, unsqueeze_93, permute_227, unsqueeze_97, permute_231, permute_234, permute_235, alias_18, permute_236, permute_237, unsqueeze_101, permute_241, unsqueeze_105, permute_247, unsqueeze_109, permute_251, unsqueeze_113, permute_255, unsqueeze_117, permute_259, permute_262, permute_263, alias_19, permute_264, permute_265, unsqueeze_121, permute_271, unsqueeze_125, permute_275, unsqueeze_129, permute_279, unsqueeze_133, permute_283, permute_286, permute_287, alias_20, permute_288, permute_289, unsqueeze_137, permute_295, unsqueeze_141, permute_299, unsqueeze_145, permute_303, unsqueeze_149, permute_307, permute_310, permute_311, alias_21, permute_312, permute_313, unsqueeze_153, permute_319, unsqueeze_157, permute_323, unsqueeze_161, permute_327, unsqueeze_165, permute_331, permute_334, permute_335, alias_22, permute_336, permute_337, unsqueeze_169, permute_343, unsqueeze_173, permute_347, unsqueeze_177, permute_351, unsqueeze_181, permute_355, permute_358, permute_359, alias_23, permute_360, permute_361, unsqueeze_185, permute_365, unsqueeze_189, permute_371, unsqueeze_193, permute_375, unsqueeze_197, permute_379, unsqueeze_201, permute_383, permute_386, permute_387, alias_24, permute_388, permute_389, unsqueeze_205, permute_395, unsqueeze_209, permute_399, unsqueeze_213, permute_403, unsqueeze_217, permute_407, permute_410, permute_411, alias_25, permute_412, permute_413, unsqueeze_221, permute_419, unsqueeze_225, permute_423, unsqueeze_229, permute_427, unsqueeze_233, permute_431, permute_434, permute_435, alias_26, permute_436, permute_437, unsqueeze_237, permute_443, unsqueeze_241, permute_447, unsqueeze_245, permute_451, unsqueeze_249, permute_455, permute_458, permute_459, alias_27, permute_460, permute_461, unsqueeze_253, permute_467, unsqueeze_259, unsqueeze_271, unsqueeze_283, unsqueeze_295]
    