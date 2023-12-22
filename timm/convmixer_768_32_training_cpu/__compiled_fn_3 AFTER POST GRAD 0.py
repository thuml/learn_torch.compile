from __future__ import annotations



def forward(self, primals_1: "f32[768, 3, 7, 7]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768, 1, 7, 7]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768, 768, 1, 1]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768, 1, 7, 7]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768, 768, 1, 1]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768, 1, 7, 7]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768, 768, 1, 1]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768, 1, 7, 7]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768, 768, 1, 1]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768, 1, 7, 7]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768, 768, 1, 1]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768, 1, 7, 7]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768, 768, 1, 1]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768, 1, 7, 7]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768, 768, 1, 1]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768, 1, 7, 7]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768, 768, 1, 1]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768, 1, 7, 7]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768, 768, 1, 1]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768, 1, 7, 7]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768, 768, 1, 1]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768, 1, 7, 7]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768, 768, 1, 1]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768, 1, 7, 7]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[768, 768, 1, 1]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768, 1, 7, 7]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768, 768, 1, 1]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[768, 1, 7, 7]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768, 768, 1, 1]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[768, 1, 7, 7]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768, 768, 1, 1]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[768, 1, 7, 7]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[768, 768, 1, 1]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[768, 1, 7, 7]", primals_134: "f32[768]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[768, 768, 1, 1]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[768, 1, 7, 7]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768, 768, 1, 1]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[768, 1, 7, 7]", primals_150: "f32[768]", primals_151: "f32[768]", primals_152: "f32[768]", primals_153: "f32[768, 768, 1, 1]", primals_154: "f32[768]", primals_155: "f32[768]", primals_156: "f32[768]", primals_157: "f32[768, 1, 7, 7]", primals_158: "f32[768]", primals_159: "f32[768]", primals_160: "f32[768]", primals_161: "f32[768, 768, 1, 1]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768, 1, 7, 7]", primals_166: "f32[768]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[768, 768, 1, 1]", primals_170: "f32[768]", primals_171: "f32[768]", primals_172: "f32[768]", primals_173: "f32[768, 1, 7, 7]", primals_174: "f32[768]", primals_175: "f32[768]", primals_176: "f32[768]", primals_177: "f32[768, 768, 1, 1]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "f32[768, 1, 7, 7]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768, 768, 1, 1]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[768, 1, 7, 7]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[768]", primals_193: "f32[768, 768, 1, 1]", primals_194: "f32[768]", primals_195: "f32[768]", primals_196: "f32[768]", primals_197: "f32[768, 1, 7, 7]", primals_198: "f32[768]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768, 768, 1, 1]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "f32[768, 1, 7, 7]", primals_206: "f32[768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[768, 768, 1, 1]", primals_210: "f32[768]", primals_211: "f32[768]", primals_212: "f32[768]", primals_213: "f32[768, 1, 7, 7]", primals_214: "f32[768]", primals_215: "f32[768]", primals_216: "f32[768]", primals_217: "f32[768, 768, 1, 1]", primals_218: "f32[768]", primals_219: "f32[768]", primals_220: "f32[768]", primals_221: "f32[768, 1, 7, 7]", primals_222: "f32[768]", primals_223: "f32[768]", primals_224: "f32[768]", primals_225: "f32[768, 768, 1, 1]", primals_226: "f32[768]", primals_227: "f32[768]", primals_228: "f32[768]", primals_229: "f32[768, 1, 7, 7]", primals_230: "f32[768]", primals_231: "f32[768]", primals_232: "f32[768]", primals_233: "f32[768, 768, 1, 1]", primals_234: "f32[768]", primals_235: "f32[768]", primals_236: "f32[768]", primals_237: "f32[768, 1, 7, 7]", primals_238: "f32[768]", primals_239: "f32[768]", primals_240: "f32[768]", primals_241: "f32[768, 768, 1, 1]", primals_242: "f32[768]", primals_243: "f32[768]", primals_244: "f32[768]", primals_245: "f32[768, 1, 7, 7]", primals_246: "f32[768]", primals_247: "f32[768]", primals_248: "f32[768]", primals_249: "f32[768, 768, 1, 1]", primals_250: "f32[768]", primals_251: "f32[768]", primals_252: "f32[768]", primals_253: "f32[768, 1, 7, 7]", primals_254: "f32[768]", primals_255: "f32[768]", primals_256: "f32[768]", primals_257: "f32[768, 768, 1, 1]", primals_258: "f32[768]", primals_259: "f32[768]", primals_260: "f32[768]", primals_261: "f32[1000, 768]", primals_262: "f32[1000]", primals_263: "f32[768]", primals_264: "f32[768]", primals_265: "i64[]", primals_266: "f32[768]", primals_267: "f32[768]", primals_268: "i64[]", primals_269: "f32[768]", primals_270: "f32[768]", primals_271: "i64[]", primals_272: "f32[768]", primals_273: "f32[768]", primals_274: "i64[]", primals_275: "f32[768]", primals_276: "f32[768]", primals_277: "i64[]", primals_278: "f32[768]", primals_279: "f32[768]", primals_280: "i64[]", primals_281: "f32[768]", primals_282: "f32[768]", primals_283: "i64[]", primals_284: "f32[768]", primals_285: "f32[768]", primals_286: "i64[]", primals_287: "f32[768]", primals_288: "f32[768]", primals_289: "i64[]", primals_290: "f32[768]", primals_291: "f32[768]", primals_292: "i64[]", primals_293: "f32[768]", primals_294: "f32[768]", primals_295: "i64[]", primals_296: "f32[768]", primals_297: "f32[768]", primals_298: "i64[]", primals_299: "f32[768]", primals_300: "f32[768]", primals_301: "i64[]", primals_302: "f32[768]", primals_303: "f32[768]", primals_304: "i64[]", primals_305: "f32[768]", primals_306: "f32[768]", primals_307: "i64[]", primals_308: "f32[768]", primals_309: "f32[768]", primals_310: "i64[]", primals_311: "f32[768]", primals_312: "f32[768]", primals_313: "i64[]", primals_314: "f32[768]", primals_315: "f32[768]", primals_316: "i64[]", primals_317: "f32[768]", primals_318: "f32[768]", primals_319: "i64[]", primals_320: "f32[768]", primals_321: "f32[768]", primals_322: "i64[]", primals_323: "f32[768]", primals_324: "f32[768]", primals_325: "i64[]", primals_326: "f32[768]", primals_327: "f32[768]", primals_328: "i64[]", primals_329: "f32[768]", primals_330: "f32[768]", primals_331: "i64[]", primals_332: "f32[768]", primals_333: "f32[768]", primals_334: "i64[]", primals_335: "f32[768]", primals_336: "f32[768]", primals_337: "i64[]", primals_338: "f32[768]", primals_339: "f32[768]", primals_340: "i64[]", primals_341: "f32[768]", primals_342: "f32[768]", primals_343: "i64[]", primals_344: "f32[768]", primals_345: "f32[768]", primals_346: "i64[]", primals_347: "f32[768]", primals_348: "f32[768]", primals_349: "i64[]", primals_350: "f32[768]", primals_351: "f32[768]", primals_352: "i64[]", primals_353: "f32[768]", primals_354: "f32[768]", primals_355: "i64[]", primals_356: "f32[768]", primals_357: "f32[768]", primals_358: "i64[]", primals_359: "f32[768]", primals_360: "f32[768]", primals_361: "i64[]", primals_362: "f32[768]", primals_363: "f32[768]", primals_364: "i64[]", primals_365: "f32[768]", primals_366: "f32[768]", primals_367: "i64[]", primals_368: "f32[768]", primals_369: "f32[768]", primals_370: "i64[]", primals_371: "f32[768]", primals_372: "f32[768]", primals_373: "i64[]", primals_374: "f32[768]", primals_375: "f32[768]", primals_376: "i64[]", primals_377: "f32[768]", primals_378: "f32[768]", primals_379: "i64[]", primals_380: "f32[768]", primals_381: "f32[768]", primals_382: "i64[]", primals_383: "f32[768]", primals_384: "f32[768]", primals_385: "i64[]", primals_386: "f32[768]", primals_387: "f32[768]", primals_388: "i64[]", primals_389: "f32[768]", primals_390: "f32[768]", primals_391: "i64[]", primals_392: "f32[768]", primals_393: "f32[768]", primals_394: "i64[]", primals_395: "f32[768]", primals_396: "f32[768]", primals_397: "i64[]", primals_398: "f32[768]", primals_399: "f32[768]", primals_400: "i64[]", primals_401: "f32[768]", primals_402: "f32[768]", primals_403: "i64[]", primals_404: "f32[768]", primals_405: "f32[768]", primals_406: "i64[]", primals_407: "f32[768]", primals_408: "f32[768]", primals_409: "i64[]", primals_410: "f32[768]", primals_411: "f32[768]", primals_412: "i64[]", primals_413: "f32[768]", primals_414: "f32[768]", primals_415: "i64[]", primals_416: "f32[768]", primals_417: "f32[768]", primals_418: "i64[]", primals_419: "f32[768]", primals_420: "f32[768]", primals_421: "i64[]", primals_422: "f32[768]", primals_423: "f32[768]", primals_424: "i64[]", primals_425: "f32[768]", primals_426: "f32[768]", primals_427: "i64[]", primals_428: "f32[768]", primals_429: "f32[768]", primals_430: "i64[]", primals_431: "f32[768]", primals_432: "f32[768]", primals_433: "i64[]", primals_434: "f32[768]", primals_435: "f32[768]", primals_436: "i64[]", primals_437: "f32[768]", primals_438: "f32[768]", primals_439: "i64[]", primals_440: "f32[768]", primals_441: "f32[768]", primals_442: "i64[]", primals_443: "f32[768]", primals_444: "f32[768]", primals_445: "i64[]", primals_446: "f32[768]", primals_447: "f32[768]", primals_448: "i64[]", primals_449: "f32[768]", primals_450: "f32[768]", primals_451: "i64[]", primals_452: "f32[768]", primals_453: "f32[768]", primals_454: "i64[]", primals_455: "f32[768]", primals_456: "f32[768]", primals_457: "i64[]", primals_458: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:85, code: x = self.stem(x)
    convolution: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(primals_458, primals_1, primals_2, [7, 7], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None
    relu: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_265, 1)
    var_mean = torch.ops.aten.var_mean.correction(relu, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 768, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 768, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu, getitem_1);  relu = None
    mul: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[768]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
    add_2: "f32[768]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[768]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0001220852154804);  squeeze_2 = None
    mul_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[768]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_3: "f32[768]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_1: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_4, primals_5, primals_6, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_6 = None
    relu_1: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_268, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(relu_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 768, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 768, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_1, getitem_3);  relu_1 = None
    mul_7: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[768]" = torch.ops.aten.mul.Tensor(primals_266, 0.9)
    add_7: "f32[768]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0001220852154804);  squeeze_5 = None
    mul_11: "f32[768]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[768]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_8: "f32[768]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_5: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_7: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    add_10: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_9, add_4);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_2: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_10, primals_9, primals_10, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_10 = None
    relu_2: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_2)
    add_11: "i64[]" = torch.ops.aten.add.Tensor(primals_271, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(relu_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 768, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 768, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_12: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_2: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_2, getitem_5);  relu_2 = None
    mul_14: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[768]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
    add_13: "f32[768]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0001220852154804);  squeeze_8 = None
    mul_18: "f32[768]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[768]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_14: "f32[768]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_9: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_11: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_15: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_3: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_15, primals_13, primals_14, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_14 = None
    relu_3: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_3)
    add_16: "i64[]" = torch.ops.aten.add.Tensor(primals_274, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(relu_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 768, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 768, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_17: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_3: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_3, getitem_7);  relu_3 = None
    mul_21: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[768]" = torch.ops.aten.mul.Tensor(primals_272, 0.9)
    add_18: "f32[768]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0001220852154804);  squeeze_11 = None
    mul_25: "f32[768]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[768]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_19: "f32[768]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_13: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_15: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_20: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    add_21: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_20, add_15);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_4: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_21, primals_17, primals_18, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_18 = None
    relu_4: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_4)
    add_22: "i64[]" = torch.ops.aten.add.Tensor(primals_277, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(relu_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 768, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 768, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_23: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_4: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_4, getitem_9);  relu_4 = None
    mul_28: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[768]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_24: "f32[768]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0001220852154804);  squeeze_14 = None
    mul_32: "f32[768]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[768]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_25: "f32[768]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_17: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_19: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_26: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_5: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_26, primals_21, primals_22, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_22 = None
    relu_5: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_5)
    add_27: "i64[]" = torch.ops.aten.add.Tensor(primals_280, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(relu_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 768, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 768, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_28: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_5: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_5, getitem_11);  relu_5 = None
    mul_35: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[768]" = torch.ops.aten.mul.Tensor(primals_278, 0.9)
    add_29: "f32[768]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001220852154804);  squeeze_17 = None
    mul_39: "f32[768]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[768]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_30: "f32[768]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_21: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_23: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_31: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    add_32: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_31, add_26);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_6: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_32, primals_25, primals_26, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_26 = None
    relu_6: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_6)
    add_33: "i64[]" = torch.ops.aten.add.Tensor(primals_283, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(relu_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 768, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 768, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_34: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_6: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_6, getitem_13);  relu_6 = None
    mul_42: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[768]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
    add_35: "f32[768]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001220852154804);  squeeze_20 = None
    mul_46: "f32[768]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[768]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_36: "f32[768]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_25: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_27: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_37: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_7: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_37, primals_29, primals_30, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_30 = None
    relu_7: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_7)
    add_38: "i64[]" = torch.ops.aten.add.Tensor(primals_286, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(relu_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 768, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 768, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_39: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_7: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_7, getitem_15);  relu_7 = None
    mul_49: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[768]" = torch.ops.aten.mul.Tensor(primals_284, 0.9)
    add_40: "f32[768]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001220852154804);  squeeze_23 = None
    mul_53: "f32[768]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[768]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_41: "f32[768]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_29: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_31: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_42: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    add_43: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_42, add_37);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_8: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_43, primals_33, primals_34, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_34 = None
    relu_8: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_8)
    add_44: "i64[]" = torch.ops.aten.add.Tensor(primals_289, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(relu_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 768, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 768, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_45: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_8: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_8, getitem_17);  relu_8 = None
    mul_56: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[768]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_46: "f32[768]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001220852154804);  squeeze_26 = None
    mul_60: "f32[768]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[768]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_47: "f32[768]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_33: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_35: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_48: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_9: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_48, primals_37, primals_38, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_38 = None
    relu_9: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_9)
    add_49: "i64[]" = torch.ops.aten.add.Tensor(primals_292, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(relu_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 768, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 768, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_50: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_9: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_9, getitem_19);  relu_9 = None
    mul_63: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[768]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_51: "f32[768]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_66: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001220852154804);  squeeze_29 = None
    mul_67: "f32[768]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[768]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_52: "f32[768]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_37: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_39: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_53: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    add_54: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_53, add_48);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_10: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_54, primals_41, primals_42, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_42 = None
    relu_10: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_10)
    add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_295, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(relu_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 768, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 768, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_56: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_10: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_10, getitem_21);  relu_10 = None
    mul_70: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[768]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_57: "f32[768]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_73: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001220852154804);  squeeze_32 = None
    mul_74: "f32[768]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[768]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_58: "f32[768]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_41: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_43: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_59: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_11: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_59, primals_45, primals_46, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_46 = None
    relu_11: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_11)
    add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_298, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(relu_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 768, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 768, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_61: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_11: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_11, getitem_23);  relu_11 = None
    mul_77: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[768]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_62: "f32[768]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_80: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001220852154804);  squeeze_35 = None
    mul_81: "f32[768]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[768]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_63: "f32[768]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_45: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_47: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_64: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    add_65: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_64, add_59);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_12: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_65, primals_49, primals_50, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_50 = None
    relu_12: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_12)
    add_66: "i64[]" = torch.ops.aten.add.Tensor(primals_301, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(relu_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 768, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 768, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_67: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_12: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_12, getitem_25);  relu_12 = None
    mul_84: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[768]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_68: "f32[768]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_87: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001220852154804);  squeeze_38 = None
    mul_88: "f32[768]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[768]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_69: "f32[768]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_49: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_51: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_70: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_13: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_70, primals_53, primals_54, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_54 = None
    relu_13: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_13)
    add_71: "i64[]" = torch.ops.aten.add.Tensor(primals_304, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(relu_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 768, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 768, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_72: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_13: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_13, getitem_27);  relu_13 = None
    mul_91: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[768]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_73: "f32[768]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_94: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001220852154804);  squeeze_41 = None
    mul_95: "f32[768]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[768]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_74: "f32[768]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_53: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_55: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_75: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    add_76: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_75, add_70);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_14: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_76, primals_57, primals_58, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_58 = None
    relu_14: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_14)
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(relu_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 768, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 768, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_78: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_14: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_14, getitem_29);  relu_14 = None
    mul_98: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[768]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_79: "f32[768]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_101: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001220852154804);  squeeze_44 = None
    mul_102: "f32[768]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[768]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_80: "f32[768]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_57: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_59: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_81: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_15: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_81, primals_61, primals_62, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_62 = None
    relu_15: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_15)
    add_82: "i64[]" = torch.ops.aten.add.Tensor(primals_310, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(relu_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 768, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 768, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_83: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_15: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_15, getitem_31);  relu_15 = None
    mul_105: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[768]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_84: "f32[768]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_108: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001220852154804);  squeeze_47 = None
    mul_109: "f32[768]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[768]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_85: "f32[768]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_61: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_63: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_86: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    add_87: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_86, add_81);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_16: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_87, primals_65, primals_66, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_66 = None
    relu_16: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_16)
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(relu_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 768, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 768, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_89: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_16: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_16, getitem_33);  relu_16 = None
    mul_112: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[768]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_90: "f32[768]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_115: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001220852154804);  squeeze_50 = None
    mul_116: "f32[768]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[768]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_91: "f32[768]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_65: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_67: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_92: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_17: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_92, primals_69, primals_70, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_70 = None
    relu_17: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_17)
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_316, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(relu_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 768, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 768, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_94: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_17: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_17, getitem_35);  relu_17 = None
    mul_119: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[768]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_95: "f32[768]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_122: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001220852154804);  squeeze_53 = None
    mul_123: "f32[768]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[768]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_96: "f32[768]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_69: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_71: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_97: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    add_98: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_97, add_92);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_18: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_98, primals_73, primals_74, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_74 = None
    relu_18: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_18)
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_319, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(relu_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 768, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 768, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_100: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_18: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_18, getitem_37);  relu_18 = None
    mul_126: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[768]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_101: "f32[768]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_129: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001220852154804);  squeeze_56 = None
    mul_130: "f32[768]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[768]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_102: "f32[768]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_73: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_75: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_103: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_19: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_103, primals_77, primals_78, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_78 = None
    relu_19: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_19)
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_322, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(relu_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 768, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 768, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_105: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_19: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_19, getitem_39);  relu_19 = None
    mul_133: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[768]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_106: "f32[768]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_136: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_137: "f32[768]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[768]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_107: "f32[768]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_77: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_79: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_108: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    add_109: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_108, add_103);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_20: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_109, primals_81, primals_82, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_82 = None
    relu_20: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_20)
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_325, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(relu_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 768, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 768, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_111: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_20: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_20, getitem_41);  relu_20 = None
    mul_140: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[768]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_112: "f32[768]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_143: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001220852154804);  squeeze_62 = None
    mul_144: "f32[768]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[768]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_113: "f32[768]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_81: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_83: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_114: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_21: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_114, primals_85, primals_86, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_86 = None
    relu_21: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_21)
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_328, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(relu_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 768, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 768, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_116: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_21: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_21, getitem_43);  relu_21 = None
    mul_147: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[768]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_117: "f32[768]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_150: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001220852154804);  squeeze_65 = None
    mul_151: "f32[768]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[768]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_118: "f32[768]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_85: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_87: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_119: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    add_120: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_119, add_114);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_22: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_120, primals_89, primals_90, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_90 = None
    relu_22: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_22)
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_331, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(relu_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 768, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 768, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_122: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_22: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_22, getitem_45);  relu_22 = None
    mul_154: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[768]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_123: "f32[768]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_157: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001220852154804);  squeeze_68 = None
    mul_158: "f32[768]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[768]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_124: "f32[768]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_89: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_91: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_125: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_23: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_125, primals_93, primals_94, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_94 = None
    relu_23: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_23)
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_334, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(relu_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 768, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 768, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_127: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_23: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_23, getitem_47);  relu_23 = None
    mul_161: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[768]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_128: "f32[768]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_164: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001220852154804);  squeeze_71 = None
    mul_165: "f32[768]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[768]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_129: "f32[768]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_93: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_95: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_130: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    add_131: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_130, add_125);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_24: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_131, primals_97, primals_98, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_98 = None
    relu_24: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_24)
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_337, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(relu_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 768, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 768, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_133: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_24: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_24, getitem_49);  relu_24 = None
    mul_168: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[768]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_134: "f32[768]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_171: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001220852154804);  squeeze_74 = None
    mul_172: "f32[768]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[768]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_135: "f32[768]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_97: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_99: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_136: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_25: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_136, primals_101, primals_102, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_102 = None
    relu_25: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_25)
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_340, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(relu_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 768, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 768, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_138: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_25: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_25, getitem_51);  relu_25 = None
    mul_175: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[768]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_139: "f32[768]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_178: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001220852154804);  squeeze_77 = None
    mul_179: "f32[768]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[768]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_140: "f32[768]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_101: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_103: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_141: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    add_142: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_141, add_136);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_26: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_142, primals_105, primals_106, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_106 = None
    relu_26: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_26)
    add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_343, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(relu_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 768, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 768, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_144: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_26: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_26, getitem_53);  relu_26 = None
    mul_182: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[768]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_145: "f32[768]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_185: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001220852154804);  squeeze_80 = None
    mul_186: "f32[768]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[768]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_146: "f32[768]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_105: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_107: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_147: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_27: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_147, primals_109, primals_110, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_110 = None
    relu_27: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_27)
    add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_346, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(relu_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 768, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 768, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_149: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_27: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_27, getitem_55);  relu_27 = None
    mul_189: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[768]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_150: "f32[768]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_192: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001220852154804);  squeeze_83 = None
    mul_193: "f32[768]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[768]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_151: "f32[768]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_109: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_111: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_152: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    add_153: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_152, add_147);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_28: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_153, primals_113, primals_114, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_114 = None
    relu_28: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_28)
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(relu_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 768, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 768, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_155: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_28: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_28, getitem_57);  relu_28 = None
    mul_196: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[768]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_156: "f32[768]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_199: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001220852154804);  squeeze_86 = None
    mul_200: "f32[768]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[768]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_157: "f32[768]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_113: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_115: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_158: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_29: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_158, primals_117, primals_118, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_118 = None
    relu_29: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_29)
    add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_352, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(relu_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 768, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 768, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_160: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    sub_29: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_29, getitem_59);  relu_29 = None
    mul_203: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[768]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_161: "f32[768]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_206: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001220852154804);  squeeze_89 = None
    mul_207: "f32[768]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[768]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_162: "f32[768]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_117: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_119: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_163: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    add_164: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_163, add_158);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_30: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_164, primals_121, primals_122, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_122 = None
    relu_30: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_30)
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(relu_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 768, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 768, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_166: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_30: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_30, getitem_61);  relu_30 = None
    mul_210: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[768]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_167: "f32[768]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_213: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001220852154804);  squeeze_92 = None
    mul_214: "f32[768]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[768]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_168: "f32[768]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_121: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_123: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_169: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_31: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_169, primals_125, primals_126, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_126 = None
    relu_31: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_31)
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_358, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(relu_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 768, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 768, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_171: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_31: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_31, getitem_63);  relu_31 = None
    mul_217: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[768]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_172: "f32[768]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_220: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001220852154804);  squeeze_95 = None
    mul_221: "f32[768]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[768]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_173: "f32[768]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_125: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_127: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_174: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    add_175: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_174, add_169);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_32: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_175, primals_129, primals_130, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_130 = None
    relu_32: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_32)
    add_176: "i64[]" = torch.ops.aten.add.Tensor(primals_361, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(relu_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 768, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 768, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_177: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_32: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_32, getitem_65);  relu_32 = None
    mul_224: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[768]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_178: "f32[768]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_227: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001220852154804);  squeeze_98 = None
    mul_228: "f32[768]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[768]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_179: "f32[768]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_129: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_131: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_180: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_33: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_180, primals_133, primals_134, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_134 = None
    relu_33: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_33)
    add_181: "i64[]" = torch.ops.aten.add.Tensor(primals_364, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(relu_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 768, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 768, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_182: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_33: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_33, getitem_67);  relu_33 = None
    mul_231: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[768]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_183: "f32[768]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_234: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001220852154804);  squeeze_101 = None
    mul_235: "f32[768]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[768]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_184: "f32[768]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1)
    unsqueeze_133: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
    unsqueeze_135: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_185: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    add_186: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_185, add_180);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_34: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_186, primals_137, primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_138 = None
    relu_34: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_34)
    add_187: "i64[]" = torch.ops.aten.add.Tensor(primals_367, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(relu_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 768, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 768, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_188: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_34: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_34, getitem_69);  relu_34 = None
    mul_238: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[768]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_189: "f32[768]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_241: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001220852154804);  squeeze_104 = None
    mul_242: "f32[768]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[768]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_190: "f32[768]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1)
    unsqueeze_137: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1);  primals_140 = None
    unsqueeze_139: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_191: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_35: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_191, primals_141, primals_142, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_142 = None
    relu_35: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_35)
    add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_370, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(relu_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 768, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 768, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_193: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_35: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_35, getitem_71);  relu_35 = None
    mul_245: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[768]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_194: "f32[768]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_248: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001220852154804);  squeeze_107 = None
    mul_249: "f32[768]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[768]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_195: "f32[768]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_141: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_143: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_196: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    add_197: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_196, add_191);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_36: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_197, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    relu_36: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_36)
    add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_373, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(relu_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 768, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 768, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_199: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_36: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_36, getitem_73);  relu_36 = None
    mul_252: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[768]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_200: "f32[768]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_255: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001220852154804);  squeeze_110 = None
    mul_256: "f32[768]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[768]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_201: "f32[768]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1)
    unsqueeze_145: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
    unsqueeze_147: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_202: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_37: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_202, primals_149, primals_150, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_150 = None
    relu_37: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_37)
    add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_376, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(relu_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 768, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 768, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_204: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_37: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_37, getitem_75);  relu_37 = None
    mul_259: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[768]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_205: "f32[768]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_262: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001220852154804);  squeeze_113 = None
    mul_263: "f32[768]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[768]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_206: "f32[768]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1)
    unsqueeze_149: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1);  primals_152 = None
    unsqueeze_151: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_207: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    add_208: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_207, add_202);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_38: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_208, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    relu_38: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_38)
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_379, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(relu_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 768, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 768, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_210: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_38: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_38, getitem_77);  relu_38 = None
    mul_266: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[768]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_211: "f32[768]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_269: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001220852154804);  squeeze_116 = None
    mul_270: "f32[768]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[768]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_212: "f32[768]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_153: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_155: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_213: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_39: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_213, primals_157, primals_158, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_158 = None
    relu_39: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_39)
    add_214: "i64[]" = torch.ops.aten.add.Tensor(primals_382, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(relu_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 768, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 768, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_215: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    sub_39: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_39, getitem_79);  relu_39 = None
    mul_273: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[768]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_216: "f32[768]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_276: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0001220852154804);  squeeze_119 = None
    mul_277: "f32[768]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[768]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_217: "f32[768]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1)
    unsqueeze_157: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
    unsqueeze_159: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_218: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    add_219: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_218, add_213);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_40: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_219, primals_161, primals_162, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_162 = None
    relu_40: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_40)
    add_220: "i64[]" = torch.ops.aten.add.Tensor(primals_385, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(relu_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 768, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 768, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_221: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_40: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_40, getitem_81);  relu_40 = None
    mul_280: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[768]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_222: "f32[768]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_283: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0001220852154804);  squeeze_122 = None
    mul_284: "f32[768]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[768]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_223: "f32[768]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
    unsqueeze_161: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1);  primals_164 = None
    unsqueeze_163: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_224: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_41: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_224, primals_165, primals_166, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_166 = None
    relu_41: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_41)
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_388, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(relu_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 768, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 768, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_226: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_41: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_41, getitem_83);  relu_41 = None
    mul_287: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[768]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_227: "f32[768]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_290: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001220852154804);  squeeze_125 = None
    mul_291: "f32[768]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[768]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_228: "f32[768]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_165: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_167: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_229: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    add_230: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_229, add_224);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_42: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_230, primals_169, primals_170, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_170 = None
    relu_42: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_42)
    add_231: "i64[]" = torch.ops.aten.add.Tensor(primals_391, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(relu_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 768, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 768, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_232: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
    sub_42: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_42, getitem_85);  relu_42 = None
    mul_294: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[768]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_233: "f32[768]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_297: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0001220852154804);  squeeze_128 = None
    mul_298: "f32[768]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[768]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_234: "f32[768]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1)
    unsqueeze_169: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1);  primals_172 = None
    unsqueeze_171: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_235: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_43: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_235, primals_173, primals_174, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_174 = None
    relu_43: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_43)
    add_236: "i64[]" = torch.ops.aten.add.Tensor(primals_394, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(relu_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 768, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 768, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_237: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
    sub_43: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_43, getitem_87);  relu_43 = None
    mul_301: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[768]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_238: "f32[768]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_304: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0001220852154804);  squeeze_131 = None
    mul_305: "f32[768]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[768]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_239: "f32[768]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1)
    unsqueeze_173: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1);  primals_176 = None
    unsqueeze_175: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_240: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    add_241: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_240, add_235);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_44: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_241, primals_177, primals_178, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_178 = None
    relu_44: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_44)
    add_242: "i64[]" = torch.ops.aten.add.Tensor(primals_397, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(relu_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 768, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 768, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_243: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
    sub_44: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_44, getitem_89);  relu_44 = None
    mul_308: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[768]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_244: "f32[768]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_311: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0001220852154804);  squeeze_134 = None
    mul_312: "f32[768]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[768]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_245: "f32[768]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
    unsqueeze_177: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
    unsqueeze_179: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_246: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_45: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_246, primals_181, primals_182, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_182 = None
    relu_45: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_45)
    add_247: "i64[]" = torch.ops.aten.add.Tensor(primals_400, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(relu_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 768, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 768, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_248: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_45: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_45, getitem_91);  relu_45 = None
    mul_315: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[768]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_249: "f32[768]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_318: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0001220852154804);  squeeze_137 = None
    mul_319: "f32[768]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[768]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_250: "f32[768]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1)
    unsqueeze_181: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1);  primals_184 = None
    unsqueeze_183: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_251: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    add_252: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_251, add_246);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_46: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_252, primals_185, primals_186, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_186 = None
    relu_46: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_46)
    add_253: "i64[]" = torch.ops.aten.add.Tensor(primals_403, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(relu_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 768, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 768, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_254: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
    sub_46: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_46, getitem_93);  relu_46 = None
    mul_322: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[768]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_255: "f32[768]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_325: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0001220852154804);  squeeze_140 = None
    mul_326: "f32[768]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[768]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_256: "f32[768]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1)
    unsqueeze_185: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1);  primals_188 = None
    unsqueeze_187: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_257: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_47: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_257, primals_189, primals_190, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_190 = None
    relu_47: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_47)
    add_258: "i64[]" = torch.ops.aten.add.Tensor(primals_406, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(relu_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 768, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 768, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_259: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
    sub_47: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_47, getitem_95);  relu_47 = None
    mul_329: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[768]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_260: "f32[768]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_332: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0001220852154804);  squeeze_143 = None
    mul_333: "f32[768]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[768]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_261: "f32[768]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1)
    unsqueeze_189: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
    unsqueeze_191: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_262: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    add_263: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_262, add_257);  add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_48: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_263, primals_193, primals_194, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_194 = None
    relu_48: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_48)
    add_264: "i64[]" = torch.ops.aten.add.Tensor(primals_409, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(relu_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 768, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 768, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_265: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
    sub_48: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_48, getitem_97);  relu_48 = None
    mul_336: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[768]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_266: "f32[768]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_339: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0001220852154804);  squeeze_146 = None
    mul_340: "f32[768]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[768]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_267: "f32[768]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1)
    unsqueeze_193: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_196, -1);  primals_196 = None
    unsqueeze_195: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_268: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_49: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_268, primals_197, primals_198, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_198 = None
    relu_49: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_49)
    add_269: "i64[]" = torch.ops.aten.add.Tensor(primals_412, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(relu_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 768, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 768, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_270: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_270);  add_270 = None
    sub_49: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_49, getitem_99);  relu_49 = None
    mul_343: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[768]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_271: "f32[768]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_346: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0001220852154804);  squeeze_149 = None
    mul_347: "f32[768]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[768]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_272: "f32[768]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1)
    unsqueeze_197: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1);  primals_200 = None
    unsqueeze_199: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_273: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    add_274: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_273, add_268);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_50: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_274, primals_201, primals_202, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_202 = None
    relu_50: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_50)
    add_275: "i64[]" = torch.ops.aten.add.Tensor(primals_415, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(relu_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 768, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 768, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_276: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_50: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_50, getitem_101);  relu_50 = None
    mul_350: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[768]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_277: "f32[768]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_353: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0001220852154804);  squeeze_152 = None
    mul_354: "f32[768]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[768]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_278: "f32[768]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_201: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_203: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_279: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_51: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_279, primals_205, primals_206, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_206 = None
    relu_51: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_51)
    add_280: "i64[]" = torch.ops.aten.add.Tensor(primals_418, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(relu_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 768, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 768, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_281: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
    sub_51: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_51, getitem_103);  relu_51 = None
    mul_357: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[768]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
    add_282: "f32[768]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_360: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0001220852154804);  squeeze_155 = None
    mul_361: "f32[768]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[768]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_283: "f32[768]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1)
    unsqueeze_205: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_208, -1);  primals_208 = None
    unsqueeze_207: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_284: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    add_285: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_284, add_279);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_52: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_285, primals_209, primals_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_210 = None
    relu_52: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_52)
    add_286: "i64[]" = torch.ops.aten.add.Tensor(primals_421, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(relu_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 768, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 768, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_287: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_52: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
    sub_52: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_52, getitem_105);  relu_52 = None
    mul_364: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[768]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
    add_288: "f32[768]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_367: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0001220852154804);  squeeze_158 = None
    mul_368: "f32[768]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[768]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_289: "f32[768]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1)
    unsqueeze_209: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1);  primals_212 = None
    unsqueeze_211: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_290: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_53: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_290, primals_213, primals_214, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_214 = None
    relu_53: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_53)
    add_291: "i64[]" = torch.ops.aten.add.Tensor(primals_424, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(relu_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 768, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 768, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_292: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_53: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_292);  add_292 = None
    sub_53: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_53, getitem_107);  relu_53 = None
    mul_371: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[768]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
    add_293: "f32[768]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_374: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0001220852154804);  squeeze_161 = None
    mul_375: "f32[768]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[768]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_294: "f32[768]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_213: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_215: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_295: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    add_296: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_295, add_290);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_54: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_296, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    relu_54: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_54)
    add_297: "i64[]" = torch.ops.aten.add.Tensor(primals_427, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(relu_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 768, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 768, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_298: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_54: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_54: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_54, getitem_109);  relu_54 = None
    mul_378: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[768]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
    add_299: "f32[768]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_381: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0001220852154804);  squeeze_164 = None
    mul_382: "f32[768]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[768]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_300: "f32[768]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1)
    unsqueeze_217: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_220, -1);  primals_220 = None
    unsqueeze_219: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_301: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_55: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_301, primals_221, primals_222, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_222 = None
    relu_55: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_55)
    add_302: "i64[]" = torch.ops.aten.add.Tensor(primals_430, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(relu_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 768, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 768, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_303: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_55: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
    sub_55: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_55, getitem_111);  relu_55 = None
    mul_385: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[768]" = torch.ops.aten.mul.Tensor(primals_428, 0.9)
    add_304: "f32[768]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_388: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0001220852154804);  squeeze_167 = None
    mul_389: "f32[768]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[768]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
    add_305: "f32[768]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1)
    unsqueeze_221: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1);  primals_224 = None
    unsqueeze_223: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_306: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    add_307: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_306, add_301);  add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_56: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_307, primals_225, primals_226, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_226 = None
    relu_56: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_56)
    add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_433, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(relu_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 768, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 768, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_309: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_56: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
    sub_56: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_56, getitem_113);  relu_56 = None
    mul_392: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[768]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
    add_310: "f32[768]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_395: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0001220852154804);  squeeze_170 = None
    mul_396: "f32[768]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[768]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
    add_311: "f32[768]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_225: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_227: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_312: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_57: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_312, primals_229, primals_230, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_230 = None
    relu_57: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_57)
    add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_436, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(relu_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 768, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 768, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_314: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_57: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
    sub_57: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_57, getitem_115);  relu_57 = None
    mul_399: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_172: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[768]" = torch.ops.aten.mul.Tensor(primals_434, 0.9)
    add_315: "f32[768]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_402: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0001220852154804);  squeeze_173 = None
    mul_403: "f32[768]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[768]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
    add_316: "f32[768]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1)
    unsqueeze_229: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_232, -1);  primals_232 = None
    unsqueeze_231: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_317: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    add_318: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_317, add_312);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_58: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_318, primals_233, primals_234, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_234 = None
    relu_58: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_58)
    add_319: "i64[]" = torch.ops.aten.add.Tensor(primals_439, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(relu_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 768, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 768, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_320: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_58: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
    sub_58: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_58, getitem_117);  relu_58 = None
    mul_406: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_175: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[768]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
    add_321: "f32[768]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_409: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0001220852154804);  squeeze_176 = None
    mul_410: "f32[768]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[768]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
    add_322: "f32[768]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
    unsqueeze_233: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1);  primals_236 = None
    unsqueeze_235: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_323: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_59: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_323, primals_237, primals_238, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_238 = None
    relu_59: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_59)
    add_324: "i64[]" = torch.ops.aten.add.Tensor(primals_442, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(relu_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 768, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 768, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_325: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_59: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
    sub_59: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_59, getitem_119);  relu_59 = None
    mul_413: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_178: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[768]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
    add_326: "f32[768]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_416: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0001220852154804);  squeeze_179 = None
    mul_417: "f32[768]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[768]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
    add_327: "f32[768]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1)
    unsqueeze_237: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
    unsqueeze_239: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_328: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    add_329: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_328, add_323);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_60: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_329, primals_241, primals_242, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_242 = None
    relu_60: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_60)
    add_330: "i64[]" = torch.ops.aten.add.Tensor(primals_445, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(relu_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 768, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 768, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_331: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_60: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
    sub_60: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_60, getitem_121);  relu_60 = None
    mul_420: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_181: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[768]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
    add_332: "f32[768]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_423: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0001220852154804);  squeeze_182 = None
    mul_424: "f32[768]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[768]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
    add_333: "f32[768]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1)
    unsqueeze_241: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1);  primals_244 = None
    unsqueeze_243: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_334: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_61: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_334, primals_245, primals_246, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_246 = None
    relu_61: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_61)
    add_335: "i64[]" = torch.ops.aten.add.Tensor(primals_448, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(relu_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 768, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 768, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_336: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_61: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
    sub_61: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_61, getitem_123);  relu_61 = None
    mul_427: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_184: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[768]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
    add_337: "f32[768]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_430: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0001220852154804);  squeeze_185 = None
    mul_431: "f32[768]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[768]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
    add_338: "f32[768]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1)
    unsqueeze_245: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1);  primals_248 = None
    unsqueeze_247: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_339: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    add_340: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_339, add_334);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_62: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_340, primals_249, primals_250, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_250 = None
    relu_62: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_62)
    add_341: "i64[]" = torch.ops.aten.add.Tensor(primals_451, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(relu_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 768, 1, 1]" = var_mean_62[0]
    getitem_125: "f32[1, 768, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_342: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_62: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_342);  add_342 = None
    sub_62: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_62, getitem_125);  relu_62 = None
    mul_434: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_187: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[768]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
    add_343: "f32[768]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_437: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0001220852154804);  squeeze_188 = None
    mul_438: "f32[768]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[768]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
    add_344: "f32[768]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1)
    unsqueeze_249: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1);  primals_252 = None
    unsqueeze_251: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_345: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    convolution_63: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_345, primals_253, primals_254, [1, 1], [3, 3], [1, 1], False, [0, 0], 768);  primals_254 = None
    relu_63: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_63)
    add_346: "i64[]" = torch.ops.aten.add.Tensor(primals_454, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(relu_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 768, 1, 1]" = var_mean_63[0]
    getitem_127: "f32[1, 768, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_347: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_63: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_347);  add_347 = None
    sub_63: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_63, getitem_127);  relu_63 = None
    mul_441: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_190: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[768]" = torch.ops.aten.mul.Tensor(primals_452, 0.9)
    add_348: "f32[768]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_444: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0001220852154804);  squeeze_191 = None
    mul_445: "f32[768]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[768]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
    add_349: "f32[768]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1)
    unsqueeze_253: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1);  primals_256 = None
    unsqueeze_255: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_350: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    add_351: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(add_350, add_345);  add_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    convolution_64: "f32[8, 768, 32, 32]" = torch.ops.aten.convolution.default(add_351, primals_257, primals_258, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_258 = None
    relu_64: "f32[8, 768, 32, 32]" = torch.ops.aten.relu.default(convolution_64)
    add_352: "i64[]" = torch.ops.aten.add.Tensor(primals_457, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(relu_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 768, 1, 1]" = var_mean_64[0]
    getitem_129: "f32[1, 768, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_353: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_64: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_353);  add_353 = None
    sub_64: "f32[8, 768, 32, 32]" = torch.ops.aten.sub.Tensor(relu_64, getitem_129);  relu_64 = None
    mul_448: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_193: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[768]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
    add_354: "f32[768]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_451: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0001220852154804);  squeeze_194 = None
    mul_452: "f32[768]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[768]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
    add_355: "f32[768]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1)
    unsqueeze_257: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 768, 32, 32]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1);  primals_260 = None
    unsqueeze_259: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_356: "f32[8, 768, 32, 32]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 768, 1, 1]" = torch.ops.aten.mean.dim(add_356, [-1, -2], True);  add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 768]" = torch.ops.aten.reshape.default(mean, [8, 768]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:95, code: return x if pre_logits else self.head(x)
    permute: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_262, view, permute);  primals_262 = None
    permute_1: "f32[1000, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_260: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_261: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
    unsqueeze_262: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_272: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_273: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_284: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_285: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
    unsqueeze_286: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_296: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_297: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
    unsqueeze_298: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_308: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_309: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
    unsqueeze_310: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_320: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_321: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_332: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_333: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_344: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_345: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_356: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_357: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_368: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_369: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_380: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_381: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_392: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_393: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_404: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_405: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_416: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_417: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_428: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_429: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_440: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_441: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_452: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_453: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_464: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_465: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_476: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_477: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_488: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_489: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_500: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_501: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_512: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_513: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_524: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_525: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_536: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_537: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_548: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_549: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_560: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_561: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_572: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_573: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_584: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_585: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_596: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_597: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_608: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_609: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_620: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_621: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_632: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_633: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_644: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_645: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_656: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_657: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_668: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_669: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_680: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_681: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_692: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_693: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_704: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_705: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_716: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_717: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_728: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_729: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_740: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_741: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_752: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_753: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_764: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_765: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_776: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_777: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_788: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_789: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_800: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_801: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_812: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_813: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_824: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_825: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_836: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_837: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_848: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_849: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_860: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_861: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_872: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_873: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_884: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_885: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_896: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_897: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_908: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_909: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_920: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_921: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_932: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_933: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_944: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_945: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_956: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_957: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_968: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_969: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_980: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_981: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_992: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_993: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:89, code: x = self.blocks(x)
    unsqueeze_1004: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1005: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:22, code: return self.fn(x) + x
    unsqueeze_1016: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1017: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convmixer.py:85, code: x = self.stem(x)
    unsqueeze_1028: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1029: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[768]" = torch.ops.aten.copy_.default(primals_263, add_2);  primals_263 = add_2 = None
    copy__1: "f32[768]" = torch.ops.aten.copy_.default(primals_264, add_3);  primals_264 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_265, add);  primals_265 = add = None
    copy__3: "f32[768]" = torch.ops.aten.copy_.default(primals_266, add_7);  primals_266 = add_7 = None
    copy__4: "f32[768]" = torch.ops.aten.copy_.default(primals_267, add_8);  primals_267 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_268, add_5);  primals_268 = add_5 = None
    copy__6: "f32[768]" = torch.ops.aten.copy_.default(primals_269, add_13);  primals_269 = add_13 = None
    copy__7: "f32[768]" = torch.ops.aten.copy_.default(primals_270, add_14);  primals_270 = add_14 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_271, add_11);  primals_271 = add_11 = None
    copy__9: "f32[768]" = torch.ops.aten.copy_.default(primals_272, add_18);  primals_272 = add_18 = None
    copy__10: "f32[768]" = torch.ops.aten.copy_.default(primals_273, add_19);  primals_273 = add_19 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_274, add_16);  primals_274 = add_16 = None
    copy__12: "f32[768]" = torch.ops.aten.copy_.default(primals_275, add_24);  primals_275 = add_24 = None
    copy__13: "f32[768]" = torch.ops.aten.copy_.default(primals_276, add_25);  primals_276 = add_25 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_277, add_22);  primals_277 = add_22 = None
    copy__15: "f32[768]" = torch.ops.aten.copy_.default(primals_278, add_29);  primals_278 = add_29 = None
    copy__16: "f32[768]" = torch.ops.aten.copy_.default(primals_279, add_30);  primals_279 = add_30 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_280, add_27);  primals_280 = add_27 = None
    copy__18: "f32[768]" = torch.ops.aten.copy_.default(primals_281, add_35);  primals_281 = add_35 = None
    copy__19: "f32[768]" = torch.ops.aten.copy_.default(primals_282, add_36);  primals_282 = add_36 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_283, add_33);  primals_283 = add_33 = None
    copy__21: "f32[768]" = torch.ops.aten.copy_.default(primals_284, add_40);  primals_284 = add_40 = None
    copy__22: "f32[768]" = torch.ops.aten.copy_.default(primals_285, add_41);  primals_285 = add_41 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_286, add_38);  primals_286 = add_38 = None
    copy__24: "f32[768]" = torch.ops.aten.copy_.default(primals_287, add_46);  primals_287 = add_46 = None
    copy__25: "f32[768]" = torch.ops.aten.copy_.default(primals_288, add_47);  primals_288 = add_47 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_289, add_44);  primals_289 = add_44 = None
    copy__27: "f32[768]" = torch.ops.aten.copy_.default(primals_290, add_51);  primals_290 = add_51 = None
    copy__28: "f32[768]" = torch.ops.aten.copy_.default(primals_291, add_52);  primals_291 = add_52 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_292, add_49);  primals_292 = add_49 = None
    copy__30: "f32[768]" = torch.ops.aten.copy_.default(primals_293, add_57);  primals_293 = add_57 = None
    copy__31: "f32[768]" = torch.ops.aten.copy_.default(primals_294, add_58);  primals_294 = add_58 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_295, add_55);  primals_295 = add_55 = None
    copy__33: "f32[768]" = torch.ops.aten.copy_.default(primals_296, add_62);  primals_296 = add_62 = None
    copy__34: "f32[768]" = torch.ops.aten.copy_.default(primals_297, add_63);  primals_297 = add_63 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_298, add_60);  primals_298 = add_60 = None
    copy__36: "f32[768]" = torch.ops.aten.copy_.default(primals_299, add_68);  primals_299 = add_68 = None
    copy__37: "f32[768]" = torch.ops.aten.copy_.default(primals_300, add_69);  primals_300 = add_69 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_301, add_66);  primals_301 = add_66 = None
    copy__39: "f32[768]" = torch.ops.aten.copy_.default(primals_302, add_73);  primals_302 = add_73 = None
    copy__40: "f32[768]" = torch.ops.aten.copy_.default(primals_303, add_74);  primals_303 = add_74 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_304, add_71);  primals_304 = add_71 = None
    copy__42: "f32[768]" = torch.ops.aten.copy_.default(primals_305, add_79);  primals_305 = add_79 = None
    copy__43: "f32[768]" = torch.ops.aten.copy_.default(primals_306, add_80);  primals_306 = add_80 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_77);  primals_307 = add_77 = None
    copy__45: "f32[768]" = torch.ops.aten.copy_.default(primals_308, add_84);  primals_308 = add_84 = None
    copy__46: "f32[768]" = torch.ops.aten.copy_.default(primals_309, add_85);  primals_309 = add_85 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_310, add_82);  primals_310 = add_82 = None
    copy__48: "f32[768]" = torch.ops.aten.copy_.default(primals_311, add_90);  primals_311 = add_90 = None
    copy__49: "f32[768]" = torch.ops.aten.copy_.default(primals_312, add_91);  primals_312 = add_91 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_88);  primals_313 = add_88 = None
    copy__51: "f32[768]" = torch.ops.aten.copy_.default(primals_314, add_95);  primals_314 = add_95 = None
    copy__52: "f32[768]" = torch.ops.aten.copy_.default(primals_315, add_96);  primals_315 = add_96 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_316, add_93);  primals_316 = add_93 = None
    copy__54: "f32[768]" = torch.ops.aten.copy_.default(primals_317, add_101);  primals_317 = add_101 = None
    copy__55: "f32[768]" = torch.ops.aten.copy_.default(primals_318, add_102);  primals_318 = add_102 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_319, add_99);  primals_319 = add_99 = None
    copy__57: "f32[768]" = torch.ops.aten.copy_.default(primals_320, add_106);  primals_320 = add_106 = None
    copy__58: "f32[768]" = torch.ops.aten.copy_.default(primals_321, add_107);  primals_321 = add_107 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_322, add_104);  primals_322 = add_104 = None
    copy__60: "f32[768]" = torch.ops.aten.copy_.default(primals_323, add_112);  primals_323 = add_112 = None
    copy__61: "f32[768]" = torch.ops.aten.copy_.default(primals_324, add_113);  primals_324 = add_113 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_325, add_110);  primals_325 = add_110 = None
    copy__63: "f32[768]" = torch.ops.aten.copy_.default(primals_326, add_117);  primals_326 = add_117 = None
    copy__64: "f32[768]" = torch.ops.aten.copy_.default(primals_327, add_118);  primals_327 = add_118 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_328, add_115);  primals_328 = add_115 = None
    copy__66: "f32[768]" = torch.ops.aten.copy_.default(primals_329, add_123);  primals_329 = add_123 = None
    copy__67: "f32[768]" = torch.ops.aten.copy_.default(primals_330, add_124);  primals_330 = add_124 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_331, add_121);  primals_331 = add_121 = None
    copy__69: "f32[768]" = torch.ops.aten.copy_.default(primals_332, add_128);  primals_332 = add_128 = None
    copy__70: "f32[768]" = torch.ops.aten.copy_.default(primals_333, add_129);  primals_333 = add_129 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_334, add_126);  primals_334 = add_126 = None
    copy__72: "f32[768]" = torch.ops.aten.copy_.default(primals_335, add_134);  primals_335 = add_134 = None
    copy__73: "f32[768]" = torch.ops.aten.copy_.default(primals_336, add_135);  primals_336 = add_135 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_337, add_132);  primals_337 = add_132 = None
    copy__75: "f32[768]" = torch.ops.aten.copy_.default(primals_338, add_139);  primals_338 = add_139 = None
    copy__76: "f32[768]" = torch.ops.aten.copy_.default(primals_339, add_140);  primals_339 = add_140 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_340, add_137);  primals_340 = add_137 = None
    copy__78: "f32[768]" = torch.ops.aten.copy_.default(primals_341, add_145);  primals_341 = add_145 = None
    copy__79: "f32[768]" = torch.ops.aten.copy_.default(primals_342, add_146);  primals_342 = add_146 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_343, add_143);  primals_343 = add_143 = None
    copy__81: "f32[768]" = torch.ops.aten.copy_.default(primals_344, add_150);  primals_344 = add_150 = None
    copy__82: "f32[768]" = torch.ops.aten.copy_.default(primals_345, add_151);  primals_345 = add_151 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_346, add_148);  primals_346 = add_148 = None
    copy__84: "f32[768]" = torch.ops.aten.copy_.default(primals_347, add_156);  primals_347 = add_156 = None
    copy__85: "f32[768]" = torch.ops.aten.copy_.default(primals_348, add_157);  primals_348 = add_157 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_154);  primals_349 = add_154 = None
    copy__87: "f32[768]" = torch.ops.aten.copy_.default(primals_350, add_161);  primals_350 = add_161 = None
    copy__88: "f32[768]" = torch.ops.aten.copy_.default(primals_351, add_162);  primals_351 = add_162 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_352, add_159);  primals_352 = add_159 = None
    copy__90: "f32[768]" = torch.ops.aten.copy_.default(primals_353, add_167);  primals_353 = add_167 = None
    copy__91: "f32[768]" = torch.ops.aten.copy_.default(primals_354, add_168);  primals_354 = add_168 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_165);  primals_355 = add_165 = None
    copy__93: "f32[768]" = torch.ops.aten.copy_.default(primals_356, add_172);  primals_356 = add_172 = None
    copy__94: "f32[768]" = torch.ops.aten.copy_.default(primals_357, add_173);  primals_357 = add_173 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_358, add_170);  primals_358 = add_170 = None
    copy__96: "f32[768]" = torch.ops.aten.copy_.default(primals_359, add_178);  primals_359 = add_178 = None
    copy__97: "f32[768]" = torch.ops.aten.copy_.default(primals_360, add_179);  primals_360 = add_179 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_361, add_176);  primals_361 = add_176 = None
    copy__99: "f32[768]" = torch.ops.aten.copy_.default(primals_362, add_183);  primals_362 = add_183 = None
    copy__100: "f32[768]" = torch.ops.aten.copy_.default(primals_363, add_184);  primals_363 = add_184 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_364, add_181);  primals_364 = add_181 = None
    copy__102: "f32[768]" = torch.ops.aten.copy_.default(primals_365, add_189);  primals_365 = add_189 = None
    copy__103: "f32[768]" = torch.ops.aten.copy_.default(primals_366, add_190);  primals_366 = add_190 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_367, add_187);  primals_367 = add_187 = None
    copy__105: "f32[768]" = torch.ops.aten.copy_.default(primals_368, add_194);  primals_368 = add_194 = None
    copy__106: "f32[768]" = torch.ops.aten.copy_.default(primals_369, add_195);  primals_369 = add_195 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_370, add_192);  primals_370 = add_192 = None
    copy__108: "f32[768]" = torch.ops.aten.copy_.default(primals_371, add_200);  primals_371 = add_200 = None
    copy__109: "f32[768]" = torch.ops.aten.copy_.default(primals_372, add_201);  primals_372 = add_201 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_373, add_198);  primals_373 = add_198 = None
    copy__111: "f32[768]" = torch.ops.aten.copy_.default(primals_374, add_205);  primals_374 = add_205 = None
    copy__112: "f32[768]" = torch.ops.aten.copy_.default(primals_375, add_206);  primals_375 = add_206 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_376, add_203);  primals_376 = add_203 = None
    copy__114: "f32[768]" = torch.ops.aten.copy_.default(primals_377, add_211);  primals_377 = add_211 = None
    copy__115: "f32[768]" = torch.ops.aten.copy_.default(primals_378, add_212);  primals_378 = add_212 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_379, add_209);  primals_379 = add_209 = None
    copy__117: "f32[768]" = torch.ops.aten.copy_.default(primals_380, add_216);  primals_380 = add_216 = None
    copy__118: "f32[768]" = torch.ops.aten.copy_.default(primals_381, add_217);  primals_381 = add_217 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_382, add_214);  primals_382 = add_214 = None
    copy__120: "f32[768]" = torch.ops.aten.copy_.default(primals_383, add_222);  primals_383 = add_222 = None
    copy__121: "f32[768]" = torch.ops.aten.copy_.default(primals_384, add_223);  primals_384 = add_223 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_385, add_220);  primals_385 = add_220 = None
    copy__123: "f32[768]" = torch.ops.aten.copy_.default(primals_386, add_227);  primals_386 = add_227 = None
    copy__124: "f32[768]" = torch.ops.aten.copy_.default(primals_387, add_228);  primals_387 = add_228 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_388, add_225);  primals_388 = add_225 = None
    copy__126: "f32[768]" = torch.ops.aten.copy_.default(primals_389, add_233);  primals_389 = add_233 = None
    copy__127: "f32[768]" = torch.ops.aten.copy_.default(primals_390, add_234);  primals_390 = add_234 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_391, add_231);  primals_391 = add_231 = None
    copy__129: "f32[768]" = torch.ops.aten.copy_.default(primals_392, add_238);  primals_392 = add_238 = None
    copy__130: "f32[768]" = torch.ops.aten.copy_.default(primals_393, add_239);  primals_393 = add_239 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_394, add_236);  primals_394 = add_236 = None
    copy__132: "f32[768]" = torch.ops.aten.copy_.default(primals_395, add_244);  primals_395 = add_244 = None
    copy__133: "f32[768]" = torch.ops.aten.copy_.default(primals_396, add_245);  primals_396 = add_245 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_397, add_242);  primals_397 = add_242 = None
    copy__135: "f32[768]" = torch.ops.aten.copy_.default(primals_398, add_249);  primals_398 = add_249 = None
    copy__136: "f32[768]" = torch.ops.aten.copy_.default(primals_399, add_250);  primals_399 = add_250 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_400, add_247);  primals_400 = add_247 = None
    copy__138: "f32[768]" = torch.ops.aten.copy_.default(primals_401, add_255);  primals_401 = add_255 = None
    copy__139: "f32[768]" = torch.ops.aten.copy_.default(primals_402, add_256);  primals_402 = add_256 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_403, add_253);  primals_403 = add_253 = None
    copy__141: "f32[768]" = torch.ops.aten.copy_.default(primals_404, add_260);  primals_404 = add_260 = None
    copy__142: "f32[768]" = torch.ops.aten.copy_.default(primals_405, add_261);  primals_405 = add_261 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_406, add_258);  primals_406 = add_258 = None
    copy__144: "f32[768]" = torch.ops.aten.copy_.default(primals_407, add_266);  primals_407 = add_266 = None
    copy__145: "f32[768]" = torch.ops.aten.copy_.default(primals_408, add_267);  primals_408 = add_267 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_409, add_264);  primals_409 = add_264 = None
    copy__147: "f32[768]" = torch.ops.aten.copy_.default(primals_410, add_271);  primals_410 = add_271 = None
    copy__148: "f32[768]" = torch.ops.aten.copy_.default(primals_411, add_272);  primals_411 = add_272 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_412, add_269);  primals_412 = add_269 = None
    copy__150: "f32[768]" = torch.ops.aten.copy_.default(primals_413, add_277);  primals_413 = add_277 = None
    copy__151: "f32[768]" = torch.ops.aten.copy_.default(primals_414, add_278);  primals_414 = add_278 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_415, add_275);  primals_415 = add_275 = None
    copy__153: "f32[768]" = torch.ops.aten.copy_.default(primals_416, add_282);  primals_416 = add_282 = None
    copy__154: "f32[768]" = torch.ops.aten.copy_.default(primals_417, add_283);  primals_417 = add_283 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_418, add_280);  primals_418 = add_280 = None
    copy__156: "f32[768]" = torch.ops.aten.copy_.default(primals_419, add_288);  primals_419 = add_288 = None
    copy__157: "f32[768]" = torch.ops.aten.copy_.default(primals_420, add_289);  primals_420 = add_289 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_421, add_286);  primals_421 = add_286 = None
    copy__159: "f32[768]" = torch.ops.aten.copy_.default(primals_422, add_293);  primals_422 = add_293 = None
    copy__160: "f32[768]" = torch.ops.aten.copy_.default(primals_423, add_294);  primals_423 = add_294 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_424, add_291);  primals_424 = add_291 = None
    copy__162: "f32[768]" = torch.ops.aten.copy_.default(primals_425, add_299);  primals_425 = add_299 = None
    copy__163: "f32[768]" = torch.ops.aten.copy_.default(primals_426, add_300);  primals_426 = add_300 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_427, add_297);  primals_427 = add_297 = None
    copy__165: "f32[768]" = torch.ops.aten.copy_.default(primals_428, add_304);  primals_428 = add_304 = None
    copy__166: "f32[768]" = torch.ops.aten.copy_.default(primals_429, add_305);  primals_429 = add_305 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_430, add_302);  primals_430 = add_302 = None
    copy__168: "f32[768]" = torch.ops.aten.copy_.default(primals_431, add_310);  primals_431 = add_310 = None
    copy__169: "f32[768]" = torch.ops.aten.copy_.default(primals_432, add_311);  primals_432 = add_311 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_433, add_308);  primals_433 = add_308 = None
    copy__171: "f32[768]" = torch.ops.aten.copy_.default(primals_434, add_315);  primals_434 = add_315 = None
    copy__172: "f32[768]" = torch.ops.aten.copy_.default(primals_435, add_316);  primals_435 = add_316 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_436, add_313);  primals_436 = add_313 = None
    copy__174: "f32[768]" = torch.ops.aten.copy_.default(primals_437, add_321);  primals_437 = add_321 = None
    copy__175: "f32[768]" = torch.ops.aten.copy_.default(primals_438, add_322);  primals_438 = add_322 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_439, add_319);  primals_439 = add_319 = None
    copy__177: "f32[768]" = torch.ops.aten.copy_.default(primals_440, add_326);  primals_440 = add_326 = None
    copy__178: "f32[768]" = torch.ops.aten.copy_.default(primals_441, add_327);  primals_441 = add_327 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_442, add_324);  primals_442 = add_324 = None
    copy__180: "f32[768]" = torch.ops.aten.copy_.default(primals_443, add_332);  primals_443 = add_332 = None
    copy__181: "f32[768]" = torch.ops.aten.copy_.default(primals_444, add_333);  primals_444 = add_333 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_445, add_330);  primals_445 = add_330 = None
    copy__183: "f32[768]" = torch.ops.aten.copy_.default(primals_446, add_337);  primals_446 = add_337 = None
    copy__184: "f32[768]" = torch.ops.aten.copy_.default(primals_447, add_338);  primals_447 = add_338 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_448, add_335);  primals_448 = add_335 = None
    copy__186: "f32[768]" = torch.ops.aten.copy_.default(primals_449, add_343);  primals_449 = add_343 = None
    copy__187: "f32[768]" = torch.ops.aten.copy_.default(primals_450, add_344);  primals_450 = add_344 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_451, add_341);  primals_451 = add_341 = None
    copy__189: "f32[768]" = torch.ops.aten.copy_.default(primals_452, add_348);  primals_452 = add_348 = None
    copy__190: "f32[768]" = torch.ops.aten.copy_.default(primals_453, add_349);  primals_453 = add_349 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_454, add_346);  primals_454 = add_346 = None
    copy__192: "f32[768]" = torch.ops.aten.copy_.default(primals_455, add_354);  primals_455 = add_354 = None
    copy__193: "f32[768]" = torch.ops.aten.copy_.default(primals_456, add_355);  primals_456 = add_355 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_457, add_352);  primals_457 = add_352 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_458, convolution, squeeze_1, add_4, convolution_1, squeeze_4, add_10, convolution_2, squeeze_7, add_15, convolution_3, squeeze_10, add_21, convolution_4, squeeze_13, add_26, convolution_5, squeeze_16, add_32, convolution_6, squeeze_19, add_37, convolution_7, squeeze_22, add_43, convolution_8, squeeze_25, add_48, convolution_9, squeeze_28, add_54, convolution_10, squeeze_31, add_59, convolution_11, squeeze_34, add_65, convolution_12, squeeze_37, add_70, convolution_13, squeeze_40, add_76, convolution_14, squeeze_43, add_81, convolution_15, squeeze_46, add_87, convolution_16, squeeze_49, add_92, convolution_17, squeeze_52, add_98, convolution_18, squeeze_55, add_103, convolution_19, squeeze_58, add_109, convolution_20, squeeze_61, add_114, convolution_21, squeeze_64, add_120, convolution_22, squeeze_67, add_125, convolution_23, squeeze_70, add_131, convolution_24, squeeze_73, add_136, convolution_25, squeeze_76, add_142, convolution_26, squeeze_79, add_147, convolution_27, squeeze_82, add_153, convolution_28, squeeze_85, add_158, convolution_29, squeeze_88, add_164, convolution_30, squeeze_91, add_169, convolution_31, squeeze_94, add_175, convolution_32, squeeze_97, add_180, convolution_33, squeeze_100, add_186, convolution_34, squeeze_103, add_191, convolution_35, squeeze_106, add_197, convolution_36, squeeze_109, add_202, convolution_37, squeeze_112, add_208, convolution_38, squeeze_115, add_213, convolution_39, squeeze_118, add_219, convolution_40, squeeze_121, add_224, convolution_41, squeeze_124, add_230, convolution_42, squeeze_127, add_235, convolution_43, squeeze_130, add_241, convolution_44, squeeze_133, add_246, convolution_45, squeeze_136, add_252, convolution_46, squeeze_139, add_257, convolution_47, squeeze_142, add_263, convolution_48, squeeze_145, add_268, convolution_49, squeeze_148, add_274, convolution_50, squeeze_151, add_279, convolution_51, squeeze_154, add_285, convolution_52, squeeze_157, add_290, convolution_53, squeeze_160, add_296, convolution_54, squeeze_163, add_301, convolution_55, squeeze_166, add_307, convolution_56, squeeze_169, add_312, convolution_57, squeeze_172, add_318, convolution_58, squeeze_175, add_323, convolution_59, squeeze_178, add_329, convolution_60, squeeze_181, add_334, convolution_61, squeeze_184, add_340, convolution_62, squeeze_187, add_345, convolution_63, squeeze_190, add_351, convolution_64, squeeze_193, view, permute_1, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030]
    